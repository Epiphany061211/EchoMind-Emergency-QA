"""
亮点：多 Agent 路由与编排（应急预案版）

Agent 角色：
  - ArchitectureAgent —— 组织架构与权限查询（查职能骨架，确定性查表）
  - PoliciesAgent     —— 预案条文检索与解读（ChromaDB policies 集合语义检索）
  - GeneralAgent      —— 通用接待与兜底（不是并列第三路，是兜底层）

路由策略（三层决策）：
  1. 紧急度覆盖 —— CRITICAL 直接升级
  2. 领域打分   —— 只给架构/条文两个方向打分，General 恒为兜底分
  3. 降级路由   —— 专属 Agent 失败时回退到 GeneralAgent

并行协作：
  - 复合问题（如"市公安局的职责是什么，三级响应时归谁指挥"）
    可同时派发给架构和条文两个 Agent，结果由 Composer 合并。
"""
import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from agents.tools import (
    AgentToolSpec,
    build_shared_rag_tools,
    escalation_tools,
    general_tools,
    architecture_tools,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


# ── 领域词表（单一常量源；后续三路融合的意图词表从这里扩展）───────────────────

# 组织架构方向：问关系、问职责、问权限 —— 答案是查表，不是检索
ARCH_KWS = [
    "上级", "平级", "下级", "职责", "分工", "归谁", "谁管", "谁负责",
    "指挥部", "指挥长", "牵头", "成员单位", "工作组", "设在", "属于",
    "组织架构", "指挥体系", "谁决定", "谁批准", "谁提出", "权限",
    "启动响应", "响应由谁", "终止响应",
]

ESCALATION_KWS = ["转人工", "人工", "找领导", "投诉"]

# 预案条文方向：问措施、问流程、问要求 —— 答案是检索条文原文
POLICY_KWS = [
    "预案", "条文", "规定", "措施", "处置", "撤离", "转移", "安置",
    "物资", "预警", "监测", "避险", "灾后", "恢复重建", "保障",
    "流程", "怎么应对", "怎么做", "第几条",
]


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL      = "general"       # 通用接待（兜底）
    ARCHITECTURE = "architecture"  # 组织架构与权限查询
    POLICIES     = "policies"      # 预案条文检索与解读
    ESCALATION   = "escalation"    # 人工核验升级


@dataclass(frozen=True)
class AgentProfile:

    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


def _env_float(name: str, default: float) -> float:
    """读取可选浮点配置；错误配置不应阻塞服务启动。"""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法浮点配置 %s=%r", name, os.getenv(name))
        return default


def _env_int(name: str, default: int) -> int:
    """读取可选整数配置；错误配置不应阻塞服务启动。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法整数配置 %s=%r", name, os.getenv(name))
        return default


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级
    tools_used:  List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentCallResult:
    """一次 LLM 调用的请求级结果，不在共享 Agent 实例上保存状态。"""
    content: str
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)


class AgentCallError(RuntimeError):
    """保留失败前已经发生的工具调用，供请求 trace 记录。"""

    def __init__(self, message: str, tools_used: List[str], tool_traces: List[Dict[str, Any]]):
        super().__init__(message)
        self.tools_used = list(tools_used)
        self.tool_traces = list(tool_traces)


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    entities:    Dict[str, List[str]] = field(default_factory=dict)
    intent:      Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    urgency:     Optional[UrgencyLevel]   = None
    intent_confidence: float = 1.0
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0


@dataclass
class RoutingDecision:
    """一次请求的结构化路由决策。"""
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用、角色契约和统计。"""

    agent_type: AgentType
    system_prompt: str
    profile: AgentProfile

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        profile: Optional[AgentProfile] = None,
    ):
        self._client = client
        self.profile = profile or self.profile
        self._model  = self.profile.model or model
        self._skill_manager = skill_manager
        self.stats   = AgentStats()
        self._shared_tools: Dict[str, AgentToolSpec] = {}

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        """返回该角色真实可调用的工具白名单。"""
        return dict(self._shared_tools)

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        self._shared_tools = dict(tools or {})

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            call_result = await self._call_llm(req)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(call_result.content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=call_result.content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                tools_used=call_result.tools_used,
                tool_traces=call_result.tool_traces,
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            tools_used = ex.tools_used if isinstance(ex, AgentCallError) else []
            tool_traces = ex.tool_traces if isinstance(ex, AgentCallError) else []
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
                tools_used=list(tools_used),
                tool_traces=list(tool_traces),
            )

    async def _call_llm(self, req: Request) -> AgentCallResult:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        if req.entities:
            entities_text = json.dumps(req.entities, ensure_ascii=False)
            messages.append({"role": "user", "content": f"[结构化实体]\n{_clean(entities_text)}"})
            messages.append({"role": "assistant", "content": "好的，我会结合这些结构化实体处理。"})
        role_packet = self._build_role_packet(req)
        if role_packet:
            messages.append({"role": "user", "content": f"[角色输入契约]\n{_clean(role_packet)}"})
            messages.append({"role": "assistant", "content": "好的，我会按照该角色的输入和输出契约处理。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        tools = self.get_tools()
        tools_used: List[str] = []
        tool_traces: List[Dict[str, Any]] = []
        try:
            for _ in range(3):
                request_kwargs: Dict[str, Any] = {
                    "model": self._model,
                    "max_tokens": self.profile.max_tokens,
                    "temperature": self.profile.temperature,
                    "system": self._build_system_prompt(req),
                    "messages": messages,
                }
                if tools:
                    request_kwargs["tools"] = [
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "input_schema": spec.input_schema,
                        }
                        for spec in tools.values()
                    ]
                resp = await self._client.messages.create(**request_kwargs)
                tool_uses = [block for block in (resp.content or []) if self._block_type(block) == "tool_use"]
                if not tool_uses:
                    return AgentCallResult(
                        content=extract_text_content(resp.content),
                        tools_used=list(tools_used),
                        tool_traces=list(tool_traces),
                    )

                messages.append({"role": "assistant", "content": resp.content})
                tool_results = []
                for block in tool_uses:
                    name = self._block_value(block, "name")
                    tool_use_id = self._block_value(block, "id")
                    args = self._block_value(block, "input") or {}
                    spec = tools.get(name)
                    tool_t0 = time.monotonic()
                    call_success = True
                    result_success: Optional[bool] = None
                    error_text = ""
                    if spec is None:
                        call_success = False
                        result: Any = {"success": False, "error": f"工具不在 {self.agent_type.value} Agent 白名单中"}
                        error_text = result["error"]
                    else:
                        try:
                            self._validate_tool_input(spec, args)
                            result = spec.handler(req, args)
                            if inspect.isawaitable(result):
                                result = await result
                            tools_used.append(name)
                            if isinstance(result, dict) and "success" in result:
                                result_success = bool(result.get("success"))
                        except Exception as ex:
                            call_success = False
                            logger.warning("Agent 工具 %s 执行失败: %s", name, ex)
                            error_text = str(ex)
                            result = {"success": False, "error": error_text}
                    tool_latency_ms = (time.monotonic() - tool_t0) * 1000
                    if not error_text and isinstance(result, dict):
                        error_text = str(result.get("error", "") or "")
                    tool_traces.append(
                        {
                            "agent_type": self.agent_type.value,
                            "tool_name": name,
                            "tool_use_id": tool_use_id,
                            "input": dict(args),
                            "success": call_success,
                            "result_success": result_success,
                            "latency_ms": round(tool_latency_ms, 1),
                            "cached": bool(result.get("cached")) if isinstance(result, dict) else False,
                            "reranked": bool(result.get("reranked")) if isinstance(result, dict) else False,
                            "error": error_text,
                        }
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                messages.append({"role": "user", "content": tool_results})

            raise AgentCallError(
                f"{self.agent_type.value} 工具调用超过最大轮数",
                tools_used,
                tool_traces,
            )
        except AgentCallError:
            raise
        except Exception as ex:
            raise AgentCallError(str(ex), tools_used, tool_traces) from ex

    @staticmethod
    def _block_type(block: Any) -> Optional[str]:
        if isinstance(block, dict):
            return block.get("type")
        return getattr(block, "type", None)

    @staticmethod
    def _block_value(block: Any, key: str) -> Any:
        if isinstance(block, dict):
            return block.get(key)
        return getattr(block, key, None)

    @staticmethod
    def _validate_tool_input(spec: AgentToolSpec, args: Any) -> None:
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        schema = spec.input_schema
        for field_name in schema.get("required", []):
            if field_name not in args:
                raise ValueError(f"缺少必需参数: {field_name}")
        properties = schema.get("properties", {})
        unknown = set(args) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"不允许的工具参数: {', '.join(sorted(unknown))}")
        type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
        for key, value in args.items():
            expected = properties.get(key, {}).get("type")
            if expected in type_map and not isinstance(value, type_map[expected]):
                raise ValueError(f"参数 {key} 类型错误，期望 {expected}")

    def _build_system_prompt(self, req: Request) -> str:
        """把角色契约和动态 Skills 拼入 system prompt。"""
        profile_prompt = (
            f"\n\n[角色契约]\n"
            f"角色：{self.profile.role}\n"
            f"职责：{self.profile.mission}\n"
            f"处理流程：{' -> '.join(self.profile.workflow)}\n"
            f"可用输入：{'；'.join(self.profile.input_contract)}\n"
            f"输出要求：{'；'.join(self.profile.output_contract)}\n"
            f"升级条件：{'；'.join(self.profile.handoff_conditions) or '无，按通用规则处理'}\n"
            f"允许的数据/工具范围：{'、'.join(self.profile.tool_scope) or '仅使用当前请求上下文'}\n"
            "不要声称执行了未提供的查询；工具没有返回的事实不要编造；"
            "引用职责、条文时保留预案名称和页码。"
        )
        base_prompt = f"{self.system_prompt}{profile_prompt}"
        if self._skill_manager is None:
            return base_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return base_prompt
        return f"{base_prompt}\n\n[动态 Skills]\n{skill_prompt}"

    def _build_role_packet(self, req: Request) -> str:
        """给子 Agent 的确定性输入包；子类可补充领域字段。"""
        packet = {
            "agent_type": self.agent_type.value,
            "intent": req.intent.value if req.intent else None,
            "intent_group": req.intent_group,
            "urgency": req.urgency.name if req.urgency else None,
            "intent_confidence": round(req.intent_confidence, 4),
            "available_entities": req.entities or {},
        }
        return json.dumps(packet, ensure_ascii=False)

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议人工核验（简单关键词检测）。"""
        keywords = ["转人工", "人工确认", "人工核验", "上报", "无法处理"]
        return any(kw in content for kw in keywords)


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    profile = AgentProfile(
        role="应急咨询接待与分流",
        mission="接待应急管理体系用户，快速判断问题是查组织架构还是查预案条文，"
                "信息不足时只询问必要字段。",
        workflow=("复述诉求", "判断属于架构查询还是条文查询", "直接回答或引导补充", "给出下一步"),
        input_contract=("对话历史", "意图与紧急度", "知识库上下文"),
        output_contract=("先回应核心问题", "信息不足时只询问必要字段", "明确下一步和边界"),
        handoff_conditions=("涉及灾情研判、响应启动决策等须由授权人员确认的事项", "用户明确要求人工"),
        tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields"),
        temperature=0.3,
        max_tokens=900,
    )
    system_prompt = (
        "你是应急预案智能问答系统的通用接待。用户是应急管理体系的工作人员，"
        "常见问题是查自己单位的上级/平级/下级和职责，或查预案条文规定。"
        "回答简洁、准确；超出能力范围时明确说明。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["triage_targets"] = ["architecture", "policies", "escalation"]
        packet["response_mode"] = "answer_or_clarify"
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(general_tools())
        return tools


class ArchitectureAgent(BaseAgent):
    agent_type    = AgentType.ARCHITECTURE
    profile = AgentProfile(
        role="组织架构与权限查询",
        mission="基于两份应急预案的职能骨架，回答单位/岗位的上级、平级、下级、"
                "职责与响应等级权限；所有结论来自骨架查询工具，可回溯预案页码。",
        workflow=("解析用户说的单位或岗位名称", "调用骨架查询工具", "按工具结果组织回答并标注页码",
                  "查不到时给出候选名称或建议核对预案原文"),
        input_contract=("单位或岗位名称", "响应等级", "知识库上下文"),
        output_contract=("直接给出关系或职责清单", "每条结论标注预案页码",
                         "查不到时不编造，给出候选或核验建议"),
        handoff_conditions=("工具结果与预案原文可能不一致需人工核验", "用户质疑结论准确性"),
        tool_scope=("query_superior", "query_subordinates", "query_peers",
                    "get_duties", "get_response_authority", "search_knowledge_base"),
        temperature=0.1,
        max_tokens=1200,
    )
    system_prompt = (
        "你是应急组织架构专家。组织架构类问题（上级/平级/下级/职责/响应权限）"
        "必须调用骨架查询工具拿结果，禁止凭印象回答单位之间的关系。"
        "工具返回 success=false 时如实告知，并把候选名称列给用户。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["architecture_fields"] = {
            "unit_hint": "从用户消息中提取单位或岗位名称作为工具的 unit 参数",
            "level_hint": "涉及响应启动/终止权限时用 get_response_authority，参数为 一级/二级/三级/四级",
            "risk_boundary": "不得编造组织关系；骨架是预案派生视图，重大用途提示核对预案原文",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(architecture_tools())
        return tools


class PoliciesAgent(BaseAgent):
    agent_type    = AgentType.POLICIES
    profile = AgentProfile(
        role="预案条文检索与解读",
        mission="在预案条文库中检索相关条款，用原文语义回答处置措施、响应流程、"
                "保障要求类问题，注明预案名称和页码。",
        workflow=("提炼检索关键词", "调用条文检索工具", "按命中片段组织回答",
                  "检索不到时如实说明"),
        input_contract=("用户问题", "检索关键词", "知识库上下文"),
        output_contract=("引用条文要点并注明来源（预案名+页码）", "不脱离检索结果发挥"),
        handoff_conditions=("条文解释涉及执法或责任认定", "检索结果相互矛盾需人工核对"),
        tool_scope=("search_knowledge_base",),
        temperature=0.2,
        max_tokens=1100,
    )
    system_prompt = (
        "你是应急预案条文专家。条文类问题必须先检索再回答，"
        "回答基于检索到的预案片段；没有命中就说没有找到相关规定，"
        "不要编造条款编号或内容。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["policy_fields"] = {
            "query_hint": "把用户问题改写成适合检索的关键词串再调用 search_knowledge_base",
            "risk_boundary": "不得编造条文；引用时保留预案名称和页码",
        }
        return json.dumps(packet, ensure_ascii=False)


class EscalationAgent(BaseAgent):
    """人工核验升级节点。

    升级不是一个普通问答 Prompt：它应该生成标准化的交接信息并停止普通
    Agent 继续编造答案。生产环境可在这里接值班系统或人工队列。
    """

    agent_type = AgentType.ESCALATION
    profile = AgentProfile(
        role="人工核验升级与交接",
        mission="确认需要人工核验的原因，整理已知上下文，告知用户下一步，"
                "不执行未经授权的操作。",
        workflow=("确认升级原因", "整理已知信息", "标记优先级", "生成交接摘要"),
        input_contract=("用户消息", "意图", "紧急度", "结构化实体", "对话背景"),
        output_contract=("升级原因", "已知信息摘要", "还需补充的信息", "保守的后续说明"),
        handoff_conditions=("用户明确要求人工", "紧急或高风险场景"),
        tool_scope=("search_knowledge_base", "create_handoff_summary"),
        temperature=0.0,
        max_tokens=500,
    )
    system_prompt = "你负责应急咨询的人工核验交接，不要继续模拟已完成的查询或决策。"

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(escalation_tools())
        return tools

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        intent = req.intent.value if req.intent else "unknown"
        urgency = req.urgency.name if req.urgency else "UNKNOWN"
        entities = json.dumps(req.entities or {}, ensure_ascii=False)
        content = (
            "这个问题已标记为需要人工核验。\n\n"
            f"升级原因：意图={intent}，紧急度={urgency}\n"
            f"已记录信息：{entities}\n"
            "涉及灾情研判和响应启动决策的内容，请由预案涉及的授权人员"
            "对照预案原文进一步确认；本系统的骨架数据是预案的派生视图，"
            "正式用途以预案原文为准。"
        )
        ms = (time.monotonic() - t0) * 1000
        self.stats.success += 1
        self.stats.total_ms += ms
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            success=True,
            latency_ms=ms,
            escalate=True,
            tools_used=[],
        )


class ResponseComposer:
    """多 Agent 汇总节点，统一主次、去重和输出边界。"""

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model = model
        self._skill_manager = skill_manager

    async def compose(self, req: Request, responses: List[AgentResponse]) -> str:
        successful = [response for response in responses if response.success and response.content.strip()]
        if not successful:
            return "抱歉，所有 Agent 均处理失败。"
        if len(successful) == 1:
            return successful[0].content

        evidence = "\n\n".join(
            f"[{response.agent_type.value} Agent 输出]\n{response.content}"
            for response in successful
        )
        prompt = (
            "你是应急预案问答系统的 Response Composer，负责把多个专业 Agent 的结果合并成一条最终回复。\n"
            "要求：以主 Agent 的结论为主，按用户问题优先级组织内容；去掉重复和冲突表述；"
            "不能补造组织关系、条文内容或响应权限结论；如果结论冲突，明确说明需要核对预案原文；"
            "保留必要的页码引用和核验边界。只输出给用户看的中文回复，不要提及 Agent。\n\n"
            f"主 Agent：{successful[0].agent_type.value}\n"
            f"用户问题：{req.message}\n"
            f"候选结果：\n{evidence}"
        )
        if self._skill_manager is not None:
            skill = self._skill_manager.prompt_for(req.message, "general")
            if skill:
                prompt += f"\n\n[通用输出边界]\n{skill}"
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=_env_int("ECHOMIND_COMPOSER_MAX_TOKENS", 1000),
                temperature=_env_float("ECHOMIND_COMPOSER_TEMPERATURE", 0.1),
                messages=[{"role": "user", "content": prompt}],
            )
            content = extract_text_content(response.content).strip()
            if content:
                return content
        except Exception as ex:
            logger.warning("Response Composer 失败，使用确定性合并: %s", ex)

        # 汇总节点不可用时保留主次标签，避免丢失某个专业 Agent 的结论。
        return "\n\n".join(
            f"{response.content}" if index == 0 else f"补充说明：\n{response.content}"
            for index, response in enumerate(successful)
        )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑（三层）：
      1. 紧急度/升级意图直达 EscalationAgent
      2. 领域关键词打分：只给架构、条文两个方向打分，高分者接手
      3. 都无命中时 GeneralAgent 兜底（General 不是并列第三路）
    """

    # 意图 → Agent 类型的静态映射（只保留升级类；领域分流交给 _domain_scores）
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.ARCH_QUERY: AgentType.ARCHITECTURE,
        IntentCategory.POLICY_QUERY: AgentType.POLICIES,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        IntentCategory.HUMAN_HANDOFF: AgentType.ESCALATION,
        # 其余意图一律进入领域打分，无命中则由 GENERAL 兜底
    }

    # Monitor 的降权达到该值时，专业 Agent 暂停接收新请求并回退到 GeneralAgent。
    MONITOR_FALLBACK_PENALTY = 0.5

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        rag_tool_manager: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)


        # 子 Agent 统一走 DeepSeek（Anthropic 兼容端点，SDK 零改动）；
        # 编排器层面（意图识别、回答组装）仍用 kimi。
        ds_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if ds_key:
            sub_client = AsyncAnthropic(
                api_key=ds_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic").strip(),
            )
            sub_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
        else:
            sub_client, sub_model = client, model

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager
        self._composer = ResponseComposer(client, model, skill_manager)
        self._shared_tools: Dict[str, AgentToolSpec] = {}
        self._recent_tool_traces = deque(maxlen=_env_int("ECHOMIND_TOOL_TRACE_MAX", 200))

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [self._make_agent(GeneralAgent, sub_client, sub_model, skill_manager)],
            AgentType.ARCHITECTURE: [self._make_agent(ArchitectureAgent, sub_client, sub_model, skill_manager)],
            AgentType.POLICIES: [self._make_agent(PoliciesAgent, sub_client, sub_model, skill_manager)],
            AgentType.ESCALATION: [self._make_agent(EscalationAgent, sub_client, sub_model, skill_manager)],
        }
        self.set_shared_tools(build_shared_rag_tools(rag_tool_manager))

    @staticmethod
    def _make_agent(
        agent_cls: type[BaseAgent],
        client: AsyncAnthropic,
        default_model: str,
        skill_manager: Optional[Any],
    ) -> BaseAgent:
        """按角色创建 Agent，并允许用环境变量覆盖该角色的模型。"""
        profile = agent_cls.profile
        env_name = f"ECHOMIND_{agent_cls.agent_type.value.upper()}_MODEL"
        model = os.getenv(env_name, "").strip() or profile.model
        configured_profile = replace(profile, model=model) if model else profile
        return agent_cls(client, default_model, skill_manager, profile=configured_profile)

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        self._composer._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        """更新所有 Agent 共享的工具白名单。"""
        self._shared_tools = dict(tools or {})
        for agents in self._pool.values():
            for agent in agents:
                agent.set_shared_tools(self._shared_tools)

    async def recognize_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """对外暴露意图识别，供 API 层先判断是否需要 RAG 等前置能力。"""
        return await self._intent_recognizer.recognize(message, history=history)

    def _record_tool_trace(self, result: OrchestratorResult) -> None:
        trace = {
            "request_id": result.request_id,
            "timestamp": datetime.now().isoformat(),
            "intent": result.intent.value if result.intent else None,
            "primary_agent": result.primary_agent.value if result.primary_agent else None,
            "supporting_agents": [agent.value for agent in result.supporting_agents],
            "tools_used": list(result.tools_used),
            "tool_calls": list(result.tool_traces),
            "escalated": result.escalated,
            "latency_ms": round(result.latency_ms, 1),
        }
        self._recent_tool_traces.append(trace)

    def get_tool_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        for trace in reversed(self._recent_tool_traces):
            if trace.get("request_id") == request_id:
                return trace
        return None

    def get_recent_tool_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._recent_tool_traces:
            return []
        limit = max(1, min(int(limit or 20), len(self._recent_tool_traces)))
        return list(reversed(list(self._recent_tool_traces)[-limit:]))

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.intent_group = intent_result.intent_group
            req.urgency = intent_result.urgency
            req.intent_confidence = intent_result.confidence

        if self._needs_clarification(req):
            result = OrchestratorResult(
                request_id=req.request_id,
                response=(
                    "我还不能确定您想查什么。请补充一下："
                    "是想查某个单位/岗位的组织架构（上级、平级、下级、职责、响应权限），"
                    "还是想查预案条文（处置措施、响应流程、保障要求）？"
                ),
                agent_type=AgentType.GENERAL,
                intent=req.intent,
                escalated=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                agent_types=[AgentType.GENERAL],
                primary_agent=AgentType.GENERAL,
                routing_reason="低置信度 OTHER 意图，先澄清用户需求",
                routing_confidence=req.intent_confidence,
            )
            self._record_tool_trace(result)
            return result

        # 复合问题自动并行协作，例如"市公安局的职责是什么，三级响应时归谁指挥"。
        decision = self._route_decision(req)
        if decision.multi_agent:
            return await self.run_parallel(req, decision)

        # 2. 执行主 Agent（含降级）
        response = await self._execute(req, decision.primary_agent)

        # 3. 升级检查
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent in (
            IntentCategory.ESCALATION,
            IntentCategory.HUMAN_HANDOFF,
        ):
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
            # 生产环境：此处接值班系统、通知人工核验

        result = OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[response.agent_type],
            primary_agent=decision.primary_agent,
            supporting_agents=[],
            tools_used=list(response.tools_used),
            tool_traces=list(response.tool_traces),
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
        )
        self._record_tool_trace(result)
        return result

    async def run_parallel(self, req: Request, decision: RoutingDecision) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复合问题（同时涉及组织架构和预案条文）。
        """
        t0 = time.monotonic()
        agent_types = decision.agent_types
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = [r for r in responses if isinstance(r, AgentResponse)]
        combined = await self._composer.compose(req, valid_responses)
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)
        tools_used = list(dict.fromkeys(
            tool_name
            for response in valid_responses
            for tool_name in response.tools_used
        ))
        tool_traces = [
            trace
            for response in valid_responses
            for trace in response.tool_traces
        ]
        result = OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=decision.primary_agent,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[
                r.agent_type for r in responses
                if isinstance(r, AgentResponse) and r.success
            ] or agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=decision.supporting_agents,
            tools_used=tools_used,
            tool_traces=tool_traces,
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
        )
        self._record_tool_trace(result)
        return result

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 紧急度覆盖（CRITICAL 直接升级）
          2. 升级意图映射
          3. 默认 GENERAL
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _route_decision(self, req: Request) -> RoutingDecision:
        """
        结构化路由决策。

        先处理紧急/人工核验，再用领域分数决定主 Agent 和辅助 Agent。
        这样可以表达"主处理 + 辅助查询"，避免关键词命中后无主次地拼接。
        """
        if req.urgency == UrgencyLevel.CRITICAL:
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason="紧急度为 CRITICAL，触发升级路由",
                confidence=1.0,
            )

        if req.intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            if any(kw in req.message for kw in ESCALATION_KWS):
                return AgentType.ESCALATION, "用户明确要求人工核验/升级"
        if req.intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[req.intent]
            if self._pool.get(target):
                return RoutingDecision(
                    primary_agent=target,
                    reason=f"意图直通: {req.intent.value} → {target.value}",
                    confidence=0.9,
                )
        scores = self._domain_scores(req)
        available_scores = {
            agent_type: score
            for agent_type, score in scores.items()
            if agent_type == AgentType.GENERAL or self._pool.get(agent_type)
        }
        if not available_scores:
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                reason="无可用专属 Agent，降级到 GeneralAgent",
                confidence=0.1,
            )

        ordered = sorted(available_scores.items(), key=lambda item: item[1], reverse=True)
        primary_agent, primary_score = ordered[0]

        collaboration_targets = self._collaboration_targets(req)
        supporting_agents = [
            agent_type
            for agent_type in collaboration_targets
            if agent_type != primary_agent and agent_type in available_scores
        ]

        if not supporting_agents:
            supporting_agents = [
                agent_type
                for agent_type, score in ordered[1:]
                if agent_type != AgentType.GENERAL
                and score >= 0.45
                and score >= primary_score * 0.55
            ]

        reason = self._routing_reason(req, available_scores, primary_agent, supporting_agents)

        return RoutingDecision(
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            reason=reason,
            confidence=round(min(primary_score, 1.0), 3),
        )

    def _domain_scores(self, req: Request) -> Dict[AgentType, float]:
        """
        按领域关键词为各专业 Agent 打分。

        General 恒为兜底分（0.1），不参与关键词加分——
        它是兜底层，不是与架构/条文并列的第三路。
        """
        msg = req.message.lower()
        scores = {
            AgentType.GENERAL: 0.1,
            AgentType.ARCHITECTURE: 0.0,
            AgentType.POLICIES: 0.0,
        }

        arch_hits = sum(1 for kw in ARCH_KWS if kw in msg)
        policy_hits = sum(1 for kw in POLICY_KWS if kw in msg)

        scores[AgentType.ARCHITECTURE] += min(0.8, arch_hits * 0.25)
        scores[AgentType.POLICIES] += min(0.8, policy_hits * 0.25)

        # 通用咨询/问候类意图给 General 小幅加分，但不足以盖过任何领域命中
        if req.intent in (
            IntentCategory.GREETING,
            IntentCategory.FEEDBACK,
            IntentCategory.OTHER,
        ):
            scores[AgentType.GENERAL] += 0.2

        return {agent_type: round(score, 3) for agent_type, score in scores.items()}

    @staticmethod
    def _routing_reason(
        req: Request,
        scores: Dict[AgentType, float],
        primary_agent: AgentType,
        supporting_agents: List[AgentType],
    ) -> str:
        score_text = ", ".join(
            f"{agent_type.value}={score:.2f}"
            for agent_type, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        support_text = ", ".join(agent.value for agent in supporting_agents) or "none"
        intent = req.intent.value if req.intent else "unknown"
        return (
            f"intent={intent}, group={req.intent_group or 'unknown'}, "
            f"primary={primary_agent.value}, supporting={support_text}, scores=[{score_text}]"
        )

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        例如"市公安局的职责是什么，三级响应时归谁指挥"需要架构和条文同时处理。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        if any(kw in msg for kw in ARCH_KWS):
            targets.append(AgentType.ARCHITECTURE)
        if any(kw in msg for kw in POLICY_KWS):
            targets.append(AgentType.POLICIES)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
        """低置信度且无明确意图时，先追问，避免误路由。"""
        if req.intent != IntentCategory.OTHER:
            return False
        text = (req.message or "").strip()
        if len(text) <= 2:
            return False
        return req.intent_confidence < 0.5

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    def _monitor_degraded(self, agent_type: AgentType) -> bool:
        """专业 Agent 降权过高时由路由层回退，确保单实例部署也能生效。"""
        if agent_type in (AgentType.GENERAL, AgentType.ESCALATION):
            return False
        agents = self._pool.get(agent_type, [])
        if not agents:
            return False
        threshold = _env_float(
            "ECHOMIND_MONITOR_FALLBACK_PENALTY",
            self.MONITOR_FALLBACK_PENALTY,
        )
        penalties = [
            float(getattr(getattr(agent, "stats", None), "monitor_penalty", 0.0) or 0.0)
            for agent in agents
        ]
        # 只要还有一个健康实例就继续使用该专业类型；全部高降权时才回退。
        return min(penalties) >= min(max(threshold, 0.0), 0.9)

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req)

        # 专属 Agent 失败时降级到 GeneralAgent
        if not response.success and agent_type not in (AgentType.GENERAL, AgentType.ESCALATION):
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req)

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                    "role": agent.profile.role,
                    "workflow": list(agent.profile.workflow),
                    "tool_scope": list(agent.profile.tool_scope),
                    "available_tools": list(agent.get_tools()),
                    "model": agent._model,
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 architecture_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)