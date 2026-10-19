"""Agent 工具定义与实现（应急预案版）。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具分三类：
  - 组织架构查询（query_superior / query_subordinates / query_peers /
    get_duties / get_response_authority）—— 查 data/architecture/ 下的
    职能骨架 JSON，确定性查表，不经模型推理
  - 通用辅助（inspect_request_context / suggest_required_fields）
  - 共享预案条文检索（search_knowledge_base，走 ChromaDB policies 集合）

骨架数据是预案原文的派生视图，真源是预案 PDF。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


# ── 组织架构查询（职能骨架）──────────────────────────────────────────────────

_ORG = None


def _get_org():
    """惰性加载职能骨架（第一次有架构查询时才读 JSON）。"""
    global _ORG
    if _ORG is None:
        from org_graph import OrgGraph
        _ORG = OrgGraph(os.getenv("ARCH_DIR", "./data/architecture"))
    return _ORG


def query_superior(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """查某单位/岗位的直接上级。"""
    unit = str(args.get("unit", "")).strip()
    if not unit:
        return {"success": False, "error": "unit 不能为空"}
    return _get_org().superiors(unit)


def query_subordinates(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """查某单位/岗位的直接下级（成员单位、下设机构、岗位）。"""
    unit = str(args.get("unit", "")).strip()
    if not unit:
        return {"success": False, "error": "unit 不能为空"}
    return _get_org().subordinates(unit)


def query_peers(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """查某单位/岗位的平级（同一上级、同一关系类型的其他单位）。"""
    unit = str(args.get("unit", "")).strip()
    if not unit:
        return {"success": False, "error": "unit 不能为空"}
    return _get_org().peers(unit)


def get_duties(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """查某单位/岗位的职责清单（含预案页码）。"""
    unit = str(args.get("unit", "")).strip()
    if not unit:
        return {"success": False, "error": "unit 不能为空"}
    return _get_org().duties(unit)


def get_response_authority(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """查某响应等级的权限：谁提出、谁决定、谁指挥、谁终止（省级市级并排对照）。"""
    level = str(args.get("level", "")).strip()
    if not level:
        return {"success": False, "error": "level 不能为空，可选：一级/二级/三级/四级"}
    return _get_org().authority(level)


# ── 通用辅助 ────────────────────────────────────────────────────────────────

def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用工具：返回当前请求的意图、紧急度等快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用工具：应急场景下建议下一轮只需向用户补充的字段。"""
    return {
        "required_fields_hint": [
            "想查组织架构（上级/平级/下级/职责）还是预案条文？",
            "查架构：给出单位或岗位名称（如 市公安局、省地灾防指、指挥长）",
            "查响应权限：给出等级（一级/二级/三级/四级）",
        ],
        "known_entities": req.entities or {},
    }


def create_handoff_summary(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """升级工具：生成需要人工核验的结构化摘要（不创建真实工单）。"""
    return {
        "request_id": req.request_id,
        "reason": str(args.get("reason", "需要人工对照预案原文核验"))[:120],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "sensitive_data_required": False,
    }


# ── 共享预案条文检索 ─────────────────────────────────────────────────────────

def build_shared_rag_tools(tool_manager: Any) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的预案条文检索工具。"""

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = int(args.get("top_k", 5) or 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await tool_manager.search_with_rewrite(
            "knowledge_search",
            query,
            top_k=top_k,
        )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "预案条文检索失败"),
                "results": [],
                "reranked": False,
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索应急预案条文库，返回最相关的预案片段（含预案名称和页码）；"
            "适用于处置措施、响应流程、保障要求等条文类问题。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


# ── 按角色分组的工具白名单 ───────────────────────────────────────────────────

def architecture_tools() -> Dict[str, AgentToolSpec]:
    """架构 Agent 专用：五个骨架查询工具。"""
    unit_prop = {
        "unit": {
            "type": "string",
            "description": "单位或岗位名称，如 市公安局、省地灾防指、指挥长",
        }
    }
    return {
        "query_superior": make_tool(
            "query_superior",
            "查询某单位/岗位的直接上级部门（领导关系、所属指挥部）。",
            unit_prop,
            query_superior,
            required=["unit"],
        ),
        "query_subordinates": make_tool(
            "query_subordinates",
            "查询某单位/岗位的直接下级：成员单位、下设机构、所设岗位。",
            unit_prop,
            query_subordinates,
            required=["unit"],
        ),
        "query_peers": make_tool(
            "query_peers",
            "查询某单位/岗位的平级单位（同一上级、同一关系类型的其他单位）。",
            unit_prop,
            query_peers,
            required=["unit"],
        ),
        "get_duties": make_tool(
            "get_duties",
            "查询某单位/岗位的职责清单，每条职责附预案页码；"
            "也包含牵头、设在、战时组建等协作关系。",
            unit_prop,
            get_duties,
            required=["unit"],
        ),
        "get_response_authority": make_tool(
            "get_response_authority",
            "查询某应急响应等级的权限划分：谁提出启动、谁决定、谁组织指挥、"
            "谁批准终止；省级与市级预案并排对照。",
            {
                "level": {
                    "type": "string",
                    "description": "响应等级：一级/二级/三级/四级",
                }
            },
            get_response_authority,
            required=["level"],
        ),
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性。",
            {"focus": {"type": "string", "description": "希望关注的方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def escalation_tools() -> Dict[str, AgentToolSpec]:
    return {
        "create_handoff_summary": make_tool(
            "create_handoff_summary",
            "生成需要人工对照预案原文核验的结构化摘要。",
            {"reason": {"type": "string", "description": "需要人工核验的原因"}},
            create_handoff_summary,
        ),
    }