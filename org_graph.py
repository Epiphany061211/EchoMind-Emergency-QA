"""
职能骨架查询引擎。

读取 data/architecture/ 下的预案骨架 JSON（机器抽取的派生视图），
把"节点 + 带类型边"装进内存，回答组织架构类问题：
上级 / 下级 / 平级 / 职责 / 响应权限。

骨架文件是预案原文的派生视图，真源是预案 PDF；
预案修订后重新生成 JSON，不手工维护本引擎的数据。
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── 协议字段（改名必须全局替换）─────────────────────────────────────────────

ARCH_DIR = os.getenv("ARCH_DIR", "./data/architecture")

# 层级边中"上级"在哪一端：
#   领导/层级指导/下设/设岗位 —— from 端是上级（省委省政府 --领导--> 省防减救灾委）
#   成员/属于                 —— to 端是上级（省公安厅 --成员--> 省地灾防指）
HIER_SUPERIOR_AT_FROM = {"领导", "层级指导", "下设", "设岗位"}
HIER_SUPERIOR_AT_TO = {"成员", "属于"}

# 信息边：描述工作关系但不构成上下级
INFO_EDGE_TYPES = {"设在", "牵头", "战时报请", "聘请组建", "战时组建"}

# 跨预案衔接边：市级节点 --属于--> 省级节点（注意全角括号）
CROSS_PLAN_LINKS = {
    "市级·市地质灾害指挥部": "省级·市（州）地灾防指",
}

TYPE_CN = {
    "party_gov": "党委政府",
    "committee": "委员会",
    "headquarters": "指挥部",
    "headquarters_lower": "下级指挥部",
    "position": "岗位",
    "office": "办公室",
    "member_unit": "成员单位",
    "work_group": "工作组",
    "grassroots_org": "基层组织",
    "enterprise_org": "企业组织",
    "expert_group": "专家组",
    "ad_hoc_org": "临时机构",
    "external": "外部主体",
}


class OrgGraph:
    """两份预案合并成的一张组织架构图。"""

    def __init__(self, arch_dir: str = ARCH_DIR):
        self.nodes: Dict[str, Dict[str, Any]] = {}   # "级别·名称" -> 节点
        self.edges: List[Dict[str, Any]] = []        # from/to 都是节点键
        self.plans: Dict[str, Dict[str, Any]] = {}   # 级别 -> meta + response_levels

        for path in sorted(Path(arch_dir).glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            level = doc["meta"]["admin_level"]
            self.plans[level] = {
                "plan_name": doc["meta"]["plan_name"],
                "response_levels": doc.get("response_levels", []),
                "disaster_level_response": doc.get("disaster_level_response", []),
            }
            for node in doc["nodes"]:
                key = f"{level}·{node['id']}"
                merged = dict(node)
                merged["plan_name"] = doc["meta"]["plan_name"]
                self.nodes[key] = merged
            for edge in doc["edges"]:
                self.edges.append({
                    "from": f"{level}·{edge['from']}",
                    "to": f"{level}·{edge['to']}",
                    "type": edge["type"],
                    "page": edge.get("page"),
                })

        # 跨预案衔接边
        for frm, to in CROSS_PLAN_LINKS.items():
            if frm in self.nodes and to in self.nodes:
                self.edges.append({"from": frm, "to": to, "type": "属于", "page": None})

        # 悬空边检查：边的两端必须是已定义节点
        dangling = [
            e for e in self.edges
            if e["from"] not in self.nodes or e["to"] not in self.nodes
        ]
        if dangling:
            raise ValueError(f"骨架存在悬空边: {dangling[:3]}")

    # ── 名称解析 ────────────────────────────────────────────────────────────

    @staticmethod
    def display(key: str) -> str:
        """"省级·省公安厅" -> "省公安厅"。"""
        return key.split("·", 1)[1] if "·" in key else key

    @staticmethod
    def level_of(key: str) -> str:
        return key.split("·", 1)[0] if "·" in key else ""

    def find(self, name: str) -> Optional[str]:
        """把用户说的单位名解析成唯一节点键；找不到或有歧义返回 None。"""
        name = (name or "").strip()
        if not name:
            return None
        if name in self.nodes:
            return name
        for level in self.plans:
            if f"{level}·{name}" in self.nodes:
                return f"{level}·{name}"
        candidates = self.find_candidates(name)
        return candidates[0] if len(candidates) == 1 else None

    # 用户口语里常见的行政区划前缀，骨架里用的是规范简称
    _NAME_PREFIXES = ("四川省", "宜宾市")

    def find_candidates(self, name: str) -> List[str]:
        """模糊匹配候选，供"找不到时给提示"用。

        双向匹配：
          1. 用户输入是节点名的子串（"公安" → "市公安局"）
          2. 节点名是用户输入的后缀（"宜宾市公安局" → "市公安局"）
        并自动剥掉"四川省/宜宾市"这类行政区划前缀。
        """
        name = (name or "").strip()
        if not name:
            return []
        variants = {name}
        for prefix in self._NAME_PREFIXES:
            if name.startswith(prefix) and len(name) > len(prefix):
                variants.add(name[len(prefix):])
        out = []
        for k in self.nodes:
            disp = self.display(k)
            if any(v in k or v in disp for v in variants):
                out.append(k)
            elif any(disp and disp in v for v in variants):
                out.append(k)
        return out

    def _node_brief(self, key: str) -> Dict[str, Any]:
        node = self.nodes[key]
        return {
            "key": key,
            "name": self.display(key),
            "level": self.level_of(key),
            "type": TYPE_CN.get(node.get("type"), node.get("type")),
            "plan": node.get("plan_name"),
            "page": node.get("page"),
        }

    # ── 关系查询 ────────────────────────────────────────────────────────────

    def _superior_edges(self, key: str) -> List[Dict[str, Any]]:
        """该节点作为"下级"出现的层级边。"""
        result = []
        for e in self.edges:
            if e["type"] in HIER_SUPERIOR_AT_FROM and e["to"] == key:
                result.append(e)
            elif e["type"] in HIER_SUPERIOR_AT_TO and e["from"] == key:
                result.append(e)
        return result

    def _subordinate_edges(self, key: str) -> List[Dict[str, Any]]:
        """该节点作为"上级"出现的层级边。"""
        result = []
        for e in self.edges:
            if e["type"] in HIER_SUPERIOR_AT_FROM and e["from"] == key:
                result.append(e)
            elif e["type"] in HIER_SUPERIOR_AT_TO and e["to"] == key:
                result.append(e)
        return result

    def superiors(self, name: str) -> Dict[str, Any]:
        """直接上级列表。"""
        key = self.find(name)
        if key is None:
            return {"success": False, "unit": name,
                    "candidates": [self.display(k) for k in self.find_candidates(name)]}
        items = []
        for e in self._superior_edges(key):
            superior = e["from"] if e["type"] in HIER_SUPERIOR_AT_FROM else e["to"]
            brief = self._node_brief(superior)
            brief["relation"] = e["type"]
            items.append(brief)
        return {"success": True, "unit": self.display(key),
                "matched_node": key, "superiors": items}

    def subordinates(self, name: str) -> Dict[str, Any]:
        """直接下级列表（含成员单位、下设机构、岗位）。"""
        key = self.find(name)
        if key is None:
            return {"success": False, "unit": name,
                    "candidates": [self.display(k) for k in self.find_candidates(name)]}
        items = []
        for e in self._subordinate_edges(key):
            sub = e["to"] if e["type"] in HIER_SUPERIOR_AT_FROM else e["from"]
            brief = self._node_brief(sub)
            brief["relation"] = e["type"]
            items.append(brief)
        return {"success": True, "unit": self.display(key),
                "matched_node": key, "subordinates": items}

    def peers(self, name: str) -> Dict[str, Any]:
        """平级 = 同一上级、同一关系类型的其他节点（现算，不预存）。"""
        key = self.find(name)
        if key is None:
            return {"success": False, "unit": name,
                    "candidates": [self.display(k) for k in self.find_candidates(name)]}
        peers: List[Dict[str, Any]] = []
        for e in self._superior_edges(key):
            superior = e["from"] if e["type"] in HIER_SUPERIOR_AT_FROM else e["to"]
            for se in self._subordinate_edges(superior):
                if se["type"] != e["type"]:
                    continue
                sibling = se["to"] if se["type"] in HIER_SUPERIOR_AT_FROM else se["from"]
                if sibling == key:
                    continue
                brief = self._node_brief(sibling)
                brief["common_superior"] = self.display(superior)
                brief["relation"] = se["type"]
                peers.append(brief)
        # 去重（可能经多条边共享上级）
        seen, unique = set(), []
        for p in peers:
            if p["key"] not in seen:
                seen.add(p["key"])
                unique.append(p)
        return {"success": True, "unit": self.display(key),
                "matched_node": key, "peers": unique, "count": len(unique)}

    def chain(self, name: str) -> Dict[str, Any]:
        """从该节点一路向上到顶的最长链路（上级不止一个时，取走得最高的那条）。"""
        key = self.find(name)
        if key is None:
            return {"success": False, "unit": name,
                    "candidates": [self.display(k) for k in self.find_candidates(name)]}

        def _climb(node_key: str, visited: set) -> List[str]:
            best = [node_key]
            for e in self._superior_edges(node_key):
                up = e["from"] if e["type"] in HIER_SUPERIOR_AT_FROM else e["to"]
                if up in visited:
                    continue
                candidate = [node_key] + _climb(up, visited | {up})
                if len(candidate) > len(best):
                    best = candidate
            return best

        path = _climb(key, {key})
        return {"success": True, "unit": self.display(key),
                "chain": [self._node_brief(k) for k in path],
                "depth": len(path)}

    def duties(self, name: str) -> Dict[str, Any]:
        """职责清单（含预案页码回链）。"""
        key = self.find(name)
        if key is None:
            return {"success": False, "unit": name,
                    "candidates": [self.display(k) for k in self.find_candidates(name)]}
        node = self.nodes[key]
        items = [{"text": d["text"], "page": d.get("page")} for d in node.get("duties", [])]
        result: Dict[str, Any] = {
            "success": True,
            "unit": self.display(key),
            "matched_node": key,
            "type": TYPE_CN.get(node.get("type"), node.get("type")),
            "activation": node.get("activation"),
            "duties": items,
        }
        if node.get("filled_by"):
            result["filled_by"] = node["filled_by"]
        if node.get("name_full"):
            result["name_full"] = node["name_full"]
        # 附带信息边（牵头/设在/战时报请等），帮助回答"它跟谁协作"
        relations = []
        for e in self.edges:
            if e["type"] not in INFO_EDGE_TYPES:
                continue
            if e["from"] == key:
                relations.append({"relation": e["type"], "target": self.display(e["to"]),
                                  "page": e.get("page")})
            elif e["to"] == key:
                relations.append({"relation": f"被{e['type']}", "target": self.display(e["from"]),
                                  "page": e.get("page")})
        if relations:
            result["relations"] = relations
        return result

    def authority(self, level: str) -> Dict[str, Any]:
        """响应等级权限对照：谁提出、谁决定、谁指挥、谁终止（省级+市级并排）。"""
        keyword = (level or "").strip()
        for lv in ("一级", "二级", "三级", "四级"):
            if lv in keyword:
                keyword = lv
                break
        result: Dict[str, Any] = {"success": True, "level": keyword, "plans": {}}
        for lv, plan in self.plans.items():
            for entry in plan["response_levels"]:
                if entry.get("level") == keyword:
                    result["plans"][plan["plan_name"]] = {
                        "conditions": entry.get("conditions", []),
                        "proposal_by": entry.get("proposal_by"),
                        "decided_by": entry.get("decided_by"),
                        "command_by": entry.get("command_by"),
                        "terminated_by": entry.get("terminated_by"),
                        "page": entry.get("page"),
                    }
        if not result["plans"]:
            return {"success": False, "level": keyword,
                    "error": "骨架中没有该等级的响应条款，可选：一级/二级/三级/四级"}
        return result