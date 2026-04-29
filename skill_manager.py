"""
Skill管理系统

Skills是以.md文件形式定义在skills/目录中的领域专用指令。
SkillManager负责扫描、解析、索引和匹配skill，
preprocess_node自动匹配 + use_skill工具供LLM主动加载。
"""

import os
import glob
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Skill:
    """单个Skill的定义"""
    name: str
    description: str
    triggers: List[str]
    tools: List[str]
    priority: int
    content: str
    file_path: str = ""


class SkillManager:
    """Skill管理器：扫描、索引、匹配、检索"""

    def __init__(self, skills_dir: str = "./skills"):
        self.skills_dir = skills_dir
        self.skills: Dict[str, Skill] = {}
        self._trigger_index: Dict[str, List[str]] = {}  # keyword_lower -> [skill_name, ...]
        self._reload()

    def _reload(self):
        """扫描skills/目录，解析所有.md文件，构建倒排索引"""
        self.skills.clear()
        self._trigger_index.clear()

        pattern = os.path.join(self.skills_dir, "*.md")
        for md_file in glob.glob(pattern):
            try:
                skill = self._parse_skill_file(md_file)
                self.skills[skill.name] = skill
                for trigger in skill.triggers:
                    key = trigger.lower()
                    if key not in self._trigger_index:
                        self._trigger_index[key] = []
                    self._trigger_index[key].append(skill.name)
            except Exception as e:
                print(f"[SkillManager] 解析skill文件失败: {md_file}, 错误: {e}")

        print(f"[SkillManager] 已加载 {len(self.skills)} 个skill: {list(self.skills.keys())}")

    def _parse_skill_file(self, file_path: str) -> Skill:
        """解析单个skill.md文件（YAML frontmatter + Markdown正文）"""
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # 解析 YAML frontmatter (--- ... ---)
        if not raw.startswith("---"):
            raise ValueError(f"Skill文件缺少frontmatter: {file_path}")

        parts = raw.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"Skill文件格式不正确: {file_path}")

        frontmatter = yaml.safe_load(parts[1].strip())
        body = parts[2].strip()

        return Skill(
            name=frontmatter.get("name", os.path.basename(file_path)),
            description=frontmatter.get("description", ""),
            triggers=frontmatter.get("triggers", []),
            tools=frontmatter.get("tools", []),
            priority=frontmatter.get("priority", 5),
            content=body,
            file_path=file_path,
        )

    def find_matching_skills(self, query: str, max_skills: int = 2) -> List[Skill]:
        """基于关键词匹配找到相关skill，按priority降序返回top-N"""
        query_lower = query.lower()
        matched: Dict[str, int] = {}  # skill_name -> match_count

        for keyword, skill_names in self._trigger_index.items():
            if keyword in query_lower:
                for skill_name in skill_names:
                    matched[skill_name] = matched.get(skill_name, 0) + 1

        # 按 priority 降序排序（同优先级按匹配关键词数量降序）
        sorted_skills = sorted(
            matched.keys(),
            key=lambda s: (self.skills[s].priority, matched[s]),
            reverse=True,
        )

        return [self.skills[name] for name in sorted_skills[:max_skills]]

    def get_skill(self, name: str) -> Optional[Skill]:
        """按名称精确查找skill"""
        return self.skills.get(name)

    def list_skills(self) -> List[Dict]:
        """返回所有skill的摘要信息（供LLM参考）"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "triggers": s.triggers,
                "tools": s.tools,
            }
            for s in self.skills.values()
        ]

    def reload(self):
        """重新加载所有skill文件（用于热更新）"""
        self._reload()


# ========== 全局单例 ==========
_skill_manager: Optional[SkillManager] = None


def get_skill_manager(skills_dir: str = "./skills") -> SkillManager:
    """获取全局SkillManager实例（单例模式）"""
    global _skill_manager
    if _skill_manager is None:
        _skill_manager = SkillManager(skills_dir=skills_dir)
    return _skill_manager


# ========== use_skill 工具 ==========
def create_use_skill_tool():
    """创建 use_skill 工具，供LLM在ReAct循环中主动加载skill指令"""
    from langchain_core.tools import tool  # 延迟导入，避免CI环境无此依赖时模块加载失败

    @tool
    def use_skill(skill_name: str) -> str:
        """加载指定领域的专业技能指令。当你需要特定领域的专业知识、
        处理流程或话术模板时调用此工具——比如用户咨询套餐、宽带、
        投诉、国际漫游等专业问题时。

        参数 skill_name: 要加载的技能名称。先思考用户问题涉及哪个领域，
        再传入对应的技能名称。可用技能请从上下文中的系统提示获取。
        """
        skill_manager = get_skill_manager()
        skill = skill_manager.get_skill(skill_name)
        if skill:
            return (
                f"已加载技能「{skill.name}」的专业指令:\n\n{skill.content}"
            )
        else:
            all_skills = skill_manager.list_skills()
            available = ", ".join([s["name"] for s in all_skills])
            return (
                f"未找到技能「{skill_name}」。当前可用技能: {available}\n"
                f"请选择以上技能之一，或直接使用通用知识回答用户问题。"
            )

    return use_skill
