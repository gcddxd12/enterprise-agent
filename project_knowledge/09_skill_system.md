# Skill技能系统

## 概述

Skill系统是一种领域专用指令注入机制，通过 `skills/*.md` 文件定义各领域的专业知识、处理流程和话术模板。Agent运行时通过**关键词自动匹配**（preprocess_node）和**LLM主动加载**（use_skill工具）两种方式获取技能指令，实现零代码扩展业务领域能力。

## 1. 核心架构

```
用户提问 → preprocess_node (关键词匹配)
                │
                ├── 命中 → 注入skill内容到系统提示 → agent_node遵循skill指令回答
                └── 未命中 → 通用提示
                
LLM在ReAct循环中 → 需要专业领域知识 → 调用use_skill工具 → 加载更多skill
```

### 1.1 两种激活方式

| 方式 | 触发机制 | 时机 | 适用场景 |
|------|---------|------|---------|
| 自动匹配 | preprocess_node关键词命中 | Agent启动前 | 用户问题包含明确关键词 |
| 主动加载 | LLM调用use_skill工具 | ReAct循环中 | 需要补充专业指令 |

## 2. Skill文件格式

每个skill是一个 `.md` 文件，包含YAML frontmatter（元数据）+ Markdown正文（LLM指令）。

### 2.1 Frontmatter字段

```yaml
---
name: 技能显示名称          # 必填
description: 简短描述        # 用于LLM判断是否需要加载
triggers:                   # 触发关键词列表（自动匹配用）
  - 关键词1
  - 关键词2
tools:                      # 该skill可能用到的工具
  - knowledge_search
priority: 5                 # 优先级1-10，越高越优先（默认5）
---
```

### 2.2 Body内容

Markdown格式的LLM指令，推荐结构：

```markdown
## 角色
你是一位专业的XXX顾问...

## 强制规则（必须遵守）
1. **规则1（加粗+禁止性措辞）**
2. 规则2

## 处理流程
1. 步骤1
2. 步骤2

## 注意事项
- 注意点1
- 注意点2
```

**重要**：LLM倾向于"过度帮助"（如列出所有套餐而不是先确认需求），所以关键约束必须用加粗、禁止性措辞（"禁止...""必须先..."）写在强制规则部分。

## 3. 当前Skill列表

| Skill | 优先级 | 触发词 | 工具 |
|-------|--------|--------|------|
| 投诉处理 | 9 | 投诉、举报、不满 | knowledge_search, escalate_to_human |
| 5G套餐咨询 | 8 | 5G, 套餐, 资费, 流量包, 月租, 畅享... | knowledge_search |
| 宽带业务 | 7 | 宽带, 光纤, 网速, 路由器, 装维 | knowledge_search |
| 国际漫游 | 6 | 国际漫游, 出国, 港澳台, 境外, 海外 | knowledge_search, get_current_date |

## 4. SkillManager实现

**文件**: [skill_manager.py](e:\my_multi_agent\skill_manager.py)

### 4.1 核心数据结构

```python
@dataclass
class Skill:
    name: str              # 技能名称
    description: str       # 简短描述
    triggers: List[str]    # 触发关键词
    tools: List[str]       # 关联工具
    priority: int          # 优先级(1-10)
    content: str           # Markdown正文(LLM指令)
    file_path: str         # 源文件路径
```

### 4.2 关键词匹配

使用倒排索引实现O(1)查找：

```python
class SkillManager:
    def __init__(self, skills_dir="./skills"):
        self._trigger_index: Dict[str, List[str]] = {}  # keyword_lower → [skill_names]
        self._reload()  # 扫描skills/*.md，构建索引

    def find_matching_skills(self, query: str, max_skills=2) -> List[Skill]:
        """基于关键词匹配，按priority降序返回top-N"""
        # 遍历query中的词，查倒排索引
        # 命中多个skill时按priority + 匹配关键词数排序
```

### 4.3 use_skill工具

供LLM在ReAct循环中主动加载skill：

```python
@tool
def use_skill(skill_name: str) -> str:
    """加载指定领域的专业技能指令。当你需要特定领域的专业知识、
    处理流程或话术模板时调用此工具。"""
    skill = skill_manager.get_skill(skill_name)
    if skill:
        return f"已加载技能「{skill.name}」:\n\n{skill.content}"
```

## 5. 集成方式

### 5.1 preprocess_node自动匹配

在用户问题进入Agent前自动匹配skill，注入state：

```python
def preprocess_node(state: AgentState) -> AgentState:
    matching_skills = skill_manager.find_matching_skills(state["user_query"], max_skills=2)
    active_skills = [s.name for s in matching_skills]
    # state会被注入到系统提示中
```

### 5.2 动态系统提示

`build_system_prompt()` 将匹配到的skill内容追加到BASE_PROMPT后面，LLM看到的是增强后的完整提示：

```python
def build_system_prompt(active_skills=None):
    prompt = BASE_SYSTEM_PROMPT
    if active_skills:
        for skill_name in active_skills:
            skill = skill_manager.get_skill(skill_name)
            prompt += f"\n### {skill.name}\n{skill.content}\n"
    return prompt
```

### 5.3 UI显示

在Streamlit调试面板的"🎯 激活的专业技能"区域展示当前激活的skill。

## 6. 新建Skill步骤

1. 在 `skills/` 目录创建 `xxx.md`
2. 编写YAML frontmatter（设置triggers关键词、priority优先级）
3. 编写Markdown body（角色、强制规则、处理流程、注意事项）
4. 重启Agent → SkillManager自动扫描加载
5. 测试：输入含触发词的查询，观察是否匹配

**无需修改任何Python代码。**

## 7. 设计要点

- **约束有效性**：规则用加粗+禁止性措辞，放在"强制规则"标题下
- **上下文控制**：最多自动匹配2个skill，避免提示过长
- **优先级机制**：多个skill命中时高priority优先（投诉处理 > 5G套餐 > 宽带 > 国际漫游）
- **热加载**：调用`skill_manager.reload()`可热更新skill，无需重启
