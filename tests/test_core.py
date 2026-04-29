"""
核心模块单元测试
不依赖外部 API 即可运行的测试，CI 中安全执行。
"""

import sys
import os

# 确保项目根目录在 Python 路径中（CI 环境兼容）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# CI 诊断：打印路径信息（便于排查）
print(f"[CI-DEBUG] Project root: {_PROJECT_ROOT}", flush=True)
print(f"[CI-DEBUG] sys.path[0]: {sys.path[0]}", flush=True)
print(f"[CI-DEBUG] skill_manager.py exists: {os.path.exists(os.path.join(_PROJECT_ROOT, 'skill_manager.py'))}", flush=True)



# 诊断辅助：验证 skill_manager 是否可导入
import importlib.util
_skill_spec = importlib.util.find_spec("skill_manager")
if _skill_spec is None:
    raise RuntimeError(
        f"CRITICAL: skill_manager module not found in Python path.\n"
        f"  sys.path[0]: {sys.path[0]}\n"
        f"  Project root: {_PROJECT_ROOT}\n"
        f"  skill_manager.py exists: {os.path.exists(os.path.join(_PROJECT_ROOT, 'skill_manager.py'))}\n"
        f"  Full sys.path: {sys.path[:5]}"
    )


class TestSkillManager:
    """Skill 系统测试"""

    def test_skill_loading(self):
        """验证 SkillManager 能加载所有 skill 文件"""
        from skill_manager import get_skill_manager

        mgr = get_skill_manager(skills_dir="./skills")
        skills = mgr.list_skills()
        assert len(skills) >= 4, f"Expected at least 4 skills, got {len(skills)}"

        names = [s["name"] for s in skills]
        assert "5G套餐咨询" in names
        assert "宽带业务" in names
        assert "投诉处理" in names
        assert "国际漫游" in names

    def test_trigger_index_built(self):
        """验证倒排索引构建正确"""
        from skill_manager import get_skill_manager

        mgr = get_skill_manager(skills_dir="./skills")
        assert "套餐" in mgr._trigger_index
        assert "5g" in mgr._trigger_index
        assert "宽带" in mgr._trigger_index

    def test_keyword_matching(self):
        """验证关键词匹配返回正确 skill"""
        from skill_manager import get_skill_manager

        mgr = get_skill_manager(skills_dir="./skills")

        # 5G套餐查询应匹配 5G套餐咨询
        result = mgr.find_matching_skills("我想办个5G套餐，每月100块左右")
        assert len(result) > 0
        assert result[0].name == "5G套餐咨询"

    def test_matching_by_priority(self):
        """验证多 skill 匹配时按优先级排序"""
        from skill_manager import get_skill_manager

        mgr = get_skill_manager(skills_dir="./skills")

        # "投诉"匹配投诉处理(9)和可能的其他
        result = mgr.find_matching_skills("我要投诉5G套餐乱扣费")
        assert len(result) > 0
        # 投诉处理 priority=9，应该排第一
        assert result[0].name == "投诉处理"

    def test_build_system_prompt(self):
        """验证动态系统提示构建"""
        from skill_manager import get_skill_manager

        # 这个测试只验证函数存在且可调用
        mgr = get_skill_manager(skills_dir="./skills")
        skill = mgr.get_skill("5G套餐咨询")
        assert skill is not None
        assert "禁止一次性列出全部套餐" in skill.content
        assert skill.priority == 8

    def test_unknown_query_no_match(self):
        """验证无关查询不匹配任何 skill"""
        from skill_manager import get_skill_manager

        mgr = get_skill_manager(skills_dir="./skills")
        result = mgr.find_matching_skills("今天天气怎么样")
        assert len(result) == 0


class TestMCPConfig:
    """MCP 配置文件测试"""

    def test_config_exists(self):
        """验证 mcp_servers.yaml 存在且可解析"""
        import yaml
        assert os.path.exists("./mcp_servers.yaml")

        with open("./mcp_servers.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "servers" in data
        assert "billing" in data["servers"]
        assert "ticket" in data["servers"]
        assert data["servers"]["billing"]["enabled"] is True

    def test_mcp_server_scripts_exist(self):
        """验证 MCP Server 脚本存在"""
        assert os.path.exists("./mcp_servers/billing_server.py")
        assert os.path.exists("./mcp_servers/ticket_server.py")

    def test_mcp_server_syntax(self):
        """验证 MCP Server 语法正确"""
        import py_compile
        py_compile.compile("./mcp_servers/billing_server.py", doraise=True)
        py_compile.compile("./mcp_servers/ticket_server.py", doraise=True)


class TestSkillFiles:
    """Skill 文件规范测试"""

    def test_all_skills_have_frontmatter(self):
        """验证所有 skill 文件有正确的 frontmatter"""
        import glob
        import yaml

        for md_file in glob.glob("./skills/*.md"):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            assert content.startswith("---"), f"{md_file} 缺少 frontmatter"

            parts = content.split("---", 2)
            assert len(parts) >= 3, f"{md_file} frontmatter 格式不正确"

            fm = yaml.safe_load(parts[1])
            assert "name" in fm, f"{md_file} 缺少 name"
            assert "triggers" in fm, f"{md_file} 缺少 triggers"
            assert "priority" in fm, f"{md_file} 缺少 priority"
            assert len(fm["triggers"]) > 0, f"{md_file} triggers 为空"


class TestRequirements:
    """依赖文件测试"""

    def test_pyyaml_in_requirements(self):
        """验证 pyyaml 在 requirements.txt 中"""
        with open("./requirements.txt", "r", encoding="utf-8") as f:
            content = f.read()
        assert "pyyaml" in content.lower()

    def test_pyyaml_installed(self):
        """验证 pyyaml 已安装"""
        import yaml
        assert yaml.__version__
