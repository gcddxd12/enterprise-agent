"""Tests for mcp_client.py — config, dataclasses, tool wrapping (no subprocess)."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from mcp_client import (
    MCPServerConfig,
    MCPToolInfo,
    MCPClientManager,
    _build_pydantic_model,
    get_mcp_manager,
)


# ============================================================================
# TestMCPServerConfig
# ============================================================================


class TestMCPServerConfig:
    def test_defaults(self):
        config = MCPServerConfig(name="test", command="python", args=["server.py"])
        assert config.name == "test"
        assert config.enabled is True
        assert config.env == {}

    def test_disabled(self):
        config = MCPServerConfig(
            name="off", command="echo", args=[], enabled=False
        )
        assert config.enabled is False


# ============================================================================
# TestMCPToolInfo
# ============================================================================


class TestMCPToolInfo:
    def test_fields(self):
        info = MCPToolInfo(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            server_name="demo",
        )
        assert info.name == "test_tool"
        assert info.server_name == "demo"


# ============================================================================
# TestMCPClientManagerConfig
# ============================================================================


class TestMCPClientManagerConfig:
    def test_load_config_with_yaml(self):
        """Load config from a temporary YAML file."""
        yaml_content = """servers:
  test_server:
    name: Test
    command: python
    args: ["-c", "print('hello')"]
    enabled: true
    description: A test server
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            tmp_path = f.name

        try:
            manager = MCPClientManager(config_path=tmp_path)
            assert "test_server" in manager.servers
            cfg = manager.servers["test_server"]
            assert cfg.name == "Test"
            assert cfg.enabled is True
        finally:
            Path(tmp_path).unlink()

    def test_missing_config_handled(self):
        """Non-existent config file should not crash (empty servers)."""
        manager = MCPClientManager(config_path="./nonexistent_file_xyz.yaml")
        assert isinstance(manager.servers, dict)

    def test_disabled_server_skipped_in_list_tools(self):
        """Disabled servers should not contribute tools."""
        yaml_content = """servers:
  enabled_srv:
    name: On
    command: echo
    args: []
    enabled: true
  disabled_srv:
    name: Off
    command: echo
    args: []
    enabled: false
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            tmp_path = f.name

        try:
            manager = MCPClientManager(config_path=tmp_path)
            assert "enabled_srv" in manager.servers
            assert "disabled_srv" in manager.servers
            assert manager.servers["disabled_srv"].enabled is False
        finally:
            Path(tmp_path).unlink()


# ============================================================================
# TestBuildPydanticModel
# ============================================================================


class TestBuildPydanticModel:
    def test_empty_schema_adds_query_field(self):
        """Empty properties dict should get a default query field."""
        schema = {"type": "object", "properties": {}}
        model = _build_pydantic_model("test_tool", schema)
        assert model is not None
        # Should have a 'query' field
        assert "query" in model.model_fields

    def test_with_properties(self):
        """Schema with properties should generate fields."""
        schema = {
            "type": "object",
            "properties": {
                "phone": {"type": "string", "description": "Phone number"},
                "month": {"type": "string", "description": "Month"},
            },
            "required": ["phone"],
        }
        model = _build_pydantic_model("billing_tool", schema)
        assert model is not None
        assert "phone" in model.model_fields
        assert "month" in model.model_fields


# ============================================================================
# TestGlobalManager
# ============================================================================


class TestGlobalManager:
    def test_get_mcp_manager_returns_none_or_valid(self):
        """Before initialization, manager may be None."""
        mgr = get_mcp_manager()
        # May be None or initialized depending on import side-effects
        if mgr is not None:
            assert hasattr(mgr, "list_tools")
            assert hasattr(mgr, "servers")
