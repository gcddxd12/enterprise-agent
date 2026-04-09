#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试监控系统导入
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_monitoring_import():
    """测试监控系统导入"""
    print("=== 测试监控系统导入 ===")

    try:
        # 测试导入监控系统
        from monitoring_system import get_monitoring_system, monitor_workflow, monitor_node
        print("[SUCCESS] monitoring_system 导入成功")

        # 测试导入监控配置
        from monitoring_config import get_config_manager, MonitoringConfig
        print("[SUCCESS] monitoring_config 导入成功")

        # 创建配置管理器
        manager = get_config_manager()
        print(f"[INFO] 配置管理器创建成功")

        # 获取监控系统实例
        monitoring = get_monitoring_system()
        print(f"[INFO] 监控系统实例创建成功: {type(monitoring)}")

        # 测试日志功能
        monitoring.log_event("INFO", "test_import", message="监控系统导入测试成功")
        print("[SUCCESS] 监控系统日志功能正常")

        return True

    except ImportError as e:
        print(f"[FAILED] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAILED] 其他错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_monitoring_system_standalone():
    """测试监控系统独立功能"""
    print("\n=== 测试监控系统独立功能 ===")

    try:
        from monitoring_system import test_monitoring_system

        success = test_monitoring_system()
        if success:
            print("[SUCCESS] 监控系统独立测试通过")
            return True
        else:
            print("[FAILED] 监控系统独立测试失败")
            return False

    except Exception as e:
        print(f"[FAILED] 监控系统独立测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("监控系统导入测试")
    print("=" * 50)

    # 测试导入
    import_success = test_monitoring_import()

    # 测试独立功能
    if import_success:
        standalone_success = test_monitoring_system_standalone()
    else:
        standalone_success = False
        print("[SKIP] 跳过独立功能测试（导入失败）")

    print(f"\n{'='*50}")
    print("测试结果汇总:")
    print(f"导入测试: {'通过' if import_success else '失败'}")
    print(f"独立功能测试: {'通过' if standalone_success else '失败'}")

    if import_success and standalone_success:
        print("\n[SUCCESS] 所有监控系统导入测试通过")
        return True
    else:
        print("\n[FAILED] 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)