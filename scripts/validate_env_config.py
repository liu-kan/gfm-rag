#!/usr/bin/env python3
"""
GFM-RAG 环境变量配置验证脚本

该脚本用于验证和检查GFM-RAG系统的环境变量配置状态，
确保Hydra配置文件能够正确读取和使用环境变量。

用法:
    python scripts/validate_env_config.py
    
    # 或者
    python -m scripts.validate_env_config
"""

import os
import sys
import logging
from typing import Dict, Any
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from gfmrag.config_manager import get_config_manager
    from gfmrag.kg_construction.langchain_util import init_langchain_model
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    print(f"错误：无法导入必要的模块: {e}")
    print("请确保您在项目根目录中运行此脚本，并且已安装所有依赖项。")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_environment_variables() -> Dict[str, Any]:
    """检查GFMRAG相关环境变量的设置状态"""
    print("=" * 60)
    print("🔧 GFM-RAG 环境变量配置检查")
    print("=" * 60)
    
    env_vars = {
        'GFMRAG_CHAT_PROVIDER': 'Chat服务提供商',
        'GFMRAG_CHAT_MODEL_NAME': 'Chat模型名称', 
        'GFMRAG_CHAT_BASE_URL': '第三方服务Base URL',
        'GFMRAG_CHAT_KEY': 'Chat服务API密钥'
    }
    
    env_status = {}
    print("\n1️⃣ 环境变量状态:")
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            env_status[var] = value
            if var == 'GFMRAG_CHAT_KEY':
                print(f"   ✅ {var}: 已设置 (***隐藏***)")
            else:
                print(f"   ✅ {var}: {value}")
        else:
            env_status[var] = None
            print(f"   ❌ {var}: 未设置")
    
    return env_status


def check_configuration_manager():
    """检查配置管理器的状态"""
    print("\n2️⃣ 配置管理器状态:")
    try:
        config_manager = get_config_manager()
        chat_config = config_manager.get_chat_config()
        
        print(f"   ✅ Provider: {chat_config.provider}")
        print(f"   ✅ Model: {chat_config.model_name}")
        print(f"   ✅ Base URL: {chat_config.base_url or '默认'}")
        print(f"   ✅ API Key: {'已设置' if chat_config.api_key else '未设置'}")
        
        return chat_config
    except Exception as e:
        print(f"   ❌ 配置管理器错误: {e}")
        return None


def test_hydra_config_loading():
    """测试Hydra配置文件的加载"""
    print("\n3️⃣ Hydra配置文件测试:")
    
    config_dir = project_root / "gfmrag" / "workflow" / "config"
    
    try:
        # 测试 NER 模型配置
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="ner_model/llm_ner_model.yaml")
            print(f"   ✅ NER配置加载成功:")
            print(f"      - llm_api: {cfg.llm_api}")
            print(f"      - model_name: {cfg.model_name}")
            print(f"      - base_url: {cfg.base_url}")
            print(f"      - api_key: {'已设置' if cfg.api_key and cfg.api_key != 'null' else '未设置'}")
        
        # 测试 OpenIE 模型配置
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="openie_model/llm_openie_model.yaml")
            print(f"   ✅ OpenIE配置加载成功:")
            print(f"      - llm_api: {cfg.llm_api}")
            print(f"      - model_name: {cfg.model_name}")
            print(f"      - base_url: {cfg.base_url}")
            print(f"      - api_key: {'已设置' if cfg.api_key and cfg.api_key != 'null' else '未设置'}")
            
        return True
    except Exception as e:
        print(f"   ❌ Hydra配置加载失败: {e}")
        return False


def test_model_initialization():
    """测试模型初始化是否正常"""
    print("\n4️⃣ 模型初始化测试:")
    try:
        print("   🔄 正在测试LangChain模型初始化...")
        model = init_langchain_model()
        print(f"   ✅ 模型初始化成功: {type(model).__name__}")
        
        # 尝试简单的模型调用（如果可能）
        try:
            response = model.invoke("Hello, this is a test message.")
            print("   ✅ 模型调用测试成功")
            print(f"      响应预览: {response.content[:100]}...")
            return True
        except Exception as e:
            print(f"   ⚠️ 模型调用测试失败（可能是网络或认证问题）: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ 模型初始化失败: {e}")
        return False


def generate_configuration_recommendations(env_status: Dict[str, Any]):
    """生成配置建议"""
    print("\n5️⃣ 配置建议:")
    
    provider = env_status.get('GFMRAG_CHAT_PROVIDER')
    base_url = env_status.get('GFMRAG_CHAT_BASE_URL')
    api_key = env_status.get('GFMRAG_CHAT_KEY')
    
    if not provider:
        print("   📝 建议设置 GFMRAG_CHAT_PROVIDER 环境变量")
        print("      例如: export GFMRAG_CHAT_PROVIDER='third-party'")
    
    if provider == 'third-party' and not base_url:
        print("   📝 使用第三方服务时，建议设置 GFMRAG_CHAT_BASE_URL")
        print("      例如: export GFMRAG_CHAT_BASE_URL='http://localhost:8000/v1'")
    
    if provider in ['openai', 'together', 'nvidia'] and not api_key:
        print(f"   📝 使用 {provider} 服务时，建议设置 GFMRAG_CHAT_KEY")
        print("      例如: export GFMRAG_CHAT_KEY='your-api-key'")
    
    print("\n   📋 完整的第三方服务配置示例:")
    print("      export GFMRAG_CHAT_PROVIDER='third-party'")
    print("      export GFMRAG_CHAT_MODEL_NAME='Qwen3-VL-30B-A3B-Instruct-FP8'")
    print("      export GFMRAG_CHAT_BASE_URL='http://192.168.110.11:8888/v1'")
    print("      export GFMRAG_CHAT_KEY='your-api-key'  # 可选")


def print_summary(results: Dict[str, bool]):
    """打印测试结果摘要"""
    print("\n" + "=" * 60)
    print("📊 测试结果摘要")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！GFM-RAG配置正常。")
    else:
        print("⚠️ 部分测试失败。请检查上述建议并修复配置问题。")
    print("=" * 60)


def main():
    """主函数"""
    print("启动 GFM-RAG 环境变量配置验证...")
    
    results = {}
    
    # 1. 检查环境变量
    env_status = check_environment_variables()
    
    # 2. 检查配置管理器
    config_manager_ok = check_configuration_manager() is not None
    results["配置管理器"] = config_manager_ok
    
    # 3. 测试Hydra配置加载
    hydra_ok = test_hydra_config_loading()
    results["Hydra配置加载"] = hydra_ok
    
    # 4. 测试模型初始化
    model_ok = test_model_initialization()
    results["模型初始化"] = model_ok
    
    # 5. 生成建议
    generate_configuration_recommendations(env_status)
    
    # 6. 打印摘要
    print_summary(results)
    
    # 返回适当的退出码
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())