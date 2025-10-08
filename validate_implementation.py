#!/usr/bin/env python3
"""
验证LangChain配置优化实现的简单脚本
"""

import sys
import os

# 添加项目路径到Python路径
sys.path.insert(0, '/data/workspace/gfm-rag')

def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")
    
    try:
        from gfmrag.config_manager import ConfigurationManager, ChatConfig, EmbeddingConfig
        print("✓ config_manager 导入成功")
    except Exception as e:
        print(f"✗ config_manager 导入失败: {e}")
        return False
    
    try:
        from gfmrag.langchain_factory import LangChainModelFactory
        print("✓ langchain_factory 导入成功")
    except Exception as e:
        print(f"✗ langchain_factory 导入失败: {e}")
        return False
    
    try:
        from gfmrag.embedding_factory import EmbeddingModelFactory
        print("✓ embedding_factory 导入成功")
    except Exception as e:
        print(f"✗ embedding_factory 导入失败: {e}")
        return False
    
    try:
        from gfmrag.config_validator import ConfigValidator
        print("✓ config_validator 导入成功")
    except Exception as e:
        print(f"✗ config_validator 导入失败: {e}")
        return False
    
    try:
        from gfmrag.error_handler import RetryHandler, FallbackManager
        print("✓ error_handler 导入成功")
    except Exception as e:
        print(f"✗ error_handler 导入失败: {e}")
        return False
    
    return True

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        from gfmrag.config_manager import ConfigurationManager, ChatConfig
        
        # 测试配置创建
        config = ChatConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0
        )
        print("✓ ChatConfig 创建成功")
        
        # 测试配置管理器
        config_manager = ConfigurationManager()
        print("✓ ConfigurationManager 创建成功")
        
        # 测试配置验证
        from gfmrag.config_validator import ConfigValidator
        validator = ConfigValidator()
        result = validator.validate_chat_config(config)
        print(f"✓ 配置验证成功，结果: {'通过' if result.is_valid else '失败'}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def test_file_structure():
    """检查文件结构"""
    print("\n检查文件结构...")
    
    required_files = [
        'gfmrag/config_manager.py',
        'gfmrag/langchain_factory.py', 
        'gfmrag/embedding_factory.py',
        'gfmrag/config_validator.py',
        'gfmrag/error_handler.py',
        'configs/langchain_config.yaml',
        'configs/vllm_config.yaml',
        'configs/llama_server_config.yaml',
        'docs/langchain_optimization_guide.md',
        'MIGRATION_GUIDE.md',
        '.env.example'
    ]
    
    base_path = '/data/workspace/gfm-rag'
    all_exist = True
    
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} 不存在")
            all_exist = False
    
    return all_exist

def main():
    """主测试函数"""
    print("LangChain配置优化实现验证")
    print("=" * 50)
    
    # 测试文件结构
    structure_ok = test_file_structure()
    
    # 测试导入
    imports_ok = test_imports()
    
    # 测试基本功能
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("验证结果:")
    print(f"文件结构: {'✓ 通过' if structure_ok else '✗ 失败'}")
    print(f"模块导入: {'✓ 通过' if imports_ok else '✗ 失败'}")
    print(f"基本功能: {'✓ 通过' if functionality_ok else '✗ 失败'}")
    
    if all([structure_ok, imports_ok, functionality_ok]):
        print("\n🎉 所有验证都通过！LangChain配置优化实现成功！")
        return 0
    else:
        print("\n❌ 部分验证失败，请检查实现。")
        return 1

if __name__ == "__main__":
    sys.exit(main())