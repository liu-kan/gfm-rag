#!/usr/bin/env python3
"""
简化的配置测试脚本

用于快速验证环境变量配置是否正常工作
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_config():
    """基本配置测试"""
    print("🔧 测试基本配置...")
    
    # 测试环境变量
    env_vars = [
        'GFMRAG_CHAT_PROVIDER',
        'GFMRAG_CHAT_MODEL_NAME',
        'GFMRAG_CHAT_BASE_URL',
        'GFMRAG_CHAT_KEY'
    ]
    
    print("环境变量状态:")
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if var == 'GFMRAG_CHAT_KEY':
                print(f"  ✅ {var}: 已设置")
            else:
                print(f"  ✅ {var}: {value}")
        else:
            print(f"  ❌ {var}: 未设置")
    
    return True

def test_yaml_config():
    """测试YAML配置文件"""
    print("\n📄 测试YAML配置...")
    
    try:
        from omegaconf import OmegaConf
        
        # 测试 NER 配置
        ner_config_path = project_root / "gfmrag/workflow/config/ner_model/llm_ner_model.yaml"
        if ner_config_path.exists():
            config = OmegaConf.load(ner_config_path)
            print(f"  ✅ NER配置: {config}")
        else:
            print("  ❌ NER配置文件不存在")
        
        # 测试 OpenIE 配置
        openie_config_path = project_root / "gfmrag/workflow/config/openie_model/llm_openie_model.yaml"
        if openie_config_path.exists():
            config = OmegaConf.load(openie_config_path)
            print(f"  ✅ OpenIE配置: {config}")
        else:
            print("  ❌ OpenIE配置文件不存在")
            
        return True
        
    except Exception as e:
        print(f"  ❌ YAML配置测试失败: {e}")
        return False

def test_model_imports():
    """测试模型导入"""
    print("\n🔌 测试模型导入...")
    
    try:
        from gfmrag.kg_construction.ner_model.llm_ner_model import LLMNERModel
        print("  ✅ LLMNERModel 导入成功")
        
        from gfmrag.kg_construction.openie_model.llm_openie_model import LLMOPENIEModel
        print("  ✅ LLMOPENIEModel 导入成功")
        
        from gfmrag.kg_construction.langchain_util import init_langchain_model
        print("  ✅ init_langchain_model 导入成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型导入失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始配置测试...\n")
    
    tests = [
        ("基本配置", test_basic_config),
        ("YAML配置", test_yaml_config),
        ("模型导入", test_model_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 测试结果:")
    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("⚠️ 部分测试失败，请检查配置。")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())