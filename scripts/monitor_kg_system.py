#!/usr/bin/env python3
"""
KG系统监控脚本 - 简化版

提供系统健康检查和基础监控功能。
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_system_health():
    """检查系统健康状态"""
    try:
        from gfmrag.kg_monitoring import get_monitoring_system
        from gfmrag.token_manager import get_token_manager
        
        monitoring = get_monitoring_system()
        token_manager = get_token_manager()
        
        # 收集状态信息
        dashboard_data = monitoring.get_dashboard_data()
        token_stats = token_manager.get_allocation_stats()
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'monitoring_active': monitoring.is_running,
            'token_allocations': token_stats.get('total_allocations', 0),
            'active_alerts': len(dashboard_data.get('active_alerts', [])),
            'error_summary': dashboard_data.get('error_summary', {}),
            'recommendations': dashboard_data.get('recommendations', [])
        }
        
        return health_report
        
    except Exception as e:
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KG系统监控工具')
    parser.add_argument('--check', action='store_true', help='执行健康检查')
    parser.add_argument('--watch', type=int, metavar='SECONDS', help='持续监控，指定检查间隔')
    parser.add_argument('--json', action='store_true', help='JSON格式输出')
    
    args = parser.parse_args()
    
    if args.check or args.watch:
        if args.watch:
            # 持续监控模式
            print(f"开始持续监控，检查间隔: {args.watch}秒")
            try:
                while True:
                    health = check_system_health()
                    if args.json:
                        print(json.dumps(health, indent=2, ensure_ascii=False))
                    else:
                        print(f"[{health['timestamp']}] 状态: {health['status']}")
                        if health.get('error'):
                            print(f"错误: {health['error']}")
                        else:
                            print(f"监控运行: {health['monitoring_active']}")
                            print(f"Token分配: {health['token_allocations']}")
                            print(f"活动告警: {health['active_alerts']}")
                    
                    time.sleep(args.watch)
            except KeyboardInterrupt:
                print("\n监控已停止")
        else:
            # 单次检查
            health = check_system_health()
            if args.json:
                print(json.dumps(health, indent=2, ensure_ascii=False))
            else:
                print("=== KG系统健康检查 ===")
                print(f"时间: {health['timestamp']}")
                print(f"状态: {health['status']}")
                
                if health.get('error'):
                    print(f"错误: {health['error']}")
                else:
                    print(f"监控系统: {'运行中' if health['monitoring_active'] else '已停止'}")
                    print(f"Token分配次数: {health['token_allocations']}")
                    print(f"活动告警数: {health['active_alerts']}")
                    
                    error_summary = health.get('error_summary', {})
                    if error_summary.get('total_errors', 0) > 0:
                        print(f"总错误数: {error_summary['total_errors']}")
                        print(f"错误率: {error_summary.get('error_rate', 0):.2f}/分钟")
                    
                    recommendations = health.get('recommendations', [])
                    if recommendations:
                        print("建议:")
                        for rec in recommendations:
                            print(f"  - {rec}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()