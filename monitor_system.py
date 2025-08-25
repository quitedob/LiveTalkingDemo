#!/usr/bin/env python3
"""
LiveTalking系统监控脚本
用于监控WebRTC连接、会话状态、资源使用等
"""
import asyncio
import aiohttp
import json
import time
import psutil
import sys
from datetime import datetime


class SystemMonitor:
    def __init__(self, base_url="http://localhost:8010"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_server_health(self):
        """检查服务器健康状态"""
        try:
            async with self.session.get(f"{self.base_url}/") as response:
                return response.status == 200
        except Exception as e:
            print(f"❌ 服务器连接失败: {e}")
            return False
    
    async def check_llm_providers(self):
        """检查LLM提供商状态"""
        try:
            async with self.session.get(f"{self.base_url}/llm/providers") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"🤖 LLM状态:")
                    print(f"   当前提供商: {data.get('current_provider', 'N/A')}")
                    print(f"   可用提供商: {data.get('available_providers', [])}")
                    
                    client_info = data.get('client_info', {})
                    if client_info:
                        print(f"   模型: {client_info.get('model', 'N/A')}")
                        print(f"   类型: {client_info.get('type', 'N/A')}")
                        if 'api_key_configured' in client_info:
                            print(f"   API密钥: {'已配置' if client_info['api_key_configured'] else '未配置'}")
                    return True
                else:
                    print(f"❌ LLM状态检查失败: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ LLM状态检查异常: {e}")
            return False
    
    async def check_rag_config(self):
        """检查RAG配置"""
        try:
            async with self.session.get(f"{self.base_url}/config/get") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"📚 RAG状态:")
                    print(f"   RAG模式: {'开启' if data.get('use_rag') else '关闭'}")
                    print(f"   当前知识库: {data.get('current_kb', '无')}")
                    return True
                else:
                    print(f"❌ RAG配置检查失败: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ RAG配置检查异常: {e}")
            return False
    
    async def check_knowledge_bases(self):
        """检查知识库列表"""
        try:
            async with self.session.get(f"{self.base_url}/kb/list") as response:
                if response.status == 200:
                    data = await response.json()
                    kbs = data.get('knowledge_bases', [])
                    print(f"📖 知识库: {len(kbs)} 个")
                    for kb in kbs:
                        print(f"   - {kb}")
                    return True
                else:
                    print(f"❌ 知识库检查失败: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 知识库检查异常: {e}")
            return False
    
    async def test_llm_connection(self):
        """测试LLM连接"""
        try:
            test_payload = {"query": "Hello, please respond with 'System test OK'"}
            async with self.session.post(
                f"{self.base_url}/llm/test",
                json=test_payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success'):
                        print(f"✅ LLM连接测试成功")
                        print(f"   提供商: {data.get('provider')}")
                        print(f"   响应: {data.get('response', '')[:100]}...")
                        return True
                    else:
                        print(f"❌ LLM测试失败: {data.get('error')}")
                        return False
                else:
                    print(f"❌ LLM测试请求失败: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ LLM测试异常: {e}")
            return False
    
    def check_system_resources(self):
        """检查系统资源使用"""
        print(f"💻 系统资源:")
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"   CPU使用率: {cpu_percent:.1f}%")
        
        # 内存使用
        memory = psutil.virtual_memory()
        print(f"   内存使用: {memory.percent:.1f}% ({memory.used // 1024 // 1024}MB / {memory.total // 1024 // 1024}MB)")
        
        # 磁盘使用
        disk = psutil.disk_usage('/')
        print(f"   磁盘使用: {disk.percent:.1f}% ({disk.used // 1024 // 1024 // 1024}GB / {disk.total // 1024 // 1024 // 1024}GB)")
        
        # GPU使用（如果可用）
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(f"   GPU{i} 使用: {gpu.load * 100:.1f}% 内存: {gpu.memoryUtil * 100:.1f}%")
        except ImportError:
            print("   GPU监控: 未安装GPUtil")
        except Exception as e:
            print(f"   GPU监控: 检查失败 ({e})")
    
    def check_network_ports(self):
        """检查网络端口"""
        print(f"🌐 网络端口:")
        
        # 检查常用端口
        ports_to_check = [8010, 11434, 8080]  # LiveTalking, Ollama, TTS
        
        for port in ports_to_check:
            connections = [conn for conn in psutil.net_connections() 
                         if conn.laddr.port == port and conn.status == 'LISTEN']
            if connections:
                print(f"   端口 {port}: ✅ 监听中")
            else:
                print(f"   端口 {port}: ❌ 未监听")
    
    async def run_full_check(self):
        """运行完整的系统检查"""
        print(f"🔍 LiveTalking系统监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 服务器健康检查
        server_ok = await self.check_server_health()
        if server_ok:
            print("✅ 服务器运行正常")
        else:
            print("❌ 服务器无法访问")
            return
        
        print()
        
        # LLM检查
        await self.check_llm_providers()
        print()
        await self.test_llm_connection()
        print()
        
        # RAG检查
        await self.check_rag_config()
        print()
        await self.check_knowledge_bases()
        print()
        
        # 系统资源检查
        self.check_system_resources()
        print()
        
        # 网络端口检查
        self.check_network_ports()
        print()
        
        print("=" * 60)
        print("✅ 系统检查完成")


async def main():
    """主函数"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8010"
    
    print(f"使用服务器地址: {base_url}")
    print("提示: 可以通过参数指定服务器地址，如: python monitor_system.py http://localhost:8010")
    print()
    
    async with SystemMonitor(base_url) as monitor:
        await monitor.run_full_check()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n监控已停止")
    except Exception as e:
        print(f"监控异常: {e}")