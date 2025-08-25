"""
LLM Manager for Hot-Swapping between different LLM providers
"""
import asyncio
from typing import Optional, Dict, Any
from .base import BaseLLMClient
from .ollama_client import OllamaClient
from .deepseek_client import DeepSeekClient
from logger import logger


class LLMManager:
    """
    Manager for hot-swapping between different LLM clients
    """
    
    def __init__(self):
        self.current_client: Optional[BaseLLMClient] = None
        self.clients: Dict[str, BaseLLMClient] = {}
        self.current_provider = "ollama"  # Default provider
        self._lock = asyncio.Lock()
    
    def register_ollama_client(self, url: str, model: str, system_prompt: str, timeout: int = 120):
        """
        Register Ollama client
        """
        client = OllamaClient(url, model, system_prompt, timeout)
        self.clients["ollama"] = client
        if self.current_provider == "ollama":
            self.current_client = client
        logger.info(f"Registered Ollama client: {url}, model: {model}")
    
    def register_deepseek_client(self, model: str = "deepseek-chat", system_prompt: str = "", timeout: int = 120):
        """
        Register DeepSeek client
        """
        client = DeepSeekClient(model, system_prompt, timeout)
        self.clients["deepseek"] = client
        if self.current_provider == "deepseek":
            self.current_client = client
        logger.info(f"Registered DeepSeek client, model: {model}")
    
    async def switch_provider(self, provider: str) -> bool:
        """
        Switch to a different LLM provider
        """
        async with self._lock:
            if provider not in self.clients:
                logger.error(f"Provider {provider} not registered")
                return False
            
            if provider == self.current_provider:
                logger.info(f"Already using provider {provider}")
                return True
            
            old_provider = self.current_provider
            old_client = self.current_client
            
            # 切换到新的提供商
            self.current_provider = provider
            self.current_client = self.clients[provider]
            
            # 如果旧客户端有清理方法，调用它
            if hasattr(old_client, 'cleanup'):
                try:
                    await old_client.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up old LLM client: {e}")
            
            logger.info(f"Switched LLM provider from {old_provider} to {provider}")
            return True
    
    def get_current_provider(self) -> str:
        """
        Get current provider name
        """
        return self.current_provider
    
    def get_available_providers(self) -> list:
        """
        Get list of available providers
        """
        return list(self.clients.keys())
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        Get current client information
        """
        if not self.current_client:
            return {"error": "No client configured"}
        
        info = self.current_client.get_client_info()
        info["current_provider"] = self.current_provider
        info["available_providers"] = self.get_available_providers()
        return info
    
    async def ask(self, user_prompt: str, system_prompt_override: Optional[str] = None):
        """
        Send request to current LLM client
        """
        if not self.current_client:
            yield "Error: No LLM client configured"
            return
        
        async for chunk in self.current_client.ask(user_prompt, system_prompt_override):
            yield chunk
    
    def update_system_prompt(self, system_prompt: str):
        """
        Update system prompt for all registered clients
        """
        for client in self.clients.values():
            client.system_prompt = system_prompt
        logger.info("Updated system prompt for all LLM clients")