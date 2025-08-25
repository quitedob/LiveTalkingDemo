"""
DeepSeek LLM Client Implementation
"""
import aiohttp
import json
import os
from typing import AsyncGenerator, Optional
from .base import BaseLLMClient
from logger import logger


class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek LLM client implementation using OpenAI-compatible API
    """
    
    def __init__(self, model: str = "deepseek-chat", system_prompt: str = "", timeout: int = 120):
        super().__init__(model, system_prompt, timeout)
        self.base_url = "https://api.deepseek.com/chat/completions"
        self.api_key = self._load_api_key()
        self.timeout_config = aiohttp.ClientTimeout(total=timeout)
    
    def _load_api_key(self) -> str:
        """
        Load DeepSeek API key from .env file
        """
        try:
            # Try to load from .env file
            env_path = ".env"
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('DEEPSEEK_API_KEY='):
                            return line.split('=', 1)[1].strip().strip('"\'')
            
            # Fallback to environment variable
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                logger.warning("DeepSeek API key not found in .env file or environment variables")
                return ""
            return api_key
        except Exception as e:
            logger.error(f"Error loading DeepSeek API key: {e}")
            return ""
    
    async def ask(self, user_prompt: str, system_prompt_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Send request to DeepSeek and return streaming response
        """
        # Reset think filtering state for new conversation
        self._inside_think = False
        
        if not self.api_key:
            yield "Error: DeepSeek API key not configured"
            return
        
        # Determine final system prompt
        final_system_prompt = system_prompt_override if system_prompt_override is not None else self.system_prompt

        messages = []
        if final_system_prompt:
            messages.append({"role": "system", "content": final_system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, json=payload, headers=headers, timeout=self.timeout_config) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API request failed, status: {response.status}, error: {error_text}")
                        yield f"Error: DeepSeek API request failed with status {response.status}"
                        return

                    async for line in response.content:
                        if not line:
                            continue
                        
                        line_str = line.decode('utf-8').strip()
                        if not line_str or not line_str.startswith('data: '):
                            continue
                        
                        # Remove 'data: ' prefix
                        json_str = line_str[6:]
                        
                        # Skip [DONE] marker
                        if json_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(json_str)
                            
                            # Extract content from DeepSeek response
                            if 'choices' in data and len(data['choices']) > 0:
                                choice = data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    raw_chunk = choice['delta']['content']
                                    
                                    if raw_chunk:
                                        # First do cross-chunk thinking chain removal, then regular cleaning
                                        visible = self._strip_think_stream(raw_chunk)
                                        cleaned_chunk = self._clean(visible)
                                        if cleaned_chunk:
                                            yield cleaned_chunk

                        except json.JSONDecodeError as e:
                            logger.warning(f"Could not decode JSON from DeepSeek stream: {json_str}, error: {e}")
                            continue

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Cannot connect to DeepSeek service: {e}")
            yield f"Error: Could not connect to DeepSeek service."
        except Exception as e:
            logger.exception("Unknown error occurred when interacting with DeepSeek:")
            yield f"Error: An unknown error occurred."
    
    def get_client_info(self) -> dict:
        """
        Get DeepSeek client information
        """
        return {
            "type": "deepseek",
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "api_key_configured": bool(self.api_key)
        }