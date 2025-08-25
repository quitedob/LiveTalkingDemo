"""
Ollama LLM Client Implementation
"""
import aiohttp
import json
from typing import AsyncGenerator, Optional
from .base import BaseLLMClient
from logger import logger


class OllamaClient(BaseLLMClient):
    """
    Ollama LLM client implementation
    """
    
    def __init__(self, url: str, model: str, system_prompt: str, timeout: int = 120):
        super().__init__(model, system_prompt, timeout)
        self.url = url
        self.timeout_config = aiohttp.ClientTimeout(total=timeout)
    
    async def ask(self, user_prompt: str, system_prompt_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Send request to Ollama and return streaming response
        """
        # Reset think filtering state for new conversation
        self._inside_think = False
        
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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload, timeout=self.timeout_config) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API request failed, status: {response.status}, error: {error_text}")
                        yield f"Error: LLM service request failed with status {response.status}"
                        return

                    async for line in response.content:
                        if not line:
                            continue
                        try:
                            data = json.loads(line.decode('utf-8'))

                            # Compatible: Ollama chat stream usually in data['message']['content']
                            # Some implementations may also be in data['response']
                            raw_chunk = None
                            if isinstance(data.get('message'), dict):
                                raw_chunk = data['message'].get('content')
                            if raw_chunk is None and 'response' in data:
                                raw_chunk = data['response']

                            if raw_chunk:
                                # First do cross-chunk thinking chain removal, then regular cleaning
                                visible = self._strip_think_stream(raw_chunk)
                                cleaned_chunk = self._clean(visible)
                                if cleaned_chunk:
                                    yield cleaned_chunk

                            # Error and end signals
                            if data.get('done') and data.get('error'):
                                logger.error(f"Ollama stream error: {data['error']}")

                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON from Ollama stream: {line}")
                            continue

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Cannot connect to Ollama service at {self.url}: {e}")
            yield f"Error: Could not connect to LLM service."
        except Exception as e:
            logger.exception("Unknown error occurred when interacting with Ollama:")
            yield f"Error: An unknown error occurred."
    
    def get_client_info(self) -> dict:
        """
        Get Ollama client information
        """
        return {
            "type": "ollama",
            "url": self.url,
            "model": self.model,
            "timeout": self.timeout
        }