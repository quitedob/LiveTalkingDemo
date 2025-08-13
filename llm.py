# /workspace/LiveTalking/llm.py
import aiohttp
import json
import re
from logger import logger

class LLMClient:
    """
    一个用于与Ollama语言模型进行异步交互的客户端。
    An asynchronous client for interacting with the Ollama language model.
    """
    def __init__(self, url: str, model: str, system_prompt: str, timeout: int = 120):
        """
        初始化LLM客户端。
        :param url: Ollama API的URL。
        :param model: 要使用的模型名称。
        :param system_prompt: 默认的系统提示词。
        :param timeout: 请求超时时间（秒）。
        """
        self.url = url
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    @staticmethod
    def _clean(text: str) -> str:
        """
        中文输出清理增强版：
        - 删除常见元叙述开场白
        - 移除思考标签和多余标点
        - 避免"很抱歉/由于您提供的参考信息仅包含…"等刺耳句式
        """
        if not text:
            return text
            
        # ① 删除 <think>…</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        # ② 删除 /nothink 等残留
        text = re.sub(r"/nothink", "", text, flags=re.I)
        
        # ③ 清理元叙述开场白（仅清理段首/句首）
        bad_starts = [
            r"^(根据|依据|参考|结合)(你|您)(的|提供的)?.{0,30}[，。:\s]",
            r"^(由于|因|鉴于)(你|您)(的|提供的)?.{0,30}[，。:\s]",
            r"^很抱歉[^。]*[。!！]?"
        ]
        for pat in bad_starts:
            text = re.sub(pat, "", text.strip(), flags=re.IGNORECASE)
        
        # ④ 特例：清理日志中出现的固定刺耳句式
        text = re.sub(r"由于您提供的参考信息仅包含[^。]*。?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"根据你的[^。]*。?", "", text, flags=re.IGNORECASE)
        
        # ⑤ 移除除中文、字母、数字、空格、常用标点外的符号
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。.,!！?？；;：:\s]', '', text)
        
        # ⑥ 压缩多余空行/空格
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        
        return text.strip()

    async def ask(self, user_prompt: str, system_prompt_override: str = None):
        """
        向Ollama发送请求并流式返回响应。
        允许临时覆盖实例的系统提示词。
        :param user_prompt: 用户的提问。
        :param system_prompt_override: 可选，用于临时覆盖默认系统提示词。
        """
        # 决定最终使用的系统提示词
        # 如果 override 是 None，则使用默认值。如果它是空字符串 ""，则表示不使用系统提示词。
        final_system_prompt = system_prompt_override if system_prompt_override is not None else self.system_prompt
        
        messages = []
        if final_system_prompt: # 只有在提示词非空时才添加
            messages.append({"role": "system", "content": final_system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload, timeout=self.timeout) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API请求失败，状态码: {response.status}, 错误: {error_text}")
                        yield f"Error: LLM service request failed with status {response.status}"
                        return
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'message' in data and 'content' in data['message']:
                                    cleaned_chunk = self._clean(data['message']['content'])
                                    if cleaned_chunk: # 确保清洗后不为空
                                        yield cleaned_chunk
                                if data.get('done') and data.get('error'):
                                    logger.error(f"Ollama stream error: {data['error']}")
                            except json.JSONDecodeError:
                                logger.warning(f"Could not decode JSON from Ollama stream: {line}")
                                continue
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Could not connect to Ollama service at {self.url}: {e}")
            yield f"Error: Could not connect to LLM service."
        except Exception as e:
            logger.exception("An unknown error occurred while interacting with Ollama:")
            yield f"Error: An unknown error occurred."
