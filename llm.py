# /workspace/LiveTalking/llm.py
import aiohttp
import json
import re
from logger import logger

class LLMClient:
    """
    一个用于与Ollama语言模型进行异步交互的客户端。
    - 新增：流式<strong>思考链</strong>有状态过滤（跨分片的 <think>…</think> 也能正确剔除）
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

        # --- 新增：跨分片 <think> 过滤状态 ---
        # inside_think=True 表示当前处于 <think>…</think> 屏蔽区间
        self._inside_think = False

    @staticmethod
    def _clean(text: str) -> str:
        """
        中文输出清理增强版（用于标签剔除后的收尾清洁）：
        - 删除常见元叙述开场白
        - 清理 /nothink /no_think 等控制片段
        - 清理残留 <think>...</think>（同分片成对出现的兜底处理）
        - 压缩多余空白
        """
        if not text:
            return text

        # ① 兜底：清理同一分片内成对出现的 <think>…</think>（跨分片交给状态机）
        text = re.sub(r"(?is)<\s*think\b[^>]*>.*?</\s*think\s*>", "", text)

        # ② 清理控制指令残留（不让它出现在终端）
        text = re.sub(r"/\s*no_?think", "", text, flags=re.I)

        # ③ 清理元叙述开场白（仅清理段首/句首，避免误伤）
        bad_starts = [
            r"^(根据|依据|参考|结合)(你|您)(的|提供的)?.{0,30}[，。:\s]",
            r"^(由于|因|鉴于)(你|您)(的|提供的)?.{0,30}[，。:\s]",
            r"^很抱歉[^。]*[。!！]?"
        ]
        for pat in bad_starts:
            text = re.sub(pat, "", text.strip(), flags=re.IGNORECASE)

        # ④ 特例句式
        text = re.sub(r"由于您提供的参考信息仅包含[^。]*。?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"根据你的[^。]*。?", "", text, flags=re.IGNORECASE)

        # ⑤ 轻度字符清洗（保留中英数与常用标点；不要过度清洗，避免误删内容）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。.,!！?？；;：:\s（）()\[\]【】“”"\'\-_/]', '', text)

        # ⑥ 压缩空白
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _strip_think_stream(self, chunk: str) -> str:
        """
        【核心改动】流式“思考链”过滤：
        - 逐分片处理；若此前遇到过 <think> 未闭合，则直接丢弃直到遇到 </think>
        - 支持同一分片出现多个 <think>…</think> 块
        - 匹配大小写与可选空白：<think ...> 与 </think>
        """
        if not chunk:
            return chunk

        s = chunk
        out = []
        i = 0
        # 如果已经在 think 块内，先尝试找到结束标签
        if self._inside_think:
            m_end = re.search(r'(?is)</\s*think\s*>', s)
            if not m_end:
                # 整个分片都在 <think> 内，全部丢弃
                return ""
            # 丢弃到结束标签为止
            i = m_end.end()
            self._inside_think = False

        # 常规扫描，剔除所有 <think>…</think>，不跨出当前分片的内容保留
        while i < len(s):
            m_open = re.search(r'(?is)<\s*think\b[^>]*>', s[i:])
            if not m_open:
                out.append(s[i:])
                break

            open_start = i + m_open.start()
            open_end = i + m_open.end()
            # 保留打开标签之前的正常内容
            out.append(s[i:open_start])

            # 尝试在本分片内寻找关闭
            m_close = re.search(r'(?is)</\s*think\s*>', s[open_end:])
            if not m_close:
                # 未找到关闭标签：从这里开始进入“屏蔽模式”，丢弃余下内容
                self._inside_think = True
                break
            # 跳过整个 think 块
            i = open_end + m_close.end()

        return "".join(out)

    async def ask(self, user_prompt: str, system_prompt_override: str = None):
        """
        向Ollama发送请求并流式返回响应。
        - 新增：在每个分片上先做 _strip_think_stream()（跨分片过滤 <think>），再 _clean()
        :param user_prompt: 用户的提问。
        :param system_prompt_override: 可选，用于临时覆盖默认系统提示词。
        """
        # 决定最终使用的系统提示词（None=用默认，""=不发送system）
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
                async with session.post(self.url, json=payload, timeout=self.timeout) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API请求失败，状态码: {response.status}, 错误: {error_text}")
                        yield f"Error: LLM service request failed with status {response.status}"
                        return

                    async for line in response.content:
                        if not line:
                            continue
                        try:
                            data = json.loads(line.decode('utf-8'))

                            # 兼容：Ollama chat 流通常在 data['message']['content']；
                            # 某些实现也可能在 data['response']。
                            raw_chunk = None
                            if isinstance(data.get('message'), dict):
                                raw_chunk = data['message'].get('content')
                            if raw_chunk is None and 'response' in data:
                                raw_chunk = data['response']

                            if raw_chunk:
                                # 先做跨分片思考链剔除，再做常规清理
                                visible = self._strip_think_stream(raw_chunk)
                                cleaned_chunk = self._clean(visible)
                                if cleaned_chunk:
                                    yield cleaned_chunk

                            # 错误与结束信号
                            if data.get('done') and data.get('error'):
                                logger.error(f"Ollama stream error: {data['error']}")

                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON from Ollama stream: {line}")
                            continue

        except aiohttp.ClientConnectorError as e:
            logger.error(f"无法连接ollama服务 at {self.url}: {e}")
            yield f"Error: Could not connect to LLM service."
        except Exception as e:
            logger.exception("一个不知道的错误发生当与ollama交互的时候:")
            yield f"Error: An unknown error occurred."
