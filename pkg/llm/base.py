"""
Base LLM Client Abstract Class
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
import re


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients
    """
    
    def __init__(self, model: str, system_prompt: str, timeout: int = 120):
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout
        
        # Cross-chunk <think> filtering state
        self._inside_think = False
    
    @staticmethod
    def _clean(text: str) -> str:
        """
        Enhanced Chinese output cleaning (for post-tag removal cleanup):
        - Remove common meta-narrative openings
        - Clean /nothink /no_think control fragments
        - Clean residual <think>...</think> (fallback for same-chunk pairs)
        - Compress excess whitespace
        """
        if not text:
            return text

        # ① Fallback: clean same-chunk <think>…</think> pairs (cross-chunk handled by state machine)
        text = re.sub(r"(?is)<\s*think\b[^>]*>.*?</\s*think\s*>", "", text)

        # ② Clean control instruction residue (don't let it appear in terminal)
        text = re.sub(r"/\s*no_?think", "", text, flags=re.I)

        # ③ Clean meta-narrative openings (only clean at beginning of paragraph/sentence to avoid false positives)
        bad_starts = [
            r"^(根据|依据|参考|结合)(你|您)(的|提供的)?.{0,30}[，。:\s]",
            r"^(由于|因|鉴于)(你|您)(的|提供的)?.{0,30}[，。:\s]",
            r"^很抱歉[^。]*[。!！]?"
        ]
        for pat in bad_starts:
            text = re.sub(pat, "", text.strip(), flags=re.IGNORECASE)

        # ④ Special sentence patterns
        text = re.sub(r"由于您提供的参考信息仅包含[^。]*。?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"根据你的[^。]*。?", "", text, flags=re.IGNORECASE)

        # ⑤ Light character cleaning (preserve Chinese, English, numbers and common punctuation)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。.,!！?？；;：:\s（）()\[\]【】"""\'\-_/]', '', text)

        # ⑥ Compress whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _strip_think_stream(self, chunk: str) -> str:
        """
        Stream "thinking chain" filtering:
        - Process chunk by chunk; if previously encountered unclosed <think>, discard until </think>
        - Support multiple <think>…</think> blocks in same chunk
        - Match case-insensitive with optional whitespace: <think ...> and </think>
        """
        if not chunk:
            return chunk

        s = chunk
        out = []
        i = 0
        
        # If already inside think block, try to find end tag first
        if self._inside_think:
            m_end = re.search(r'(?is)</\s*think\s*>', s)
            if not m_end:
                # Entire chunk is inside <think>, discard all
                return ""
            # Discard up to end tag
            i = m_end.end()
            self._inside_think = False

        # Regular scan, remove all <think>…</think>, preserve content not crossing current chunk
        while i < len(s):
            m_open = re.search(r'(?is)<\s*think\b[^>]*>', s[i:])
            if not m_open:
                out.append(s[i:])
                break

            open_start = i + m_open.start()
            open_end = i + m_open.end()
            # Preserve normal content before opening tag
            out.append(s[i:open_start])

            # Try to find closing tag in this chunk
            m_close = re.search(r'(?is)</\s*think\s*>', s[open_end:])
            if not m_close:
                # No closing tag found: enter "masking mode" from here, discard remaining content
                self._inside_think = True
                break
            # Skip entire think block
            i = open_end + m_close.end()

        return "".join(out)
    
    @abstractmethod
    async def ask(self, user_prompt: str, system_prompt_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Send request to LLM and return streaming response
        """
        pass
    
    @abstractmethod
    def get_client_info(self) -> dict:
        """
        Get client information for debugging/monitoring
        """
        pass