# /workspace/LiveTalking/pkg/rag/rag_core.py
import asyncio
import json
from pathlib import Path
from typing import Tuple, Optional

from logger import logger
from pkg.rag.config import (DEFAULT_RAG_MODE, DYNAMIC_PROMPT_FILE)

class RAGCore:
    """
    RAG核心逻辑类，负责管理RAG状态、构建提示词。
    """

    def __init__(self, knowledge_base, llm_client):
        """
        初始化RAG核心。
        :param knowledge_base: 一个KnowledgeBase的实例。
        :param llm_client: 一个LLMClient的实例。
        """
        self.kb = knowledge_base
        self.llm_client = llm_client # 注入共享的LLM客户端
        self.use_rag = DEFAULT_RAG_MODE
        self.current_kb = None
        self.system_prompt = self._load_system_prompt()
        logger.info("RAG核心初始化完成，并已连接到共享LLM客户端。")

    def _load_system_prompt(self) -> str:
        """从文件加载动态提示词，如果文件不存在则创建默认值。"""
        try:
            if DYNAMIC_PROMPT_FILE.exists():
                with open(DYNAMIC_PROMPT_FILE, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            else:
                default_dynamic_prompt = "请根据我提供的上下文，以专业、严谨的风格回答问题。"
                with open(DYNAMIC_PROMPT_FILE, 'w', encoding='utf-8') as f:
                    f.write(default_dynamic_prompt)
                return default_dynamic_prompt
        except Exception as e:
            logger.error(f"加载或创建动态提示词文件失败: {e}")
            return "请根据上下文回答问题。"

    def update_system_prompt(self, new_prompt: str):
        """更新动态系统提示词并保存到文件。"""
        try:
            with open(DYNAMIC_PROMPT_FILE, 'w', encoding='utf-8') as f:
                f.write(new_prompt)
            self.system_prompt = new_prompt
            logger.info(f"系统提示词已更新为: '{new_prompt[:50]}...'")
            return True
        except Exception as e:
            logger.error(f"更新系统提示词文件失败: {e}")
            return False

    def set_rag_mode(self, use_rag: bool):
        """设置RAG模式的开关。"""
        self.use_rag = use_rag
        logger.info(f"RAG 模式已切换为: {'ON' if use_rag else 'OFF'}")

    def switch_kb(self, kb_name: str) -> bool:
        """切换当前使用的知识库。"""
        existing_kbs = self.kb.list_kbs()
        if kb_name in existing_kbs:
            self.current_kb = kb_name
            logger.info(f"当前知识库已切换为: '{kb_name}'")
            return True
        elif kb_name == "" or kb_name is None:
            self.current_kb = None
            logger.info("已取消使用知识库。")
            return True
        else:
            logger.warning(f"尝试切换到不存在的知识库: '{kb_name}'")
            return False

    def set_current_kb(self, kb_name: Optional[str]):
        """
        安全地设置当前知识库的名称。
        如果名称无效或为空，则禁用知识库。
        """
        if kb_name and kb_name in self.kb.list_kbs():
            self.current_kb = kb_name
            logger.info(f"RAG对话使用的知识库已设置为: '{kb_name}'")
        else:
            self.current_kb = None
            if kb_name:
                logger.warning(f"尝试为RAG对话设置不存在的知识库 '{kb_name}'，将不使用知识库。")
            else:
                 logger.info("RAG对话将不使用特定知识库。")

    def _build_prompt(self, user_query: str, context: str) -> Tuple[str, Optional[str]]:
        """
        根据模板构建最终的用户问题和可选的系统提示词覆盖。
        :return: (final_user_query, system_prompt_override)
        """
        additional_requirements = self.system_prompt
        
        if self.use_rag and context:
            # RAG模式且有上下文
            final_user_query = (
                f"--- 以下是参考信息 ---\n"
                f"{context}\n"
                f"--- 参考信息结束 ---\n\n"
                f"请严格基于以上参考信息，并结合你自身的知识，来回答问题。\n"
                f"附加要求: {additional_requirements}\n\n"
                f"问题: {user_query}"
            )
            # 对于RAG查询，我们覆盖默认的“芝麻编程老师”身份，让模型更专注于上下文。
            # 传递一个空字符串""来清空system_prompt。
            system_prompt_override = "" 
            return final_user_query, system_prompt_override
        else:
            # 纯LLM模式或无上下文
            final_user_query = (
                 f"附加要求: {additional_requirements}\n\n"
                 f"问题: {user_query}"
            )
            # 使用LLMClient中默认的“芝麻编程老师”系统提示词。
            # 传递None表示不覆盖。
            system_prompt_override = None 
            return final_user_query, system_prompt_override
            
    async def get_response(self, user_query: str):
        """
        根据当前RAG模式，获取LLM的响应。
        """
        context = ""
        # 1. 如果RAG模式开启且已选择知识库，则进行检索
        if self.use_rag and self.current_kb:
            retrieved_docs = await self.kb.query(self.current_kb, user_query)
            if retrieved_docs:
                context = "\n\n---\n\n".join([doc['content'] for doc in retrieved_docs])

        # 2. 构建最终的提示词和可选的系统提示词覆盖
        final_query, system_prompt_override = self._build_prompt(user_query, context)

        # 3. 使用共享的LLM客户端进行流式调用
        async for chunk in self.llm_client.ask(final_query, system_prompt_override=system_prompt_override):
            yield chunk
