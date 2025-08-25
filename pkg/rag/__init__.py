# pkg/rag/__init__.py
"""
RAG (Retrieval-Augmented Generation) 模块
提供知识库管理、文件处理、RAG核心功能
"""

from .rag_core import RAGCore
from .knowledge_base import KnowledgeBase
from .file_processor import FileProcessor

__all__ = ['RAGCore', 'KnowledgeBase', 'FileProcessor']