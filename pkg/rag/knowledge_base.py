# /workspace/LiveTalking/pkg/rag/knowledge_base.py
import asyncio
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

from logger import logger
from pkg.rag.config import (CHROMA_PERSIST_DIR, CHUNK_OVERLAP, CHUNK_SIZE,
                            EMBEDDING_MODEL_NAME)


class KnowledgeBase:
    """
    知识库管理类，负责与ChromaDB交互，包括数据的增、删、查。
    """

    def __init__(self):
        """
        初始化ChromaDB客户端和嵌入函数。
        """
        # 1. 初始化ChromaDB客户端，并指定持久化路径
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        
        # 2. 初始化嵌入函数，使用指定的预训练模型
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        
        # 3. 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        logger.info("ChromaDB 客户端和嵌入函数初始化完成。")

    async def add_document(self, kb_name: str, txt_path: Path):
        """
        将文本文档处理并添加到指定的知识库（集合）中。
        :param kb_name: 知识库名称，对应ChromaDB中的集合名称。
        :param txt_path: 待处理的文本文档路径。
        """
        logger.info(f"正在向知识库 '{kb_name}' 添加文档 '{txt_path.name}'...")
        
        # 在线程池中执行同步的文件读取和文本分割操作
        loop = asyncio.get_event_loop()
        documents, metadatas, ids = await loop.run_in_executor(
            None, self._prepare_data_for_chroma, txt_path
        )

        if not documents:
            logger.warning(f"文档 '{txt_path.name}' 未能提取出任何文本块，跳过添加。")
            return

        # 4. 获取或创建集合
        collection = self.client.get_or_create_collection(
            name=kb_name,
            embedding_function=self.embedding_function
        )

        # 5. 将处理好的数据添加到集合中
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"成功将 {len(documents)} 个文本块从 '{txt_path.name}' 添加到知识库 '{kb_name}'。")
        except Exception as e:
            logger.error(f"向 ChromaDB 集合 '{kb_name}' 添加数据时出错: {e}")
            raise

    def _prepare_data_for_chroma(self, txt_path: Path):
        """
        读取文本文件，进行分割，并准备成ChromaDB所需的格式。
        这是一个同步函数，应在线程池中运行。
        """
        # 1. 读取文本内容
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 2. 使用文本分割器分割文本
        chunks = self.text_splitter.split_text(text)
        
        # 3. 准备ChromaDB所需的数据结构
        documents = chunks
        metadatas = [{"source": str(txt_path.name)} for _ in chunks]
        ids = [f"{txt_path.name}_{i}" for i in range(len(chunks))]
        
        return documents, metadatas, ids

    def delete_kb(self, kb_name: str):
        """
        删除指定的知识库（集合）。
        这是一个同步操作，因为ChromaDB的删除操作通常很快。
        """
        logger.info(f"正在删除知识库 '{kb_name}'...")
        try:
            self.client.delete_collection(name=kb_name)
            logger.info(f"知识库 '{kb_name}' 已成功删除。")
        except Exception as e:
            # 在ChromaDB中，如果集合不存在，删除会抛出异常
            logger.warning(f"尝试删除知识库 '{kb_name}' 时出错 (可能集合不存在): {e}")
            # 根据需求，这里可以不向上抛出异常，因为目标（集合不存在）已经达成
            pass

    async def query(self, kb_name: str, query_text: str, n_results: int = 3) -> list:
        """
        在指定的知识库中查询与问题最相关的文本块。
        :param kb_name: 知识库名称。
        :param query_text: 用户查询的问题。
        :param n_results: 返回的最相关结果数量。
        :return: 包含相关文本块和元数据的列表。
        """
        logger.info(f"正在知识库 '{kb_name}' 中查询: '{query_text[:50]}...'")
        try:
            collection = self.client.get_collection(
                name=kb_name,
                embedding_function=self.embedding_function
            )
            
            # 在线程池中执行查询
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: collection.query(
                    query_texts=[query_text],
                    n_results=n_results
                )
            )
            
            # 提取并格式化结果
            retrieved_docs = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    retrieved_docs.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i]
                    })
            
            logger.info(f"查询到 {len(retrieved_docs)} 个相关结果。")
            return retrieved_docs

        except Exception as e:
            # 如果集合不存在，get_collection会抛出异常
            logger.error(f"查询知识库 '{kb_name}' 时出错 (可能集合不存在): {e}")
            return []

    def list_kbs(self) -> list:
        """
        列出当前所有的知识库（集合）。
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]

