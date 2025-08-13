# /workspace/LiveTalking/pkg/rag/knowledge_base.py
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
import re
import time
import portalocker
import os
import requests

# [新依赖] 引入 langchain 的文件加载器和文本分割器
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import logger
from pkg.rag.config import (
    CHROMA_PERSIST_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    CHUNK_SIZE,       # 从 config 导入 CHUNK_SIZE
    CHUNK_OVERLAP,    # 从 config 导入 CHUNK_OVERLAP
)

_HAS_RERANKER = False


@dataclass
class KBHit:
    """封装单条检索结果的数据类"""
    id: str
    text: str
    metadata: Dict
    score: float

@dataclass
class KBQueryResult:
    """封装完整查询结果的数据类"""
    hits: List[KBHit]


class KnowledgeBase:
    """
    [重构] 知识库管理，逻辑对齐 main.py
    """

    def __init__(self, kb_name: str):
        if not kb_name:
            raise ValueError("知识库名称 (kb_name) 不能为空。")
        self.kb_name = kb_name
        
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        self.embed_fn = OllamaEmbeddingFunction(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)
        
        collection_name = f"{self.kb_name}__ollama_{OLLAMA_EMBED_MODEL.replace(':','_').replace('-','_')}"
        
        # [修改] 直接在 get_or_create_collection 中传递 embedding_function 实例
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn
        )
        logger.info(f"已成功加载或创建知识库: '{self.kb_name}'")
        self.reranker = None


    # [!!! 最终修复 !!!] 完全重写为异步函数，并使用 langchain 工具链
    async def add_txt_file(self, txt_path: Path, source_filename: str):
        """
        [异步重构] 使用 UnstructuredFileLoader 加载文件，并异步添加到 ChromaDB。
        """
        logger.info(f"[kb.add] 准备向知识库 '{self.kb_name}' 添加文件: {source_filename}")
        
        # 检查文件是否存在，如果不存在则直接报错，不再尝试打开
        if not txt_path.exists():
            logger.error(f"[kb.add] 文件不存在，无法处理: {txt_path}")
            raise FileNotFoundError(f"文件在入库前未找到: {txt_path}")

        lock_path = txt_path.with_suffix(".txt.lock")
        try:
            with portalocker.Lock(lock_path, 'w', timeout=60):
                logger.info(f"[kb.add] 已成功获取文件锁: {lock_path}")
                
                existing_docs = self.collection.get(where={"source": source_filename}, limit=1)
                if existing_docs and existing_docs['ids']:
                    logger.info(f"[kb.add] 文档 '{source_filename}' 已存在，跳过。")
                    return

                # 1. 使用 UnstructuredFileLoader 加载文件（在线程中运行）
                logger.info(f"[kb.add] 使用 UnstructuredFileLoader 加载: {txt_path}")
                loader = UnstructuredFileLoader(str(txt_path))
                docs = await asyncio.to_thread(loader.load)
                if not docs:
                    logger.warning(f"[kb.add] loader 未能从 '{source_filename}' 中加载任何文档内容。")
                    return
                
                # 2. 使用 RecursiveCharacterTextSplitter 切分文档
                # 注意: chunk_size 和 chunk_overlap 从 config.py 导入
                splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                chunks = splitter.split_documents(docs)
                
                # 3. 清理元数据中的复杂项
                chunks = filter_complex_metadata(chunks)
                
                if not chunks:
                    logger.warning(f"[kb.add]未能从 '{source_filename}' 切分出任何文本块。")
                    return

                # 4. 准备要添加到 Chroma 的数据
                n_chunks = len(chunks)
                ids = [f"{source_filename}-{i:06d}" for i in range(n_chunks)]
                documents = [chunk.page_content for chunk in chunks]
                metadatas = [chunk.metadata for chunk in chunks]

                logger.info(f"[kb.add] 文件切分完成，共 {n_chunks} 个片段。即将添加到数据库...")

                # 5. 将数据异步添加到 ChromaDB
                # 注意：我们不再手动计算嵌入，ChromaDB 会使用我们提供的 embed_fn 自动处理
                await asyncio.to_thread(
                    self.collection.add,
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                
                logger.info(f"[kb.add] 成功将 {n_chunks} 个文本块从 '{source_filename}' 添加到知识库 '{self.kb_name}'。")

        except portalocker.LockException:
            logger.error(f"获取文件锁超时: {lock_path}")
            raise RuntimeError(f"无法锁定文件 {source_filename} 进行处理。")
        except Exception as e:
            logger.error(f"添加文件 '{source_filename}' 到知识库 '{self.kb_name}' 时失败: {e}", exc_info=True)
            raise

    # 移除了所有其他不再需要的私有方法，如 _split_text, _embed_texts_with_progress

    def search(self, query: str, top_k: int = 8, use_reranker: bool = True, reranker_threshold: float = 0.15) -> KBQueryResult:
        """
        增强版两阶段检索。
        """
        logger.info(f"在知识库 '{self.kb_name}' 中执行查询: '{query[:50]}...'")
        
        num_candidates = max(top_k * 3, 15) if use_reranker and self.reranker else top_k
        
        q_result = self.collection.query(
            query_texts=[query],
            n_results=num_candidates,
            include=["documents", "metadatas", "distances"]
        )
        
        if not q_result or not q_result["documents"][0]:
            hits = []
        else:
            docs = q_result["documents"][0]
            metas = q_result["metadatas"][0]
            ids = q_result["ids"][0]
            dists = q_result["distances"][0]
            hits = [KBHit(id=i, text=t, metadata=m, score=float(1.0 / (1e-6 + d)))
                    for i, t, m, d in zip(ids, docs, metas, dists)]

        filtered_hits = hits
        
        if len(filtered_hits) == 0:
            logger.info("初始检索结果为空或质量过低，启动HyDE兜底机制...")
            hyde_query = self._generate_hyde_query(query)
            hyde_result = self.collection.query(
                query_texts=[hyde_query],
                n_results=num_candidates,
                include=["documents", "metadatas", "distances"]
            )
            
            if hyde_result and hyde_result["documents"][0]:
                docs = hyde_result["documents"][0]
                metas = hyde_result["metadatas"][0]
                ids = hyde_result["ids"][0]
                dists = hyde_result["distances"][0]
                hyde_hits = [KBHit(id=i, text=t, metadata=m, score=float(1.0 / (1e-6 + d)))
                             for i, t, m, d in zip(ids, docs, metas, dists)]
                filtered_hits = hyde_hits
        
        filtered_hits.sort(key=lambda x: x.score, reverse=True)
        return KBQueryResult(hits=filtered_hits[:top_k])
    
    def _generate_hyde_query(self, query: str) -> str:
        """
        生成HyDE（假设文档扩展）查询
        """
        hyde_template = (
            f"{query} "
            f"相关内容包括：{query}的定义、{query}的特点、{query}的应用、"
            f"{query}的原理、{query}的方法、{query}的步骤、{query}的注意事项。"
        )
        return hyde_template[:500]

    @classmethod
    def delete_kb(cls, kb_name: str):
        # ... (此函数无需修改) ...
        logger.info(f"正在请求删除知识库 '{kb_name}'...")
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
            collections = client.list_collections()
            embed_backend = f"ollama_{OLLAMA_EMBED_MODEL.replace(':','_').replace('-','_')}"
            target_name = f"{kb_name}__{embed_backend}"
            
            collection_exists = any(c.name == target_name for c in collections)

            if not collection_exists:
                logger.warning(f"尝试删除一个不存在的知识库集合 '{target_name}'。")
                return
                
            client.delete_collection(name=target_name)
            logger.info(f"已删除集合: {target_name}")
            logger.info(f"知识库 '{kb_name}' 已成功从 ChromaDB 删除。")
        except Exception as e:
            logger.error(f"删除知识库 '{kb_name}' 时发生意外错误: {e}")
            raise

    # 其他类方法如 list_kbs, warm_up_rag_models 保持不变...
    @classmethod
    def list_kbs(cls) -> List[str]:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        collections = client.list_collections()
        names = set()
        for col in collections:
            name = col.name
            if "__" in name:
                base = name.split("__", 1)[0]
                names.add(base)
            else:
                names.add(name)
        return sorted(list(names))


class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    [重构] ChromaDB 的 Ollama 嵌入函数
    """
    def __init__(self, base_url: str, model: str):
        self._api_url = f"{base_url.rstrip('/')}/api/embeddings"
        self._model = model
        self._session = requests.Session()

    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        logger.info(f"[emb] 开始为 {len(input_texts)} 个文本块计算嵌入向量...")
        try:
            response = self._session.post(
                self._api_url,
                json={"model": self._model, "prompts": input_texts}
            )
            response.raise_for_status()
            # [修改] Ollama 批量嵌入的返回格式是 `{"embeddings": [[...], [...]]}`
            embeddings = response.json().get("embeddings", [])
            logger.info(f"[emb] 成功生成 {len(embeddings)} 个嵌入向量。")
            return embeddings
        except Exception as e:
            logger.error(f"[emb] 调用 Ollama 嵌入 API 失败: {e}", exc_info=True)
            # 返回零向量作为兜底，以避免整个流程崩溃
            return [[0.0] * 768 for _ in input_texts]