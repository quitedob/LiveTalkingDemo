# /workspace/LiveTalking/pkg/avatars/session_manager.py
# 会话管理器 - 管理数字人的热切换功能

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class SessionManager:
    """会话管理器类"""
    
    def __init__(self):
        """初始化会话管理器"""
        self.sessions: Dict[str, Dict] = {}
        self.avatar_sessions: Dict[str, Set[str]] = {}  # avatar_id -> session_ids
    
    def create_session(self, session_id: str, initial_avatar_id: Optional[str] = None) -> Dict:
        """
        创建新会话
        
        Args:
            session_id: 会话ID
            initial_avatar_id: 初始数字人ID
            
        Returns:
            会话信息
        """
        session_info = {
            "session_id": session_id,
            "avatar_id": initial_avatar_id,
            "created_at": time.time(),
            "last_activity": time.time(),
            "status": "active"
        }
        
        self.sessions[session_id] = session_info
        
        # 更新数字人使用情况
        if initial_avatar_id:
            if initial_avatar_id not in self.avatar_sessions:
                self.avatar_sessions[initial_avatar_id] = set()
            self.avatar_sessions[initial_avatar_id].add(session_id)
        
        logger.info(f"创建会话: {session_id}, 数字人: {initial_avatar_id}")
        return session_info
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话信息
        """
        session = self.sessions.get(session_id)
        if session:
            # 更新最后活动时间
            session["last_activity"] = time.time()
        return session
    
    def switch_avatar(self, session_id: str, new_avatar_id: str) -> Dict:
        """
        切换会话的数字人（热切换）
        
        Args:
            session_id: 会话ID
            new_avatar_id: 新的数字人ID
            
        Returns:
            切换结果
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"会话不存在: {session_id}")
        
        previous_avatar_id = session.get("avatar_id")
        
        # 更新会话的数字人
        session["avatar_id"] = new_avatar_id
        session["last_activity"] = time.time()
        
        # 更新数字人使用情况
        if previous_avatar_id:
            if previous_avatar_id in self.avatar_sessions:
                self.avatar_sessions[previous_avatar_id].discard(session_id)
                if not self.avatar_sessions[previous_avatar_id]:
                    del self.avatar_sessions[previous_avatar_id]
        
        if new_avatar_id not in self.avatar_sessions:
            self.avatar_sessions[new_avatar_id] = set()
        self.avatar_sessions[new_avatar_id].add(session_id)
        
        logger.info(f"会话 {session_id} 切换数字人: {previous_avatar_id} -> {new_avatar_id}")
        
        return {
            "previous": previous_avatar_id,
            "current": new_avatar_id,
            "switched_at": time.time()
        }
    
    def is_avatar_in_use(self, avatar_id: str) -> Tuple[bool, Optional[str]]:
        """
        检查数字人是否被会话占用
        
        Args:
            avatar_id: 数字人ID
            
        Returns:
            (是否被占用, 占用的session_id)
        """
        if avatar_id not in self.avatar_sessions:
            return False, None
        
        session_ids = self.avatar_sessions[avatar_id]
        if not session_ids:
            return False, None
        
        # 返回第一个占用的会话ID
        return True, next(iter(session_ids))
    
    def close_session(self, session_id: str) -> bool:
        """
        关闭会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功关闭
        """
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # 更新数字人使用情况
        avatar_id = session.get("avatar_id")
        if avatar_id and avatar_id in self.avatar_sessions:
            self.avatar_sessions[avatar_id].discard(session_id)
            if not self.avatar_sessions[avatar_id]:
                del self.avatar_sessions[avatar_id]
        
        # 删除会话
        del self.sessions[session_id]
        
        logger.info(f"关闭会话: {session_id}")
        return True
    
    def cleanup_inactive_sessions(self, timeout: int = 3600) -> int:
        """
        清理不活跃的会话
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            清理的会话数量
        """
        current_time = time.time()
        inactive_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session["last_activity"] > timeout:
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            self.close_session(session_id)
        
        if inactive_sessions:
            logger.info(f"清理了 {len(inactive_sessions)} 个不活跃会话")
        
        return len(inactive_sessions)
    
    def get_active_sessions(self) -> Dict[str, Dict]:
        """
        获取所有活跃会话
        
        Returns:
            活跃会话字典
        """
        return self.sessions.copy()
    
    def get_avatar_usage(self) -> Dict[str, list[str]]:
        """
        获取数字人使用情况
        
        Returns:
            数字人使用情况字典
        """
        return {avatar_id: list(session_ids) for avatar_id, session_ids in self.avatar_sessions.items()}

# 全局会话管理器实例
_session_manager = None

def get_session_manager() -> SessionManager:
    """获取全局会话管理器实例"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
