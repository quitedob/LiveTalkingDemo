# /workspace/LiveTalking/pkg/avatars/generator_musetalk.py
# 数字人生成器 - 基于genavatar_musetalk.py的生成流程

import asyncio
import json
import os
import subprocess
import sys
import uuid
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# 中文注释：引入会话管理器，用于检查数字人是否被占用
from pkg.avatars.session_manager import get_session_manager


# 配置日志
logger = logging.getLogger(__name__)

class AvatarGenerator:
    """数字人生成器类"""
    
    def __init__(self, base_path: str = None):
        """
        【修改后】
        初始化数字人生成器 (使用更稳健的路径推断)
        
        Args:
            base_path: 可选的项目根目录绝对路径。若不提供，则自动从当前文件位置向上推断。
        """
        # 中文注释：优先使用传入的绝对路径，否则根据当前文件位置动态推断项目根目录
        if base_path and Path(base_path).is_absolute():
            self.base_path = Path(base_path)
        else:
            # 中文注释：以本文件 (.../pkg/avatars/generator_musetalk.py) 为基准向上回溯两级，找到项目根目录
            self.base_path = Path(__file__).resolve().parents[2]

        self.genavatar_script = self.base_path / "genavatar_musetalk.py"
        self.data_dir = self.base_path / "data" / "avatars"
        
        # 中文注释：确保数据目录存在，如果不存在则创建
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 中文注释：用于在内存中存储异步任务的状态
        self.tasks: Dict[str, Dict] = {}
    
    def _validate_avatar_id(self, avatar_id: str) -> Tuple[bool, str]:
        """
        验证avatar_id是否符合命名规范
        
        Args:
            avatar_id: 数字人ID
            
        Returns:
            (是否有效, 错误信息)
        """
        # 中文注释：命名规则：先是1-16位的小写字母，然后是1-4位数字，总长度不超过20
        pattern = r'^[a-z]{1,16}\d{1,4}$'
        
        if not re.match(pattern, avatar_id):
            return False, f"avatar_id不符合命名规范。规则：^[a-z]{{1,16}}\\d{{1,4}}$，示例：xiaoli0001、mike1"
        
        if len(avatar_id) > 20:
            return False, "avatar_id长度不能超过20个字符"
        
        # 中文注释：检查对应ID的目录是否已存在
        avatar_dir = self.data_dir / avatar_id
        if avatar_dir.exists():
            return False, f"avatar_id '{avatar_id}' 已存在"
        
        return True, ""
    
    def _generate_avatar_id(self) -> str:
        """
        生成符合规范且唯一的avatar_id
        
        Returns:
            生成的avatar_id
        """
        import random
        
        # 中文注释：预设一些常见拼音前缀用于随机生成
        prefixes = ['xiaoli', 'mike', 'zhang', 'wang', 'li', 'chen', 'liu', 'yang', 'huang', 'zhao']
        
        while True:
            # 中文注释：随机选择前缀并拼接1-4位随机数字
            prefix = random.choice(prefixes)
            suffix = str(random.randint(1, 9999)).zfill(4) # 修正：固定为4位，保证唯一性概率更高
            avatar_id = f"{prefix[:16]}{suffix}" # 修正：确保前缀不超过16位
            
            # 中文注释：检查生成的ID是否已存在，直到找到一个可用的
            avatar_dir = self.data_dir / avatar_id
            if not avatar_dir.exists():
                return avatar_id
    
    async def create_avatar(self, file_path: str, avatar_id: Optional[str] = None) -> Dict:
        """
        创建数字人（异步任务）
        
        Args:
            file_path: 输入文件路径（图片/视频/ZIP）
            avatar_id: 数字人ID（可选，不填则自动生成）
            
        Returns:
            任务信息字典
        """
        # 中文注释：为创建任务生成一个唯一的job_id
        job_id = str(uuid.uuid4())
        
        # 中文注释：如果未提供avatar_id，则自动生成；否则验证其有效性
        if avatar_id is None:
            avatar_id = self._generate_avatar_id()
        else:
            is_valid, error_msg = self._validate_avatar_id(avatar_id)
            if not is_valid:
                raise ValueError(error_msg)
        
        # 中文注释：创建并存储任务的初始信息
        task_info = {
            "job_id": job_id,
            "avatar_id": avatar_id,
            "file_path": file_path,
            "status": "PENDING",
            "created_at": asyncio.get_event_loop().time(),
            "error_code": None,
            "stderr_tail": None
        }
        
        self.tasks[job_id] = task_info
        
        # 中文注释：在事件循环中创建一个后台任务来执行耗时的生成过程
        asyncio.create_task(self._generate_avatar_task(job_id))
        
        return {
            "avatar_id": avatar_id,
            "job_id": job_id,
            "status": "PENDING"
        }
    
    async def _generate_avatar_task(self, job_id: str):
        """
        执行数字人生成任务
        
        Args:
            job_id: 任务ID
        """
        task = self.tasks[job_id]
        
        try:
            logger.info(f"开始生成数字人: {task['avatar_id']}")
            
            # 中文注释：构建用于调用外部生成脚本的命令行参数
            cmd = [
                sys.executable,  # Python解释器
                str(self.genavatar_script),
                "--file", task['file_path'],
                "--avatar_id", task['avatar_id'],
                "--version", "v15",  # 使用v15版本
                "--gpu_id", "0",     # 使用GPU 0
                "--bbox_shift", "0", # 边界框偏移
                "--extra_margin", "10", # 额外边距
                "--parsing_mode", "jaw" # 解析模式
            ]
            
            # 中文注释：异步执行子进程，并捕获其标准输出和错误输出
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.base_path)
            )
            
            stdout, stderr = await process.communicate()
            
            # 中文注释：根据子进程的返回码判断成功或失败，并更新任务状态
            if process.returncode == 0:
                task['status'] = "SUCCEEDED"
                logger.info(f"数字人生成成功: {task['avatar_id']}")
                self._update_avatar_info(task['avatar_id'], task['file_path'])
            else:
                task['status'] = "FAILED"
                task['error_code'] = process.returncode
                task['stderr_tail'] = stderr.decode('utf-8', errors='ignore')[-500:]  # 保留最后500字符的错误信息
                logger.error(f"数字人生成失败: {task['avatar_id']}, 错误码: {process.returncode}")
            
        except Exception as e:
            # 中文注释：捕获执行过程中的任何异常，并标记任务为失败
            task['status'] = "FAILED"
            task['error_code'] = -1
            task['stderr_tail'] = str(e)
            logger.error(f"数字人生成异常: {task['avatar_id']}, 异常: {e}")
    
    def _update_avatar_info(self, avatar_id: str, file_path: str):
        """
        更新数字人信息文件
        
        Args:
            avatar_id: 数字人ID
            file_path: 原始文件路径
        """
        try:
            import time
            avatar_dir = self.data_dir / avatar_id
            info_file = avatar_dir / "avator_info.json"
            
            info = {
                "avatar_id": avatar_id,
                "video_path": file_path,
                "created_at": time.time(),
                "bbox_shift": 0
            }
            
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"更新数字人信息文件失败: {e}")
    
    def get_avatar_info(self, avatar_id: str) -> Optional[Dict]:
        """
        获取数字人信息
        
        Args:
            avatar_id: 数字人ID
            
        Returns:
            数字人信息字典
        """
        avatar_dir = self.data_dir / avatar_id
        
        if not avatar_dir.exists():
            return None
        
        # 中文注释：读取包含元数据的avator_info.json文件
        info_file = avatar_dir / "avator_info.json"
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
            except Exception as e:
                logger.error(f"读取数字人信息文件失败: {e}")
                info = {}
        else:
            info = {}
        
        # 中文注释：检查生成数字人所需的关键文件是否存在
        full_imgs_dir = avatar_dir / "full_imgs"
        mask_dir = avatar_dir / "mask"
        latents_file = avatar_dir / "latents.pt"
        coords_file = avatar_dir / "coords.pkl"
        mask_coords_file = avatar_dir / "mask_coords.pkl"
        
        # 中文注释：统计图片帧数
        frames_count = 0
        if full_imgs_dir.exists():
            frames_count = len(list(full_imgs_dir.glob("*.png")))
        
        # 中文注释：构建并返回包含所有相关信息的字典
        avatar_info = {
            "avatar_id": avatar_id,
            "created_at": info.get("created_at", os.path.getctime(avatar_dir)),
            "frames": frames_count,
            "preview": f"/api/avatars/{avatar_id}/preview" if frames_count > 0 else None,
            "status": "SUCCEEDED" if all([
                full_imgs_dir.exists(),
                mask_dir.exists(),
                latents_file.exists(),
                coords_file.exists(),
                mask_coords_file.exists()
            ]) else "FAILED",
            "video_path": info.get("video_path", ""),
            "bbox_shift": info.get("bbox_shift", 0)
        }
        
        return avatar_info
    
    def list_avatars(self, page: int = 1, page_size: int = 20) -> Dict:
        """
        获取数字人列表
        
        Args:
            page: 页码（从1开始）
            page_size: 每页大小
            
        Returns:
            分页结果
        """
        if not self.data_dir.exists():
            return {
                "avatars": [],
                "total": 0,
                "page": page,
                "page_size": page_size
            }
        
        # 中文注释：获取所有数字人的目录列表
        avatar_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        # 中文注释：按创建时间降序排序
        avatar_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
        
        # 中文注释：根据页码和页面大小进行分页
        total = len(avatar_dirs)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_dirs = avatar_dirs[start_idx:end_idx]
        
        # 中文注释：获取分页后每个数字人的详细信息
        avatars = []
        for avatar_dir in page_dirs:
            avatar_id = avatar_dir.name
            avatar_info = self.get_avatar_info(avatar_id)
            if avatar_info:
                avatars.append(avatar_info)
        
        return {
            "avatars": avatars,
            "total": total,
            "page": page,
            "page_size": page_size
        }
    
    def delete_avatar(self, avatar_id: str) -> bool:
        """
        删除数字人
        
        Args:
            avatar_id: 数字人ID
            
        Returns:
            是否删除成功
        """
        avatar_dir = self.data_dir / avatar_id
        
        if not avatar_dir.exists():
            return False
        
        try:
            # 中文注释：使用shutil.rmtree递归删除整个目录
            import shutil
            shutil.rmtree(avatar_dir)
            logger.info(f"数字人删除成功: {avatar_id}")
            return True
        except Exception as e:
            logger.error(f"删除数字人失败: {avatar_id}, 错误: {e}")
            return False
    
    def get_task_status(self, job_id: str) -> Optional[Dict]:
        """
        获取任务状态
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务状态信息
        """
        return self.tasks.get(job_id)
    
    def is_avatar_in_use(self, avatar_id: str) -> Tuple[bool, Optional[str]]:
        """
        中文注释：【已修正】通过 SessionManager 判断数字人是否正被会话占用
        
        Args:
            avatar_id: 数字人ID
            
        Returns:
            (是否被占用, 占用的session_id)
        """
        # 中文注释：此功能需要与会话管理器(SessionManager)集成
        sm = get_session_manager()
        in_use, sid = sm.is_avatar_in_use(avatar_id)
        return in_use, sid

# 中文注释：全局生成器单例，避免重复实例化
_generator = None

def get_generator() -> AvatarGenerator:
    """获取全局生成器实例"""
    global _generator
    if _generator is None:
        _generator = AvatarGenerator()
    return _generator