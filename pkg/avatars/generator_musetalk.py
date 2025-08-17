# /workspace/LiveTalking/pkg/avatars/generator_musetalk.py
# 数字人生成器 - 已整合核心功能，并实现模型动态加载与闲置自动释放

import asyncio
import aiohttp
import json
import os
import sys
import uuid
import re
import glob
import pickle
import shutil
import time
import logging
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 中文注释：引入会话管理器，用于检查数字人是否被占用
from pkg.avatars.session_manager import get_session_manager

# --- 新增：从 musetalk 和 utils 导入必要的模块 ---
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.blending import get_image_prepare_material
from musetalk.utils.utils import load_all_model

try:
    from musetalk.utils.face_parsing import FaceParsing
except ModuleNotFoundError:
    # 兼容不同的项目结构或潜在的路径问题
    from utils.face_parsing import FaceParsing

# 配置日志
logger = logging.getLogger(__name__)

# --- 新增：配置项 ---
# 中文注释：模型在闲置超过这个秒数后，将被自动从内存和显存中卸载
MODELS_IDLE_TIMEOUT_SECONDS = 600  # 10分钟

# --- 新增：从 genavatar_musetalk_demo.py 迁移过来的辅助函数 ---
def is_video_file(file_path: str) -> bool:
    """
    简单根据文件后缀判断是否为视频文件。
    """
    video_exts = ['.mp4', '.mkv', '.flv', '.avi', '.mov']
    return os.path.splitext(file_path)[1].lower() in video_exts

def video2imgs(vid_path: str, save_path: str, ext: str = '.png'):
    """
    将视频文件逐帧转换为图片。
    """
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{save_path}/{count:08d}{ext}", frame)
        count += 1
    cap.release()

def create_dir(dir_path: Path):
    """
    如果目录不存在，则创建它。
    """
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)


class AvatarGenerator:
    """
    数字人生成器类
    【新增特性】: 实现了模型的动态加载与闲置自动释放机制。
    """
    
    def __init__(self):
        """
        初始化数字人生成器。
        - 此时不加载AI模型，仅设置基本路径和状态。
        - 启动一个后台任务，用于监控模型闲置并自动释放资源。
        """
        self.base_path = Path(__file__).resolve().parents[2]
        logger.info(f"AvatarGenerator 自动推断出项目根目录: {self.base_path}")

        self.data_dir = self.base_path / "data" / "avatars"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.tasks: Dict[str, Dict] = {}

        # --- 核心改造：模型初始化为 None ---
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.fp = None
        
        # --- 新增：用于管理模型加载/卸载的状态和锁 ---
        self._models_lock = asyncio.Lock() # 防止加载和卸载同时进行
        self.last_used_time = 0
        self._monitor_task = asyncio.create_task(self._monitor_idle_and_release())

    async def _load_models_if_needed(self):
        """
        【新增】按需加载模型的核心方法（惰性加载）。
        如果模型未加载，则获取锁并加载模型到内存/显存。
        """
        if self.vae is not None:
            # 模型已加载，直接返回
            return

        async with self._models_lock:
            # 获取锁后再次检查，防止在等待锁的过程中模型已被其他协程加载
            if self.vae is not None:
                return
            
            logger.info("检测到模型未加载，开始加载数字人生成AI模型...")
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            # 加载VAE, UNet, PositionalEncoding
            self.vae, self.unet, self.pe = load_all_model(device=self.device)
            self.vae.vae = self.vae.vae.half().to(self.device)
            
            # 初始化面部解析器 (FaceParsing)
            self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
            
            logger.info(f"数字人生成模型加载完成，使用设备: {self.device}")

    async def _release_models(self):
        """
        【新增】释放模型资源的方法。
        将模型对象设为None，并清理CUDA缓存。
        """
        async with self._models_lock:
            if self.vae is None:
                # 模型已释放，无需操作
                return

            logger.info("检测到模型闲置超时，正在释放资源...")
            self.vae = None
            self.unet = None
            self.pe = None
            self.fp = None
            
            # 清理垃圾并释放显存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("模型资源已成功释放。")

    async def _monitor_idle_and_release(self):
        """
        【新增】后台监控任务，定期检查模型是否闲置并触发释放。
        """
        while True:
            await asyncio.sleep(60) # 每分钟检查一次
            if self.vae is not None:
                idle_time = time.time() - self.last_used_time
                if idle_time > MODELS_IDLE_TIMEOUT_SECONDS:
                    await self._release_models()

    async def _run_avatar_generation(self, job_id: str):
        """
        【核心重构】
        执行数字人生成任务。先确保模型已加载，任务结束后更新最后使用时间。
        """
        task = self.tasks[job_id]
        avatar_id = task['avatar_id']
        file_path = task['file_path']
        
        # --- 步骤 1: 确保模型已加载 ---
        await self._load_models_if_needed()
        self.last_used_time = time.time() # 标记为正在使用
        
        avatar_dir = self.data_dir / avatar_id
        try:
            avatar_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            task.update({"status": "FAILED", "error_code": -2, "stderr_tail": f"预创建目录失败: {e}"})
            logger.error(f"为数字人 '{avatar_id}' 预创建目录失败: {e}")
            return

        try:
            await self._quiesce_runtime()
            task['status'] = "RUNNING"

            # --- 步骤 2: 执行实际的生成逻辑 ---
            save_path = avatar_dir
            save_full_path = save_path / 'full_imgs'
            mask_out_path = save_path / 'mask'
            create_dir(save_full_path)
            create_dir(mask_out_path)

            bbox_shift, version, extra_margin, parsing_mode = 0, "v15", 10, "jaw"

            # --- 【关键修复】 START ---
            # 中文注释：修复单张图片上传时的文件名问题，确保其被重命名为序列帧格式。
            if os.path.isfile(file_path):
                if is_video_file(file_path):
                    video2imgs(file_path, str(save_full_path), ext='.png')
                else:
                    # 如果是单张图片，获取其扩展名
                    ext = os.path.splitext(file_path)[1]
                    if not ext:
                        ext = '.png' # 如果没有扩展名，默认.png
                    # 复制文件并将其重命名为第一帧 (00000000.ext)
                    shutil.copyfile(file_path, save_full_path / f"00000000{ext}")
            # --- 【关键修复】 END ---
            else: # 如果是目录
                files = sorted([f for f in os.listdir(file_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                for filename in files:
                    shutil.copyfile(os.path.join(file_path, filename), save_full_path / filename)
            
            input_img_list = sorted(glob.glob(os.path.join(str(save_full_path), '*.[jpJP][pnPN]*[gG]')))
            if not input_img_list:
                raise FileNotFoundError("未能找到任何输入图片用于生成数字人。")

            logger.info(f"[{avatar_id}] 正在提取面部关键点...")
            try:
                coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
            except ZeroDivisionError:
                raise ValueError("在所有提供的图片帧中都未能检测到人脸，请检查输入文件是否包含清晰、正面的人脸。")
            
            input_latent_list = []
            coord_placeholder = (0.0, 0.0, 0.0, 0.0)
            
            logger.info(f"[{avatar_id}] 正在生成潜变量...")
            for idx, (bbox, frame) in enumerate(tqdm(zip(coord_list, frame_list), total=len(coord_list))):
                if bbox == coord_placeholder:
                    input_latent_list.append(None)
                    continue
                x1, y1, x2, y2 = bbox
                if version == "v15":
                    y2 = y2 + extra_margin
                    y2 = min(y2, frame.shape[0])
                    coord_list[idx] = [x1, y1, x2, y2]
                
                crop_frame = frame[y1:y2, x1:x2]
                resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                latents = self.vae.get_latents_for_unet(resized_crop_frame)
                input_latent_list.append(latents)

            mask_coords_list = []
            
            logger.info(f"[{avatar_id}] 正在生成遮罩并保存中间文件...")
            for i, frame in enumerate(tqdm(frame_list)):
                cv2.imwrite(f"{save_full_path}/{str(i).zfill(8)}.png", frame)
                
                if coord_list[i] == coord_placeholder:
                    mask_coords_list.append(None)
                    continue

                x1, y1, x2, y2 = coord_list[i]
                mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=self.fp, mode=parsing_mode)
                cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png", mask)
                mask_coords_list.append(crop_box)

            with open(save_path / 'mask_coords.pkl', 'wb') as f:
                pickle.dump(mask_coords_list, f)
            with open(save_path / 'coords.pkl', 'wb') as f:
                pickle.dump(coord_list, f)
            
            valid_latents = [l for l in input_latent_list if l is not None]
            if not valid_latents:
                 raise ValueError("未能从输入图片中生成任何有效的潜变量。")
            torch.save(valid_latents, save_path / 'latents.pt')
            
            task.update({"status": "SUCCEEDED", "error_code": 0, "stderr_tail": ""})
            logger.info(f"数字人生成成功: {avatar_id}")
            self._update_avatar_info(avatar_id, file_path)

        except Exception as e:
            task.update({"status": "FAILED", "error_code": -1, "stderr_tail": str(e)})
            logger.exception(f"执行数字人生成任务时发生意外错误: job_id={job_id}, avatar_id={avatar_id}")
            try:
                self.delete_avatar(avatar_id)
            except Exception as cleanup_e:
                logger.error(f"生成异常后清理目录 '{avatar_id}' 时出错: {cleanup_e}")
        finally:
            self.last_used_time = time.time()
            
    async def create_avatar(self, file_path: str, avatar_id: Optional[str] = None) -> Dict:
        job_id = str(uuid.uuid4())
        
        if avatar_id is None:
            avatar_id = self._generate_avatar_id()
        else:
            is_valid, error_msg = self._validate_avatar_id(avatar_id)
            if not is_valid:
                raise ValueError(error_msg)
        
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
        
        asyncio.create_task(self._run_avatar_generation(job_id))
        
        return {"avatar_id": avatar_id, "job_id": job_id, "status": "PENDING"}

    def _update_avatar_info(self, avatar_id: str, file_path: str):
        try:
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

    async def _quiesce_runtime(self, timeout: float = 8.0):
        try:
            sm = get_session_manager()
            active = getattr(sm, "get_active_sessions", lambda: {})()
            session_ids = list(active.keys()) if isinstance(active, dict) else []
            base_url = os.environ.get("LIVETALKING_BASE_URL", "").rstrip("/")
            if base_url and session_ids:
                async with aiohttp.ClientSession() as sess:
                    for sid in session_ids:
                        try:
                            await sess.post(f"{base_url}/interrupt_talk", json={"sessionid": sid}, timeout=timeout)
                        except Exception:
                            pass
            if session_ids:
                for sid in session_ids:
                    try:
                        sm.close_session(str(sid))
                    except Exception:
                        pass
            await asyncio.sleep(0.5)
        except Exception:
            return

    def _validate_avatar_id(self, avatar_id: str) -> Tuple[bool, str]:
        pattern = r'^[a-z]{1,16}\d{1,4}$'
        if not re.match(pattern, avatar_id):
            return False, f"avatar_id不符合命名规范。规则：^[a-z]{{1,16}}\\d{{1,4}}$，示例：xiaoli0001、mike1"
        if len(avatar_id) > 20:
            return False, "avatar_id长度不能超过20个字符"
        avatar_dir = self.data_dir / avatar_id
        if avatar_dir.exists():
            return False, f"avatar_id '{avatar_id}' 已存在"
        return True, ""
    
    def _generate_avatar_id(self) -> str:
        import random
        prefixes = ['xiaoli', 'mike', 'zhang', 'wang', 'li', 'chen', 'liu', 'yang', 'huang', 'zhao']
        while True:
            prefix = random.choice(prefixes)
            suffix = str(random.randint(1, 9999)).zfill(4)
            avatar_id = f"{prefix[:16]}{suffix}"
            avatar_dir = self.data_dir / avatar_id
            if not avatar_dir.exists():
                return avatar_id

    def get_avatar_info(self, avatar_id: str) -> Optional[Dict]:
        avatar_dir = self.data_dir / avatar_id
        if not avatar_dir.exists():
            return None
        
        info_file = avatar_dir / "avator_info.json"
        info = {}
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
            except Exception as e:
                logger.error(f"读取数字人信息文件失败: {e}")
        
        full_imgs_dir = avatar_dir / "full_imgs"
        mask_dir = avatar_dir / "mask"
        latents_file = avatar_dir / "latents.pt"
        coords_file = avatar_dir / "coords.pkl"
        mask_coords_file = avatar_dir / "mask_coords.pkl"
        
        frames_count = len(list(full_imgs_dir.glob("*.[jp][pn]g"))) if full_imgs_dir.is_dir() else 0
        
        avatar_info = {
            "avatar_id": avatar_id,
            "created_at": info.get("created_at", os.path.getctime(avatar_dir)),
            "frames": frames_count,
            "preview": f"/api/avatars/{avatar_id}/preview" if frames_count > 0 else None,
            "status": "SUCCEEDED" if all([
                full_imgs_dir.exists(), mask_dir.exists(), latents_file.exists(),
                coords_file.exists(), mask_coords_file.exists()
            ]) else "FAILED",
            "video_path": info.get("video_path", ""),
            "bbox_shift": info.get("bbox_shift", 0)
        }
        return avatar_info
    
    def list_avatars(self, page: int = 1, page_size: int = 20) -> Dict:
        if not self.data_dir.exists():
            return {"avatars": [], "total": 0, "page": page, "page_size": page_size}
        
        avatar_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir()],
            key=lambda x: os.path.getctime(x),
            reverse=True
        )
        
        total = len(avatar_dirs)
        start_idx = (page - 1) * page_size
        page_dirs = avatar_dirs[start_idx : start_idx + page_size]
        
        avatars = [self.get_avatar_info(d.name) for d in page_dirs]
        avatars = [info for info in avatars if info]
        
        return {"avatars": avatars, "total": total, "page": page, "page_size": page_size}

    def delete_avatar(self, avatar_id: str) -> bool:
        avatar_dir = self.data_dir / avatar_id
        if not avatar_dir.exists():
            return False
        try:
            shutil.rmtree(avatar_dir)
            logger.info(f"数字人删除成功: {avatar_id}")
            return True
        except Exception as e:
            logger.error(f"删除数字人失败: {avatar_id}, 错误: {e}")
            return False

    def get_task_status(self, job_id: str) -> Optional[Dict]:
        return self.tasks.get(job_id)

    def is_avatar_in_use(self, avatar_id: str) -> Tuple[bool, Optional[str]]:
        sm = get_session_manager()
        in_use, sid = sm.is_avatar_in_use(avatar_id)
        return in_use, sid

# 全局生成器单例
_generator = None

def get_generator() -> AvatarGenerator:
    """获取全局生成器实例。"""
    global _generator
    if _generator is None:
        _generator = AvatarGenerator()
    return _generator