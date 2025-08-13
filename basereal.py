# /mnt/data/LiveTalking-main/basereal.py
import math
import torch
import numpy as np

import subprocess
import os
import time
import cv2
import glob
import resampy

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf

import asyncio
from av import AudioFrame, VideoFrame

import av
from fractions import Fraction

from ttsreal import EdgeTTS,SovitsTTS,XTTS,CosyVoiceTTS,FishTTS,TencentTTS,DoubaoTTS
from logger import logger

from tqdm import tqdm
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def play_audio(quit_event,queue):      
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=16000,
        channels=1,
        format=pyaudio.paInt16,  # 格式通常是 paInt16
        output=True,
        # 根据您的系统调整 output_device_index
        # output_device_index=1, 
    )
    stream.start_stream()
    while not quit_event.is_set():
        try:
            data = queue.get(block=True, timeout=1)
            stream.write(data)
        except queue.Empty:
            continue
    stream.stop_stream()
    stream.close()
    p.terminate()

class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.sessionid = self.opt.sessionid

        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt,self)
        elif opt.tts == "gpt-sovits":
            self.tts = SovitsTTS(opt,self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt,self)
        elif opt.tts == "cosyvoice":
            self.tts = CosyVoiceTTS(opt,self)
        elif opt.tts == "fishtts":
            self.tts = FishTTS(opt,self)
        elif opt.tts == "tencent":
            self.tts = TencentTTS(opt,self)
        elif opt.tts == "doubao":
            self.tts = DoubaoTTS(opt,self)
        
        self.speaking = False

        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.curr_state=0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        self.__loadcustom()

        # 日志节流阀
        self.last_log_time_qsize = 0
        self.last_log_time_fps = 0

    async def put_msg_txt(self, text_stream, eventpoint=None, **tts_options):
        """
        [MODIFIED] 接收一个文本流，并逐句送入TTS引擎。
        Receives a text stream and sends it sentence by sentence to the TTS engine.
        """
        async for sentence in text_stream:
            if sentence:
                self.tts.put_msg_txt(sentence, eventpoint, **tts_options)
    
    def put_audio_frame(self,audio_chunk,eventpoint=None): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk,eventpoint)

    def put_audio_file(self,filebyte): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk:
            self.put_audio_frame(stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self,byte_stream):
        # 从字节流中读取音频数据
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        # 如果是多声道，则只取第一个声道
        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        # 如果采样率不匹配，则进行重采样
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        self.tts.flush_talk()
        self.asr.flush_talk()

    def is_speaking(self)->bool:
        return self.speaking
    
    def __loadcustom(self):
        # 加载自定义的待机动画和音频
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        # 初始化自定义状态的索引
        self.curr_state=0
        for key in self.custom_audio_index:
            self.custom_audio_index[key]=0
        for key in self.custom_index:
            self.custom_index[key]=0

    def notify(self,eventpoint):
        logger.info("notify:%s",eventpoint)

    def start_recording(self):
        """开始录制视频"""
        if self.recording:
            return

        # 视频录制命令
        command = ['ffmpeg',
                   '-y', '-an',
                   '-f', 'rawvideo',
                   '-vcodec','rawvideo',
                   '-pix_fmt', 'bgr24', #像素格式
                   '-s', "{}x{}".format(self.width, self.height),
                   '-r', str(25),
                   '-i', '-',
                   '-pix_fmt', 'yuv420p', 
                   '-vcodec', "h264",
                   f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        # 音频录制命令
        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
    
    def record_video_data(self,image):
        # 写入视频帧数据
        if self.width == 0:
            print("image.shape:",image.shape)
            self.height,self.width,_ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tobytes()) # 使用 tobytes() 替代 tostring()

    def record_audio_data(self,frame):
        # 写入音频帧数据
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tobytes()) # 使用 tobytes() 替代 tostring()
    
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
        self.recording = False 
        
        # 关闭并等待视频管道
        if self._record_video_pipe and self._record_video_pipe.stdin:
            self._record_video_pipe.stdin.close()
            self._record_video_pipe.wait()
        
        # 关闭并等待音频管道
        if self._record_audio_pipe and self._record_audio_pipe.stdin:
            self._record_audio_pipe.stdin.close()
            self._record_audio_pipe.wait()
            
        # 合并音视频文件
        output_path = os.path.join("data", "record.mp4")
        temp_audio = f'temp{self.opt.sessionid}.aac'
        temp_video = f'temp{self.opt.sessionid}.mp4'
        cmd_combine_audio = f"ffmpeg -y -i {temp_video} -i {temp_audio} -c:v copy -c:a aac {output_path}"
        
        try:
            os.makedirs("data", exist_ok=True)
            subprocess.run(cmd_combine_audio, shell=True, check=True)
            logger.info(f"Recording saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to combine audio and video: {e}")
        finally:
            # 清理临时文件
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            if os.path.exists(temp_video):
                os.remove(temp_video)


    def mirror_index(self,size, index):
        # 计算镜像索引，用于循环播放动画
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1 
    
    def get_audio_stream(self,audiotype):
        # 获取自定义状态的音频流
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype]>=self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  # 切换到静音状态
        return stream
    
    def set_custom_state(self,audiotype, reinit=True):
        # 设置当前的自定义状态
        print('set_custom_state:',audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        # 是否启用说话和静音状态之间的平滑过渡效果
        enable_transition = False
        
        if enable_transition:
            _last_speaking = False
            _transition_start = time.time()
            _transition_duration = 0.1  # 过渡时间 (秒)
            _last_silent_frame = None   # 静音帧缓存
            _last_speaking_frame = None # 说话帧缓存
        
        # 如果使用虚拟摄像头输出
        if self.opt.transport=='virtualcam':
            import pyvirtualcam
            vircam = None

            audio_tmp = queue.Queue(maxsize=3000)
            audio_thread = Thread(target=play_audio, args=(quit_event,audio_tmp,), daemon=True, name="pyaudio_stream")
            audio_thread.start()
        
        while not quit_event.is_set():
            try:
                # 从结果队列中获取处理好的帧
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            # 检测状态变化，用于平滑过渡
            if enable_transition:
                current_speaking = not (audio_frames[0][1]!=0 and audio_frames[1][1]!=0)
                if current_speaking != _last_speaking:
                    logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                    _transition_start = time.time()
                _last_speaking = current_speaking

            # 判断是否为静音帧
            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: 
                self.speaking = False
                audiotype = audio_frames[0][1]
                # 如果有自定义待机视频，则播放
                if self.custom_index.get(audiotype) is not None: 
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    # 否则播放默认的循环动画
                    target_frame = self.frame_list_cycle[idx]
                
                # 处理过渡效果
                if enable_transition:
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else: # 是说话帧
                self.speaking = True
                try:
                    # 将生成的口型区域贴回到原始视频帧上
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                
                # 处理过渡效果
                if enable_transition:
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            # 在画面上添加水印
            cv2.putText(combine_frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            
            # 根据传输方式发送视频帧
            if self.opt.transport=='virtualcam':
                if vircam==None:
                    height, width,_= combine_frame.shape
                    vircam = pyvirtualcam.Camera(width=width, height=height, fps=25, fmt=pyvirtualcam.PixelFormat.BGR,print_fps=True)
                vircam.send(combine_frame)
            else: #webrtc
                image = combine_frame
                new_frame = VideoFrame.from_ndarray(image, format="bgr24")
                asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
            
            # 录制视频数据
            self.record_video_data(combine_frame)

            # 处理并发送音频帧
            for audio_frame in audio_frames:
                frame,type,eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)

                if self.opt.transport=='virtualcam':
                    audio_tmp.put(frame.tobytes())
                else: #webrtc
                    new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                    new_frame.planes[0].update(frame.tobytes())
                    new_frame.sample_rate=16000
                    asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
                
                # 录制音频数据
                self.record_audio_data(frame)
            
            if self.opt.transport=='virtualcam':
                vircam.sleep_until_next_frame()

        # 清理工作
        if self.opt.transport=='virtualcam':
            if 'audio_thread' in locals() and audio_thread.is_alive():
                audio_thread.join()
            if vircam:
                vircam.close()
        logger.info('basereal process_frames thread stop')
