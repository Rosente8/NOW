#!/usr/bin/env python3
"""
- 摄像头实时采集
- YOLOv5 推理，返回圆形桶圆心坐标
- 通过命名管道将坐标发给 C++ 控制程序
- RTSP 推流供远程观看
"""

import cv2
import numpy as np
import threading
import queue
import time
import struct
import os
from ultralytics import YOLO

# 配置参数
PIPE_PATH = "/tmp/vision_pipe"   # 命名管道路径

CONF_THRESH = 0.7
IMG_SIZE = 416
MODEL_ENGINE = "best.engine"
MODEL_PT = "best.pt"
USE_CSI = True
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
RTSP_UDP_PORT = 8554

# 管道初始化
# 如果管道文件不存在则创建
if not os.path.exists(PIPE_PATH):
    os.mkfifo(PIPE_PATH)
    print(f"✅ 创建命名管道: {PIPE_PATH}")

# 打开管道写端（会阻塞直到 C++ 打开读端）
print("等待 C++ 程序连接管道...")
pipe_fd = os.open(PIPE_PATH, os.O_WRONLY)
print("✅ C++ 已连接，开始发送数据")
# 包装为文件对象方便写入
pipe_out = os.fdopen(pipe_fd, 'wb')

# ================== 模型加载 ==================
if os.path.exists(MODEL_ENGINE):
    model = YOLO(MODEL_ENGINE, task='detect')
    print(f"✅ 加载 TensorRT 引擎: {MODEL_ENGINE}")
else:
    model = YOLO(MODEL_PT, task='detect')
    print(f"⚠️ 未找到引擎，加载 PyTorch 模型: {MODEL_PT}")

model.conf = CONF_THRESH
model.classes = [0]

# ================== 摄像头初始化 ==================
if USE_CSI:
    pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={FRAME_WIDTH}, height={FRAME_HEIGHT}, framerate={FPS}/1 ! "
        f"nvvidconv flip-method=2 ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink"
    )
    cap = cv2.VideoCapture(pipeline)
else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    print("❌ 摄像头打开失败")
    exit(1)

# ================== 线程安全的帧队列 ==================
frame_queue = queue.Queue(maxsize=1)

def capture_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

threading.Thread(target=capture_loop, daemon=True).start()

# ================== RTSP 推流管道 ==================
try:
    rtsp_out = cv2.VideoWriter(
        f"appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=800 speed-preset=superfast ! "
        f"rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port={RTSP_UDP_PORT}",
        cv2.CAP_GSTREAMER,
        0,
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
        True
    )
except Exception as e:
    print(f"❌ RTSP 推流初始化失败: {e}")
    rtsp_out = None

# ================== 主循环 ==================
print("🚀 视觉模块已启动，按 Ctrl+C 退出")
try:
    while True:
        frame = frame_queue.get()

        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, classes=[0], verbose=False)
        boxes = results[0].boxes

        best_cx, best_cy, best_conf = 0.0, 0.0, 0.0
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            idx = np.argmax(confs)
            box = boxes.xyxy[idx].cpu().numpy()
            best_conf = confs[idx]
            x1, y1, x2, y2 = box
            best_cx = (x1 + x2) / 2.0
            best_cy = (y1 + y2) / 2.0

            cv2.circle(frame, (int(best_cx), int(best_cy)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"({best_cx:.1f},{best_cy:.1f}) c:{best_conf:.2f}",
                        (int(best_cx) + 10, int(best_cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 通过管道发送坐标（12字节）
        data = struct.pack('fff', best_cx, best_cy, best_conf)
        pipe_out.write(data)
        pipe_out.flush()   # 立即推送

        if rtsp_out is not None:
            rtsp_out.write(frame)

        cv2.imshow("Detector", frame)
        if cv2.waitKey(1) == 27:
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    if rtsp_out is not None:
        rtsp_out.release()
    cv2.destroyAllWindows()
    pipe_out.close()
    print("🛑 视觉模块已退出")