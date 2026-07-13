#!/usr/bin/env python3
"""
Jetson Nano 视觉检测 - 从 TCP 视频流拉取图像
- 服务端用 gst-launch 推送 jpeg 到 tcp://127.0.0.1:5000
- 客户端通过 OpenCV 或 GStreamer pipeline 拉流
- 模型 imgsz=416, class_id=0
- 通过命名管道发送圆心坐标给 C++
- 按 ESC 退出
"""

import cv2
import torch
import struct
import os
import sys
import time
import numpy as np

# ================== 用户配置 ==================
MODEL_PATH = "/home/hy/yolo_test/best.pt"
YOLOV5_REPO = "/home/hy/yolov5"
PIPE_PATH = "/tmp/vision_pipe"
IMG_SIZE = 416
CONF_THRESH = 0.6
TARGET_CLASS = 0

# TCP 视频流地址（与服务端一致）
TCP_STREAM_URL = "tcp://127.0.0.1:5000"
# 备选：如果 OpenCV 不支持 tcp://，可使用 GStreamer 管道
GST_PIPELINE = (
    "tcpclientsrc host=127.0.0.1 port=5000 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
)
# ==============================================

# ---------- 检查 CUDA ----------
if not torch.cuda.is_available():
    print("⚠️ CUDA 不可用，使用 CPU")
    device = 'cpu'
else:
    print(f"✅ CUDA 可用，设备: {torch.cuda.get_device_name(0)}")

# ---------- 加载模型 ----------
if not os.path.exists(YOLOV5_REPO):
    print(f"❌ YOLOv5 仓库不存在: {YOLOV5_REPO}")
    sys.exit(1)
if not os.path.exists(MODEL_PATH):
    print(f"❌ 模型文件不存在: {MODEL_PATH}")
    sys.exit(1)

print("正在加载模型...")
try:
    model = torch.hub.load(YOLOV5_REPO, 'custom',
                           path=MODEL_PATH, source='local',
                           device='0', force_reload=True)
    model.conf = CONF_THRESH
    model.classes = [TARGET_CLASS]
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# ---------- 创建命名管道（与 C++ 通信） ----------
if not os.path.exists(PIPE_PATH):
    os.mkfifo(PIPE_PATH)
    print(f"📁 创建管道 {PIPE_PATH}")

print("等待 C++ 程序连接管道...")
pipe_w = open(PIPE_PATH, 'wb')

# ---------- 打开 TCP 视频流 ----------
def open_tcp_stream():
    """
    尝试以多种方式打开 TCP 视频流。
    优先使用直接地址（简单），若不支持则使用 GStreamer pipeline。
    """
    # 方式1：直接使用 tcp:// 地址（需要 OpenCV 支持 GStreamer 后端）
    cap = cv2.VideoCapture(TCP_STREAM_URL)
    if cap.isOpened():
        print("✅ 通过 tcp:// 地址成功打开视频流")
        return cap

    # 方式2：使用 GStreamer pipeline（更通用，但需要 OpenCV 支持 GStreamer）
    # 检查 OpenCV 是否支持 GStreamer
    has_gst = cv2.videoio_registry.hasBackend(cv2.CAP_GSTREAMER)
    if has_gst:
        cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("✅ 通过 GStreamer pipeline 成功打开视频流")
            return cap

    print("❌ 无法打开 TCP 视频流，请确保服务端已启动（gst-launch 命令）")
    return None

cap = open_tcp_stream()
if cap is None:
    sys.exit(1)

# ---------- 显示窗口 ----------
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection", 960, 540)
print("开始检测，按 ESC 退出")

# ---------- 主循环 ----------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取帧失败，重试...")
            time.sleep(0.05)
            continue
        cv2.imshow("Video Stream", frame)


        # ---------- 推理 ----------
        results = model(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy()

        cx = cy = conf = 0.0
        if len(detections) > 0:
            target_dets = detections[detections[:, 5] == TARGET_CLASS]
            if len(target_dets) > 0:
                best = target_dets[target_dets[:, 4].argmax()]
                x1, y1, x2, y2, conf, cls = best
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # 可视化
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({cx:.1f},{cy:.1f}) {conf:.2f}",
                            (int(cx)+10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        # ---------- 通过命名管道发送坐标 ----------
        pipe_w.write(struct.pack('fff', cx, cy, conf))
        pipe_w.flush()

        # ---------- 显示 ----------
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) == 27:   # ESC
            break

        time.sleep(0.005)

except KeyboardInterrupt:
    print("用户中断")
finally:
    cap.release()
    pipe_w.close()
    cv2.destroyAllWindows()
    print("程序已退出")