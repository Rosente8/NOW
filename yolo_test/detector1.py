#!/usr/bin/env python3
"""
Jetson Nano 视觉检测节点（优先 GStreamer CSI，备选 v4l2）
- 使用 GStreamer pipeline 调用 CSI 摄像头（如果 OpenCV 支持）
- 若 GStreamer 失败，自动回退到 /dev/video0 (v4l2)
- 加载 best.pt 检测目标，提取圆心像素坐标
- 通过命名管道 /tmp/vision_pipe 发送给 C++ 程序
- 实时显示检测画面（按 ESC 退出）
"""

import cv2
import torch
import struct
import os
import time
import sys

# ================== 用户配置（请根据实际修改） ==================
MODEL_PATH = "/home/hy/yolo_test/best.pt"   # 训练好的模型路径
YOLOV5_REPO = "/home/hy/yolov5"             # 本地 YOLOv5 仓库路径
PIPE_PATH = "/tmp/vision_pipe"              # 与 C++ 通信的管道文件
IMG_SIZE = 640                              # 推理尺寸（与摄像头采集尺寸一致最佳）
CONF_THRESH = 0.6                           # 置信度阈值
TARGET_CLASS = 0                            # 要检测的类别ID（circle 为 0）
# ===============================================================

# ---------- 检查 CUDA ----------
if not torch.cuda.is_available():
    print("⚠️ CUDA 不可用，将使用 CPU（速度慢）")
    device = 'cpu'
else:
    print(f"✅ CUDA 可用，设备: {torch.cuda.get_device_name(0)}")

# ---------- 加载模型（兼容新旧 YOLOv5） ----------
if not os.path.exists(YOLOV5_REPO):
    print(f"❌ YOLOv5 仓库不存在: {YOLOV5_REPO}")
    print("请先克隆: git clone https://github.com/ultralytics/yolov5.git ~/yolov5")
    sys.exit(1)
if not os.path.exists(MODEL_PATH):
    print(f"❌ 模型文件不存在: {MODEL_PATH}")
    sys.exit(1)

print("正在加载模型...")
try:
    # 方式1：直接指定 device='0'（新版 YOLOv5 要求）
    model = torch.hub.load(YOLOV5_REPO, 'custom',
                           path=MODEL_PATH, source='local',
                           device='0', force_reload=True)
    model.conf = CONF_THRESH
    if TARGET_CLASS is not None:
        model.classes = [TARGET_CLASS]
    print("✅ 模型加载成功 (device='0')")
except Exception as e1:
    print(f"⚠️ 方式1 失败: {e1}")
    try:
        # 方式2：先加载到 CPU，再手动移到 GPU（兼容旧版）
        model = torch.hub.load(YOLOV5_REPO, 'custom',
                               path=MODEL_PATH, source='local',
                               device='cpu', force_reload=True)
        model.to('cuda')
        model.conf = CONF_THRESH
        if TARGET_CLASS is not None:
            model.classes = [TARGET_CLASS]
        print("✅ 模型加载成功 (先 CPU 后转 GPU)")
    except Exception as e2:
        print(f"❌ 模型加载失败: {e2}")
        sys.exit(1)

# ---------- 创建命名管道 ----------
if not os.path.exists(PIPE_PATH):
    try:
        os.mkfifo(PIPE_PATH)
        print(f"📁 创建管道 {PIPE_PATH}")
    except Exception as e:
        print(f"❌ 创建管道失败: {e}")
        sys.exit(1)

# ---------- GStreamer Pipeline 函数（NVIDIA 官方推荐） ----------
def gstreamer_pipeline(
    capture_width=640,
    capture_height=640,
    display_width=640,
    display_height=640,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# ---------- 打开摄像头（优先 GStreamer，失败则 v4l2） ----------
def open_camera():
    # 检查 OpenCV 是否支持 GStreamer
    has_gst = cv2.videoio_registry.hasBackend(cv2.CAP_GSTREAMER)
    print(f"OpenCV GStreamer 支持: {has_gst}")

    if has_gst:
        # 使用 GStreamer pipeline
        pipeline = gstreamer_pipeline(
            capture_width=640,
            capture_height=640,
            display_width=640,      # 输出尺寸与 IMG_SIZE 一致
            display_height=640,
            framerate=30,
            flip_method=0           # 如果图像倒置可改为 2 或 4
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("✅ GStreamer CSI 摄像头打开成功")
            return cap
        else:
            print("⚠️ GStreamer pipeline 打开失败，尝试 v4l2...")
    else:
        print("⚠️ OpenCV 未编译 GStreamer，直接使用 v4l2")

    # 备选：v4l2 设备 /dev/video0
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("❌ v4l2 设备打开失败")
        return None
    # 设置参数以提高兼容性
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 减少缓冲，降低延迟
    print("✅ v4l2 摄像头打开成功")
    return cap

cap = open_camera()
if cap is None:
    print("❌ 所有摄像头方案均失败，请检查硬件连接和驱动")
    sys.exit(1)

# ---------- 等待 C++ 接收端连接管道 ----------
print("等待 C++ 程序连接管道...")
pipe_w = open(PIPE_PATH, 'wb')   # 阻塞，直到 C++ 打开读端
print("C++ 已连接，开始检测")

# ---------- 显示窗口 ----------
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection", 960, 540)

# ---------- 主循环 ----------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取帧失败，跳过")
            time.sleep(0.01)
            continue

        # ---------- 推理 ----------
        results = model(frame, size=IMG_SIZE)   # 推理
        detections = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,class]

        cx = cy = conf = 0.0
        if len(detections) > 0:
            # 过滤目标类别
            if TARGET_CLASS is not None:
                target_dets = detections[detections[:, 5] == TARGET_CLASS]
            else:
                target_dets = detections
            if len(target_dets) > 0:
                # 选择置信度最高的
                best = target_dets[target_dets[:, 4].argmax()]
                x1, y1, x2, y2, conf, cls = best
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # 可视化（在 frame 上绘制）
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({cx:.1f},{cy:.1f}) {conf:.2f}",
                            (int(cx)+10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        # ---------- 通过管道发送坐标（3个 float） ----------
        data = struct.pack('fff', cx, cy, conf)
        pipe_w.write(data)
        pipe_w.flush()

        # ---------- 显示 ----------
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) == 27:   # ESC 键退出
            break

        # 轻微延时，降低 CPU 占用
        time.sleep(0.005)

except KeyboardInterrupt:
    print("\n用户中断")
finally:
    cap.release()
    pipe_w.close()
    cv2.destroyAllWindows()
    print("程序已退出")