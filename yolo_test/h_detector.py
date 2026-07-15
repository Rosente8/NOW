#!/usr/bin/env python3
"""
起降区 - H 标识检测（多线程文本协议版）
- 采集线程：从 TCP 拉流
- 推理线程：YOLO 检测 H，计算中心坐标
- 发送线程：通过管道发送文本消息 "None" 或 "cx,cy"
- 显示线程：实时显示检测画面
"""

import cv2
import torch
import os
import sys
import time
import queue
import threading
import numpy as np
import fcntl

# ================== 用户配置 ==================
MODEL_PATH = "/home/hy/yolo_test/best_h.pt"   # H 模型路径
YOLOV5_REPO = "/home/hy/yolov5"
PIPE_PATH = "/tmp/h_pipe"                     # H 专用管道
IMG_SIZE = 416
CONF_THRESH = 0.6
H_CLASS = 0                                   # H 在训练中的类别 ID

STREAM_URL = "tcp://127.0.0.1:5000"

USE_DISPLAY = True
DISPLAY_FPS = 15
# ==============================================

raw_queue = queue.Queue(maxsize=1)
disp_queue = queue.Queue(maxsize=1)
pipe_queue = queue.Queue(maxsize=1)           # 存储待发送的字符串

running = True
pipe_w = None

# ---------- 检查 CUDA ----------
if not torch.cuda.is_available():
    print("⚠️ CUDA 不可用，使用 CPU")
else:
    print(f"✅ CUDA 可用，设备: {torch.cuda.get_device_name(0)}")

# ---------- 加载 H 模型 ----------
if not os.path.exists(YOLOV5_REPO):
    print(f"❌ YOLOv5 仓库不存在: {YOLOV5_REPO}")
    sys.exit(1)
if not os.path.exists(MODEL_PATH):
    print(f"❌ H 模型文件不存在: {MODEL_PATH}")
    sys.exit(1)

print("正在加载 H 模型...")
try:
    model = torch.hub.load(YOLOV5_REPO, 'custom',
                           path=MODEL_PATH, source='local',
                           device='0', force_reload=True)
    model.conf = CONF_THRESH
    model.classes = [H_CLASS]
    print("✅ H 模型加载成功")
except Exception as e:
    print(f"❌ H 模型加载失败: {e}")
    sys.exit(1)

# ---------- 创建命名管道（非阻塞） ----------
if not os.path.exists(PIPE_PATH):
    os.mkfifo(PIPE_PATH)
    print(f"📁 创建 H 管道 {PIPE_PATH}")

print("等待 C++ 程序连接 H 管道...")
pipe_w = open(PIPE_PATH, 'wb')
# 设置非阻塞（避免写满时阻塞）
fd = pipe_w.fileno()
flags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
print("C++ 已连接 H 管道")

# ---------- 采集线程 ----------
def capture_worker():
    global running
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("❌ 无法连接 TCP 视频流")
        running = False
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("✅ H 采集线程已启动")

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue
        while not raw_queue.empty():
            try:
                raw_queue.get_nowait()
            except queue.Empty:
                break
        raw_queue.put(frame)

    cap.release()
    print("H 采集线程退出")

# ---------- 推理线程 ----------
def inference_worker():
    global running
    print("H 推理线程已启动")

    while running:
        if raw_queue.empty():
            time.sleep(0.001)
            continue

        frame = raw_queue.get()

        results = model(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy()

        # 筛选 H 类别
        h_dets = detections[detections[:, 5] == H_CLASS] if len(detections) > 0 else []

        cx = cy = 0.0
        result_frame = frame.copy()
        message = "None"   # 默认消息

        if len(h_dets) > 0:
            # 取置信度最高的 H
            best = h_dets[h_dets[:, 4].argmax()]
            x1, y1, x2, y2, conf, cls = best
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # 构造消息：坐标字符串
            message = f"{cx:.2f},{cy:.2f}"

            # 绘制
            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.circle(result_frame, (int(cx), int(cy)), 5, (0, 255, 255), -1)
            cv2.putText(result_frame, f"H ({cx:.1f},{cy:.1f})", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # 控制台输出
        print(message)

        # 发送消息到管道队列（添加换行符）
        while not pipe_queue.empty():
            try:
                pipe_queue.get_nowait()
            except queue.Empty:
                break
        pipe_queue.put(message + "\n")

        # 显示
        if USE_DISPLAY:
            while not disp_queue.empty():
                try:
                    disp_queue.get_nowait()
                except queue.Empty:
                    break
            disp_queue.put(result_frame)

    print("H 推理线程退出")

# ---------- 管道发送线程（发送文本消息） ----------
def pipe_sender_worker():
    global running, pipe_w
    print("H 管道发送线程已启动")

    while running:
        if pipe_queue.empty():
            time.sleep(0.002)
            continue

        msg = pipe_queue.get()
        try:
            # 发送文本（已包含换行符）
            pipe_w.write(msg.encode('utf-8'))
            pipe_w.flush()
        except (BlockingIOError, BrokenPipeError, OSError):
            # 管道忙或断开，丢弃该条消息
            pass
        except Exception as e:
            print(f"H 管道发送错误: {e}")
            running = False
            break

    print("H 管道发送线程退出")

# ---------- 显示线程 ----------
def display_worker():
    global running
    if not USE_DISPLAY:
        return

    print("H 显示线程已启动")
    cv2.namedWindow("H Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("H Detection", 960, 540)

    delay = 1.0 / DISPLAY_FPS

    while running:
        if not disp_queue.empty():
            frame = disp_queue.get()
            cv2.imshow("H Detection", frame)
            if cv2.waitKey(1) == 27:   # ESC
                running = False
                break
        else:
            time.sleep(delay)

    cv2.destroyAllWindows()
    print("H 显示线程退出")

# ---------- 主程序 ----------
def main():
    global running, pipe_w

    threads = []
    t_cap = threading.Thread(target=capture_worker, daemon=True)
    t_cap.start()
    threads.append(t_cap)

    t_infer = threading.Thread(target=inference_worker, daemon=True)
    t_infer.start()
    threads.append(t_infer)

    t_pipe = threading.Thread(target=pipe_sender_worker, daemon=True)
    t_pipe.start()
    threads.append(t_pipe)

    if USE_DISPLAY:
        t_disp = threading.Thread(target=display_worker, daemon=True)
        t_disp.start()
        threads.append(t_disp)

    print("所有 H 检测线程已启动，按 Ctrl+C 退出")
    print("检测到 H 时发送坐标字符串，未检测到时发送 'None'")

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n用户中断")
        running = False
    finally:
        for t in threads:
            t.join(timeout=1)
        if pipe_w:
            pipe_w.close()
        print("H 检测程序退出")

if __name__ == "__main__":
    main()