#!/usr/bin/env python3
"""
侦察区 - 流式桶内颜色标示检测
- 无人机边飞边看，每帧检测当前画面中的桶
- 检测到桶内有颜色时，立即保存图片并显示
- 多线程架构：采集、推理、显示分离
"""

import cv2
import torch
import os
import sys
import time
import queue
import threading
import numpy as np
from datetime import datetime

# ================== 用户配置 ==================
MODEL_PATH = "/home/hy/yolo_test/best.pt"
YOLOV5_REPO = "/home/hy/yolov5"
IMG_SIZE = 416
CONF_THRESH = 0.6
TARGET_CLASS = 0

STREAM_URL = "tcp://127.0.0.1:5000"

USE_DISPLAY = True
SAVE_DIR = "/home/hy/marker_captures"
MARKER_MIN_AREA = 100
# ==============================================

# ---------- 颜色阈值（HSV） ----------
COLOR_RANGES = {
    '红色': ((0, 50, 50), (10, 255, 255)),
    '红色2': ((170, 50, 50), (180, 255, 255)),
    '橙色': ((5, 50, 50), (15, 255, 255)),
    '黄色': ((20, 50, 50), (30, 255, 255)),
    '绿色': ((40, 50, 50), (80, 255, 255)),
    '蓝色': ((100, 50, 50), (130, 255, 255)),
    '紫色': ((130, 50, 50), (160, 255, 255)),
}

def get_color_name(h, s, v):
    if s < 30:
        return '白色' if v > 70 else '灰色' if v > 30 else '黑色'
    if h < 10 or h >= 170: return '红色'
    if h < 20: return '橙色'
    if h < 35: return '黄色'
    if h < 80: return '绿色'
    if h < 140: return '蓝色'
    return '紫色'

# ---------- 全局队列 ----------
raw_queue = queue.Queue(maxsize=1)
disp_queue = queue.Queue(maxsize=1)

running = True

# ---------- 创建保存目录 ----------
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"📁 创建保存目录 {SAVE_DIR}")

# ---------- 检查 CUDA ----------
if not torch.cuda.is_available():
    print("⚠️ CUDA 不可用，使用 CPU")
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

# ---------- 颜色标示检测 ----------
def detect_color_in_roi(roi):
    if roi is None or roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MARKER_MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cy, cx = h//2, w//2
            size = 5
            center_roi = roi[max(0, cy-size):cy+size, max(0, cx-size):cx+size]
            if center_roi.size == 0:
                center_roi = roi[y:y+h, x:x+w]
            mean_bgr = cv2.mean(center_roi)[:3]
            r, g, b = mean_bgr[2], mean_bgr[1], mean_bgr[0]
            mean_hsv = cv2.mean(cv2.cvtColor(roi[y:y+h, x:x+w], cv2.COLOR_BGR2HSV))[:3]
            color = get_color_name(mean_hsv[0], mean_hsv[1], mean_hsv[2])
            return {
                'color': color,
                'rgb': (r, g, b),
                'image': roi[y:y+h, x:x+w].copy(),
                'bbox': (x, y, w, h)
            }
    return None

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

# ---------- 采集线程 ----------
def capture_worker():
    global running
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("❌ 无法连接 TCP 视频流")
        running = False
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("✅ 采集线程已启动")

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
    print("采集线程退出")

# ---------- 推理线程 ----------
def inference_worker():
    global running
    print("推理线程已启动")
    # 记录已经检测过的桶编号（按位置编号，防止重复保存）
    detected_ids = set()

    while running:
        if raw_queue.empty():
            time.sleep(0.001)
            continue

        frame = raw_queue.get()
        result_frame = frame.copy()
        has_detection = False

        # ----- YOLO 检测桶 -----
        results = model(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy()
        yolo_boxes = []
        if len(detections) > 0:
            target_dets = detections[detections[:, 5] == TARGET_CLASS]
            for box in target_dets:
                x1, y1, x2, y2, conf, cls = box
                yolo_boxes.append((x1, y1, x2, y2, conf))

        # 按 x 坐标排序
        yolo_boxes.sort(key=lambda b: b[0])

        # ----- 对每个桶检测颜色 -----
        for idx, box in enumerate(yolo_boxes):
            # 用位置编号（左到右）作为桶标识，超过5个只处理前5个
            bucket_id = idx + 1
            if bucket_id > 5:
                break

            # 如果该桶已经检测过颜色，跳过
            if bucket_id in detected_ids:
                continue

            x1, y1, x2, y2, conf = box
            x1i = max(0, int(x1))
            y1i = max(0, int(y1))
            x2i = min(frame.shape[1], int(x2))
            y2i = min(frame.shape[0], int(y2))

            # 提取桶内区域（缩进避免桶壁干扰）
            margin = 10
            roi = frame[y1i+margin:y2i-margin, x1i+margin:x2i-margin]
            if roi.size == 0:
                continue

            result = detect_color_in_roi(roi)
            if result:
                has_detection = True
                color = result['color']
                r, g, b = result['rgb']

                # 清晰度评价
                sharpness = compute_sharpness(result['image'])

                # 标记该桶已检测
                detected_ids.add(bucket_id)

                # 保存图片
                timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                filename = f"{SAVE_DIR}/bucket_{bucket_id}_{color}_{timestamp}.jpg"
                cv2.imwrite(filename, result['image'])

                # 终端输出
                print(f"\n📸 桶{bucket_id} 检测到颜色: {color}  (RGB: {r:.0f}, {g:.0f}, {b:.0f})  清晰度: {sharpness:.1f}")
                print(f"📁 已保存: {filename}")

                # 在画面绘制
                cv2.rectangle(result_frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                cv2.putText(result_frame, f"B{bucket_id}", (x1i, y1i-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                bx, by, bw, bh = result['bbox']
                cv2.rectangle(result_frame, (x1i+bx, y1i+by), (x1i+bx+bw, y1i+by+bh), (0, 0, 255), 2)
                cv2.putText(result_frame, color, (x1i+bx, y1i+by-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # 如果当前帧没有任何桶有颜色，但画面中有桶被检测到但还没颜色，显示桶框
        # 这样可以实时看到识别过程
        if not has_detection and len(yolo_boxes) > 0:
            for idx, box in enumerate(yolo_boxes):
                if idx >= 5:
                    break
                bucket_id = idx + 1
                x1, y1, x2, y2, conf = box
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(result_frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                status = "✅" if bucket_id in detected_ids else "⏳"
                cv2.putText(result_frame, f"B{bucket_id} {status}", (x1i, y1i-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 显示
        if USE_DISPLAY:
            while not disp_queue.empty():
                try:
                    disp_queue.get_nowait()
                except queue.Empty:
                    break
            disp_queue.put(result_frame)

    print("推理线程退出")

# ---------- 显示线程 ----------
def display_worker():
    global running
    if not USE_DISPLAY:
        return
    print("显示线程已启动")
    cv2.namedWindow("Scout Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scout Detection", 960, 540)
    delay = 1.0 / 15

    while running:
        if not disp_queue.empty():
            frame = disp_queue.get()
            cv2.imshow("Scout Detection", frame)
            if cv2.waitKey(1) == 27:   # ESC
                running = False
                break
        else:
            time.sleep(delay)

    cv2.destroyAllWindows()
    print("显示线程退出")

# ---------- 主程序 ----------
def main():
    global running

    threads = []
    t_cap = threading.Thread(target=capture_worker, daemon=True)
    t_cap.start()
    threads.append(t_cap)

    t_infer = threading.Thread(target=inference_worker, daemon=True)
    t_infer.start()
    threads.append(t_infer)

    if USE_DISPLAY:
        t_disp = threading.Thread(target=display_worker, daemon=True)
        t_disp.start()
        threads.append(t_disp)

    print("\n" + "="*50)
    print("侦察区检测已启动")
    print("无人机向前飞，检测到桶内有颜色时自动保存图片")
    print("按 Ctrl+C 退出")
    print("="*50 + "\n")

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n用户中断")
        running = False
    finally:
        for t in threads:
            t.join(timeout=1)
        print("程序退出")

if __name__ == "__main__":
    main()