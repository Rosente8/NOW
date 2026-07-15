#!/usr/bin/env python3
"""
侦察区 - 桶内颜色标示检测（多线程 + 颜色分割备选）
- 采集线程：从 TCP 拉流
- 推理线程：YOLO 检测桶，不足时用颜色分割补全；检测桶内颜色
- 管道发送线程：发送结果到 C++（文本协议）
- 显示线程：实时显示画面
- 保存纯净的标示图片（最清晰的一张）
"""

import cv2
import torch
import struct
import os
import sys
import time
import queue
import threading
import numpy as np
import fcntl
from datetime import datetime

# ================== 用户配置 ==================
MODEL_PATH = "/home/hy/yolo_test/best_scout.pt"   # 侦察区模型
YOLOV5_REPO = "/home/hy/yolov5"
PIPE_PATH = "/tmp/vision_pipe"                     # 管道（可与投放区共用，但建议独立）
IMG_SIZE = 416
CONF_THRESH = 0.6
TARGET_CLASS = 0

STREAM_URL = "tcp://127.0.0.1:5000"

USE_DISPLAY = True
DISPLAY_FPS = 15
ENABLE_PIPE = True                                 # 是否通过管道发送结果

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
pipe_queue = queue.Queue(maxsize=1)   # 存储待发送的字符串

running = True
pipe_w = None

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

print("正在加载侦察区模型...")
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

# ---------- 创建管道（可选） ----------
if ENABLE_PIPE:
    if not os.path.exists(PIPE_PATH):
        os.mkfifo(PIPE_PATH)
        print(f"📁 创建管道 {PIPE_PATH}")
    print("等待 C++ 程序连接管道...")
    pipe_w = open(PIPE_PATH, 'wb')
    fd = pipe_w.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    print("C++ 已连接管道")
else:
    pipe_w = None

# ---------- 颜色分割找桶（备选方案） ----------
def find_buckets_by_color(frame):
    """通过颜色分割寻找白色圆桶（当YOLO漏检时使用）"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 白色桶的范围（亮度高，饱和度低）
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    buckets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        # 圆度检查
        circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)
        if circularity > 0.6:
            buckets.append((int(cx), int(cy), int(radius)))
    return buckets

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
            # 取区域中心小区域计算 RGB 均值，避免边缘干扰
            cy, cx = h//2, w//2
            size = 5
            center_roi = roi[max(0, cy-size):cy+size, max(0, cx-size):cx+size]
            if center_roi.size == 0:
                center_roi = roi[y:y+h, x:x+w]
            mean_bgr = cv2.mean(center_roi)[:3]
            r, g, b = mean_bgr[2], mean_bgr[1], mean_bgr[0]
            # 用 HSV 均值判断颜色名称
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
    detected = [False] * 5          # 记录每个桶是否已检测到颜色
    best_sharpness = [0.0] * 5      # 每个桶的最佳清晰度
    best_images = [None] * 5        # 每个桶的最佳图片
    best_info = [None] * 5          # 每个桶的最佳颜色信息

    while running:
        if raw_queue.empty():
            time.sleep(0.001)
            continue

        frame = raw_queue.get()

        # ----- 1. YOLO 检测桶 -----
        results = model(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy()
        yolo_boxes = []
        if len(detections) > 0:
            target_dets = detections[detections[:, 5] == TARGET_CLASS]
            for box in target_dets:
                x1, y1, x2, y2, conf, cls = box
                yolo_boxes.append((x1, y1, x2, y2, conf))

        # ----- 2. 如果 YOLO 检测少于 5 个，用颜色分割补全 -----
        color_buckets = find_buckets_by_color(frame) if len(yolo_boxes) < 5 else []

        # ----- 3. 合并结果（去重） -----
        all_centers = []
        # 先加入 YOLO 中心
        for box in yolo_boxes:
            x1, y1, x2, y2, conf = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            all_centers.append((cx, cy, 'yolo', (x1, y1, x2, y2)))
        # 加入颜色分割结果（去重）
        for cx, cy, radius in color_buckets:
            overlap = False
            for mx, my, _, _ in all_centers:
                if np.sqrt((cx - mx)**2 + (cy - my)**2) < radius * 0.8:
                    overlap = True
                    break
            if not overlap:
                # 颜色分割没有框，用半径生成一个框
                x1 = cx - radius
                y1 = cy - radius
                x2 = cx + radius
                y2 = cy + radius
                all_centers.append((cx, cy, 'color', (x1, y1, x2, y2)))

        # 按 x 坐标排序
        all_centers.sort(key=lambda p: p[0])

        # ----- 4. 对每个桶检测颜色（只检测未检测到的桶） -----
        result_frame = frame.copy()
        has_detection = False
        current_results = []   # 用于管道发送

        for idx, (cx, cy, src, bbox) in enumerate(all_centers):
            if idx >= 5:
                break
            if detected[idx]:
                continue

            x1, y1, x2, y2 = bbox
            x1i = max(0, int(x1))
            y1i = max(0, int(y1))
            x2i = min(frame.shape[1], int(x2))
            y2i = min(frame.shape[0], int(y2))
            roi = frame[y1i:y2i, x1i:x2i]
            if roi.size == 0:
                continue

            result = detect_color_in_roi(roi)
            if result:
                sharpness = compute_sharpness(result['image'])
                # 更新最佳图片（保留最清晰的）
                if sharpness > best_sharpness[idx]:
                    best_sharpness[idx] = sharpness
                    best_images[idx] = result['image'].copy()
                    best_info[idx] = (result['color'], result['rgb'], idx+1)
                    # 一旦检测到颜色，标记该桶已检测（但我们允许后续更清晰的图片覆盖，但不再重复输出）
                    # 为了不重复输出，我们可以在最终保存时输出，这里我们先标记 detected
                    if not detected[idx]:
                        detected[idx] = True
                        has_detection = True
                        color, rgb, _ = best_info[idx]
                        # 保存图片
                        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                        filename = f"{SAVE_DIR}/bucket_{idx+1}_{color}_{timestamp}.jpg"
                        cv2.imwrite(filename, best_images[idx])
                        print(f"\n✅ 桶{idx+1} 检测到颜色: {color}  (RGB: {rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f})")
                        print(f"📁 已保存纯净图片: {filename}")
                        # 记录结果用于管道发送
                        current_results.append((idx+1, color, rgb))

                # 在画面上绘制桶框和颜色框
                cv2.rectangle(result_frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                cv2.putText(result_frame, f"B{idx+1}", (x1i, y1i-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                if result:
                    bx, by, bw, bh = result['bbox']
                    cv2.rectangle(result_frame, (x1i+bx, y1i+by), (x1i+bx+bw, y1i+by+bh), (0, 0, 255), 2)
                    cv2.putText(result_frame, result['color'], (x1i+bx, y1i+by-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # ----- 5. 如果没有任何检测，输出 None -----
        if not has_detection and not any(detected):
            print("None")
            current_results.append(("None",))   # 用于管道发送

        # ----- 6. 发送结果到管道 -----
        if ENABLE_PIPE and pipe_w is not None:
            msg = ""
            if len(current_results) == 0:
                msg = "None"
            else:
                # 构建消息，如 "桶1:红色(200,50,50);桶2:黄色(220,210,40)"
                parts = []
                for idx, color, rgb in current_results:
                    parts.append(f"桶{idx}:{color}({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f})")
                msg = ";".join(parts)
            while not pipe_queue.empty():
                try:
                    pipe_queue.get_nowait()
                except queue.Empty:
                    break
            pipe_queue.put(msg + "\n")

        # ----- 7. 显示 -----
        if USE_DISPLAY:
            while not disp_queue.empty():
                try:
                    disp_queue.get_nowait()
                except queue.Empty:
                    break
            disp_queue.put(result_frame)

    print("推理线程退出")

# ---------- 管道发送线程 ----------
def pipe_sender_worker():
    global running, pipe_w
    if not ENABLE_PIPE or pipe_w is None:
        return
    print("管道发送线程已启动")
    while running:
        if pipe_queue.empty():
            time.sleep(0.002)
            continue
        msg = pipe_queue.get()
        try:
            pipe_w.write(msg.encode('utf-8'))
            pipe_w.flush()
        except (BlockingIOError, BrokenPipeError, OSError):
            pass
        except Exception as e:
            print(f"管道发送错误: {e}")
            running = False
            break
    print("管道发送线程退出")

# ---------- 显示线程 ----------
def display_worker():
    global running
    if not USE_DISPLAY:
        return
    print("显示线程已启动")
    cv2.namedWindow("Scout Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scout Detection", 960, 540)

    delay = 1.0 / DISPLAY_FPS

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
    global running, pipe_w

    threads = []
    t_cap = threading.Thread(target=capture_worker, daemon=True)
    t_cap.start()
    threads.append(t_cap)

    t_infer = threading.Thread(target=inference_worker, daemon=True)
    t_infer.start()
    threads.append(t_infer)

    if ENABLE_PIPE and pipe_w is not None:
        t_pipe = threading.Thread(target=pipe_sender_worker, daemon=True)
        t_pipe.start()
        threads.append(t_pipe)

    if USE_DISPLAY:
        t_disp = threading.Thread(target=display_worker, daemon=True)
        t_disp.start()
        threads.append(t_disp)

    print("侦察区多线程检测已启动，按 Ctrl+C 退出")
    print("检测到颜色时输出桶号、颜色和RGB，无检测时输出 None")

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
        print("侦察区程序退出")

if __name__ == "__main__":
    main()