#!/usr/bin/env python3
"""
Jetson Nano 视觉检测 - 投放区专用（基于物理尺寸分类）
- 固定飞行高度：2.0 米
- 利用相机参数反推桶的真实物理直径
- 分类为桶1(15cm)、桶2(20cm)、桶3(25cm)
- 管道发送：数量(1字节) + 每个桶 (ID 1字节 + cx float + cy float)
- 无桶时发送数量0
- 包含分辨率自适应、透视畸变补偿、模糊区间映射
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
import math
import fcntl

# ================== 用户配置 ==================
# --- 模型与路径 ---
MODEL_PATH = "/home/hy/yolo_test/best.pt"
YOLOV5_REPO = "/home/hy/yolov5"
PIPE_PATH = "/tmp/vision_pipe"
IMG_SIZE = 416
CONF_THRESH = 0.6
TARGET_CLASS = 0

STREAM_URL = "tcp://127.0.0.1:5000"

USE_DISPLAY = True
DISPLAY_FPS = 15

# ================== 物理尺寸计算参数 ==================
# 1. 相机标定参数（假设在 640×640 分辨率下测得）
CALIB_FX = 600.0          # 焦距 (像素)
CALIB_FY = 600.0
CALIB_CX = 320.0          # 主点 (图像中心)
CALIB_CY = 320.0
CALIB_WIDTH = 640
CALIB_HEIGHT = 640

# 2. 固定飞行高度 (米) —— 已改为 2.0 米
DEFAULT_HEIGHT = 2.0      # ⚠️ 固定高度 2.0 米

# 3. 分类区间（带缓冲带，单位：米）
CLASSIFICATION_INTERVALS = [
    (0.125, 0.175, 1),   # 12.5cm ~ 17.5cm → 桶1 (15cm)
    (0.175, 0.225, 2),   # 17.5cm ~ 22.5cm → 桶2 (20cm)
    (0.225, 0.325, 3)    # 22.5cm ~ 32.5cm → 桶3 (25cm)
]
# ==============================================

raw_queue = queue.Queue(maxsize=1)
disp_queue = queue.Queue(maxsize=1)
pipe_queue = queue.Queue(maxsize=1)   # 存储 (count, bucket_list)

running = True
pipe_w = None


# ---------- 物理尺寸估算器 ----------
class PhysicalSizeEstimator:
    def __init__(self, calib_fx=CALIB_FX, calib_fy=CALIB_FY,
                 calib_cx=CALIB_CX, calib_cy=CALIB_CY,
                 calib_width=CALIB_WIDTH, calib_height=CALIB_HEIGHT):
        self.fx_ref = calib_fx
        self.fy_ref = calib_fy
        self.cx_ref = calib_cx
        self.cy_ref = calib_cy
        self.width_ref = calib_width
        self.height_ref = calib_height

    def adapt_to_resolution(self, current_width, current_height):
        """线性缩放法：适配当前分辨率"""
        scale_x = current_width / self.width_ref
        scale_y = current_height / self.height_ref
        self.fx_cur = self.fx_ref * scale_x
        self.fy_cur = self.fy_ref * scale_y
        self.cx_cur = self.cx_ref * scale_x
        self.cy_cur = self.cy_ref * scale_y
        self.width_cur = current_width
        self.height_cur = current_height

    def compute_physical_diameter(self, pixel_width, pixel_height, height_m, cx, cy):
        """
        相似三角形 + 透视畸变补偿
        返回：(修正后的物理直径, 离轴角)
        """
        if height_m is None or height_m <= 0:
            height_m = DEFAULT_HEIGHT

        pixel_diameter = (pixel_width + pixel_height) / 2.0

        # 离轴角 θ
        dx = cx - self.cx_cur
        dy = cy - self.cy_cur
        f_pixel = (self.fx_cur + self.fy_cur) / 2.0
        distance_pixel = math.sqrt(dx*dx + dy*dy)
        theta = math.atan(distance_pixel / f_pixel)

        # 原始物理直径（未补偿）
        physical_raw = pixel_diameter * (height_m / f_pixel)

        # 余弦修正
        cos_theta = math.cos(theta)
        if cos_theta > 0.001:
            physical_corrected = physical_raw / cos_theta
        else:
            physical_corrected = physical_raw

        return physical_corrected, theta

    @staticmethod
    def classify_bucket(physical_diameter_m):
        """模糊区间映射：返回 (桶ID, 直径) 或 (None, 直径)"""
        for lower, upper, bucket_id in CLASSIFICATION_INTERVALS:
            if lower <= physical_diameter_m < upper:
                return bucket_id, physical_diameter_m
        return None, physical_diameter_m


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

# ---------- 创建命名管道（非阻塞） ----------
if not os.path.exists(PIPE_PATH):
    os.mkfifo(PIPE_PATH)
    print(f"📁 创建管道 {PIPE_PATH}")

print("等待 C++ 程序连接管道...")
pipe_w = open(PIPE_PATH, 'wb')
fd = pipe_w.fileno()
flags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
print("C++ 已连接")


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


# ---------- 推理线程（物理尺寸分类） ----------
def inference_worker():
    global running
    print("推理线程已启动")

    # 初始化估算器
    estimator = PhysicalSizeEstimator()

    while running:
        if raw_queue.empty():
            time.sleep(0.001)
            continue

        frame = raw_queue.get()
        frame_height, frame_width = frame.shape[:2]

        # 1. 分辨率自适应
        estimator.adapt_to_resolution(frame_width, frame_height)

        # 2. YOLO 检测
        results = model(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy()

        if len(detections) > 0:
            target_dets = detections[detections[:, 5] == TARGET_CLASS]
        else:
            target_dets = []

        bucket_list = []  # 每个元素 (bucket_id, cx, cy)
        result_frame = frame.copy()

        for box in target_dets:
            x1, y1, x2, y2, conf, cls = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            pixel_width = x2 - x1
            pixel_height = y2 - y1

            # 3. 计算物理直径 + 透视补偿
            physical_diam, theta = estimator.compute_physical_diameter(
                pixel_width, pixel_height, DEFAULT_HEIGHT, cx, cy
            )

            # 4. 映射到桶编号
            bucket_id, final_diam = estimator.classify_bucket(physical_diam)

            if bucket_id is not None:
                bucket_list.append((bucket_id, cx, cy))
                # 绘制
                cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(result_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                label = f"桶{bucket_id} ({final_diam*100:.1f}cm)"
                cv2.putText(result_frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                # 无法分类（直径超出范围），仍然显示但标记为?
                cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 红色框提示异常
                label = f"? ({physical_diam*100:.1f}cm)"
                cv2.putText(result_frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # 控制台输出
        if len(bucket_list) == 0:
            print("None")
        else:
            for bid, cx, cy in bucket_list:
                print(f"桶{bid}: ({cx:.1f}, {cy:.1f})")

        # 发送到管道队列
        while not pipe_queue.empty():
            try:
                pipe_queue.get_nowait()
            except queue.Empty:
                break
        pipe_queue.put((len(bucket_list), bucket_list))

        # 显示
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
    print("管道发送线程已启动")

    while running:
        if pipe_queue.empty():
            time.sleep(0.002)
            continue

        count, bucket_list = pipe_queue.get()
        try:
            # 1. 发送桶数量（1字节）
            pipe_w.write(struct.pack('B', count))
            # 2. 发送每个桶的 (ID 1字节 + cx float + cy float)
            for bucket_id, cx, cy in bucket_list:
                pipe_w.write(struct.pack('B', bucket_id))
                pipe_w.write(struct.pack('ff', cx, cy))
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
    cv2.namedWindow("Drop Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drop Detection", 960, 540)

    delay = 1.0 / DISPLAY_FPS

    while running:
        if not disp_queue.empty():
            frame = disp_queue.get()
            cv2.imshow("Drop Detection", frame)
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

    t_pipe = threading.Thread(target=pipe_sender_worker, daemon=True)
    t_pipe.start()
    threads.append(t_pipe)

    if USE_DISPLAY:
        t_disp = threading.Thread(target=display_worker, daemon=True)
        t_disp.start()
        threads.append(t_disp)

    print("\n" + "="*60)
    print("投放区检测已启动（物理尺寸分类）")
    print(f"固定高度: {DEFAULT_HEIGHT} 米")
    print("检测到桶时输出：桶ID (cx, cy)；无桶时输出 None")
    print("按 Ctrl+C 退出")
    print("="*60 + "\n")

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
        print("程序已退出")


if __name__ == "__main__":
    main()