#!/usr/bin/env python3
"""
Jetson Nano 视觉检测 - 投放区（自动标定 + 稳定分类）
- 支持交互式标定（输入已知直径）
- 标定后自动保存焦距，后续直接加载
- 高度和像素直径双重平滑
- 在任何高度下稳定输出桶ID和物理直径
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
import json

# ================== 用户配置 ==================
MODEL_PATH = "/home/hy/yolo_test/midtest/best.pt"
YOLOV5_REPO = "/home/hy/yolov5"
PIPE_PATH = "/tmp/vision_pipe"
HEIGHT_PIPE = "/tmp/lidar_height_pipe"
IMG_SIZE = 416
CONF_THRESH = 0.6
TARGET_CLASS = 0

STREAM_URL = "tcp://127.0.0.1:5000"
USE_DISPLAY = True
DISPLAY_FPS = 15
DEBUG_MODE = True

# ================== 物理尺寸参数 ==================
CALIB_CX = 640.0
CALIB_CY = 360.0

# 默认焦距（如果无标定文件）
DEFAULT_FX = 1800.0
FOCAL_FILE = "/home/hy/focal_length.json"

# 分类区间
CLASSIFICATION_INTERVALS = [
    (0.110, 0.190, 1),
    (0.160, 0.240, 2),
    (0.210, 0.290, 3)
]

DEFAULT_HEIGHT = 1.0
# ==============================================

# 全局变量
current_height = DEFAULT_HEIGHT
height_lock = threading.Lock()
running = True
focal_ready = False

raw_queue = queue.Queue(maxsize=1)
disp_queue = queue.Queue(maxsize=1)
pipe_queue = queue.Queue(maxsize=1)
pipe_w = None


# ================== 标定相关函数 ==================
def load_focal_length():
    try:
        with open(FOCAL_FILE, 'r') as f:
            data = json.load(f)
            return data.get('fx', None)
    except:
        return None

def save_focal_length(fx):
    with open(FOCAL_FILE, 'w') as f:
        json.dump({'fx': fx}, f)
    print(f"💾 焦距已保存: {fx:.1f}")

def interactive_calibration():
    """交互式标定：用户放置已知直径的桶，程序自动计算焦距"""
    print("\n" + "="*60)
    print("🔧 进入焦距标定模式")
    print("请将一个已知直径的桶（如20cm）放置在画面中央")
    print("确保激光雷达高度数据正常")
    print("按 'c' 开始标定，按 'q' 退出标定使用默认值")
    print("="*60 + "\n")

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("❌ 无法连接视频流")
        return None

    # 加载模型（已在主程序加载，但此处需要引用）
    model_local = torch.hub.load(YOLOV5_REPO, 'custom', path=MODEL_PATH,
                                 source='local', device='0', force_reload=False)
    model_local.conf = CONF_THRESH
    model_local.classes = [TARGET_CLASS]

    samples = []
    calibrating = False
    real_diameter = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 读取当前高度
        height = DEFAULT_HEIGHT
        try:
            if os.path.exists(HEIGHT_PIPE):
                with open(HEIGHT_PIPE, 'rb') as f:
                    data = f.read(4)
                    if len(data) == 4:
                        h = struct.unpack('f', data)[0]
                        if 0.1 < h < 20.0:
                            height = h
        except:
            pass

        # YOLO检测
        results = model_local(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy()
        target_dets = detections[detections[:, 5] == TARGET_CLASS] if len(detections) > 0 else []

        # 在画面上显示
        for box in target_dets:
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"conf={conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if calibrating and len(target_dets) > 0:
            # 取置信度最高的桶
            best = target_dets[np.argmax(target_dets[:, 4])]
            x1, y1, x2, y2, conf, cls = best
            pixel_diam = (x2 - x1 + y2 - y1) / 2.0
            if pixel_diam > 20 and real_diameter > 0:
                f_est = pixel_diam * height / real_diameter
                if 500 < f_est < 3000:
                    samples.append(f_est)
                    print(f"采样 {len(samples)}: 像素直径={pixel_diam:.1f}px, 高度={height:.2f}m, 焦距={f_est:.1f}")

        # 显示信息
        info = f"高度: {height:.2f}m"
        if calibrating:
            info += f" | 采样数: {len(samples)}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and not calibrating:
            if len(target_dets) == 0:
                print("⚠️ 未检测到桶，请将桶置于画面中")
                continue
            try:
                diam_cm = float(input("请输入该桶的真实直径（cm）: "))
                real_diameter = diam_cm / 100.0
                calibrating = True
                samples.clear()
                print("开始采样... 请保持桶在画面中，程序将自动采样")
            except:
                print("输入无效")

        elif key == ord('q'):
            break

        elif calibrating and len(samples) >= 15:
            # 取中位数
            focal = np.median(samples)
            print(f"\n✅ 标定完成！焦距 = {focal:.1f}")
            save_focal_length(focal)
            cv2.destroyAllWindows()
            cap.release()
            return focal

    cv2.destroyAllWindows()
    cap.release()
    return None


# ================== 物理尺寸估算器 ==================
class PhysicalSizeEstimator:
    def __init__(self, fx, fy):
        self.fx = fx
        self.fy = fy
        self.cx = CALIB_CX
        self.cy = CALIB_CY
        self.height_filter = []
        self.filter_window = 10

    def compute_physical_diameter(self, pixel_width, pixel_height, height_m, cx, cy):
        if height_m > 0.1 and height_m < 20.0:
            self.height_filter.append(height_m)
            if len(self.height_filter) > self.filter_window:
                self.height_filter.pop(0)
            height_m = sum(self.height_filter) / len(self.height_filter)

        pixel_diameter = (pixel_width + pixel_height) / 2.0
        f_pixel = (self.fx + self.fy) / 2.0

        if pixel_diameter > 500 or pixel_diameter < 5 or f_pixel <= 0:
            return 0.0, 0.0

        dx = cx - self.cx
        dy = cy - self.cy
        distance_pixel = math.sqrt(dx*dx + dy*dy)
        theta = math.atan(distance_pixel / f_pixel) if f_pixel > 0 else 0
        cos_theta = math.cos(theta)
        if cos_theta < 0.1:
            cos_theta = 0.1

        physical_raw = pixel_diameter * (height_m / f_pixel)
        physical_corrected = physical_raw / cos_theta

        if DEBUG_MODE:
            print(f"   [诊断] 像素宽={pixel_width:.1f}, 高={pixel_height:.1f}, 直径={pixel_diameter:.1f}px")
            print(f"          高度={height_m:.2f}m, 离轴角={theta:.3f}rad, cosθ={cos_theta:.3f}")
            print(f"          修正前={physical_raw*100:.1f}cm, 修正后={physical_corrected*100:.1f}cm")

        if physical_corrected < 0.05 or physical_corrected > 0.50:
            return 0.0, theta
        return physical_corrected, theta

    @staticmethod
    def classify_bucket(physical_diameter_m):
        if physical_diameter_m <= 0:
            return None, physical_diameter_m
        STANDARD = {1: 0.15, 2: 0.20, 3: 0.25}
        for lower, upper, bid in CLASSIFICATION_INTERVALS:
            if lower <= physical_diameter_m < upper:
                return bid, physical_diameter_m
        # 兜底
        best_id = 1
        best_dist = float('inf')
        for bid, std in STANDARD.items():
            dist = abs(physical_diameter_m - std)
            if dist < best_dist:
                best_dist = dist
                best_id = bid
        if best_dist > 0.05:
            return None, physical_diameter_m
        return best_id, physical_diameter_m


# ---------- 高度读取线程 ----------
def height_reader():
    global current_height
    while running:
        try:
            if not os.path.exists(HEIGHT_PIPE):
                time.sleep(0.5)
                continue
            with open(HEIGHT_PIPE, 'rb') as f:
                while running:
                    data = f.read(4)
                    if len(data) == 4:
                        h = struct.unpack('f', data)[0]
                        if 0.1 < h < 20.0:
                            with height_lock:
                                current_height = h
                    else:
                        time.sleep(0.01)
        except Exception as e:
            time.sleep(1)


# ---------- 检查 CUDA ----------
if not torch.cuda.is_available():
    print("⚠️ CUDA 不可用，使用 CPU")
else:
    print(f"✅ CUDA 可用，设备: {torch.cuda.get_device_name(0)}")

# ---------- 加载模型 ----------
if not os.path.exists(YOLOV5_REPO) or not os.path.exists(MODEL_PATH):
    print("❌ 模型或仓库路径错误")
    sys.exit(1)

print("正在加载模型...")
try:
    model = torch.hub.load(YOLOV5_REPO, 'custom',
                           path=MODEL_PATH, source='local',
                           device='0', force_reload=False)
    model.conf = CONF_THRESH
    model.classes = [TARGET_CLASS]
    print("✅ 模型加载成功")
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to('cuda' if torch.cuda.is_available() else 'cpu')
    model(dummy)
    print("✅ 模型预热完成")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# ---------- 标定或加载焦距 ----------
focal = load_focal_length()
if focal is None:
    print("未找到标定文件，进入交互式标定...")
    focal = interactive_calibration()
    if focal is None:
        focal = DEFAULT_FX
        print(f"⚠️ 标定未完成，使用默认焦距 {DEFAULT_FX}")
    else:
        print(f"✅ 标定完成，焦距 = {focal:.1f}")
else:
    print(f"📌 加载标定焦距: {focal:.1f}")

CALIB_FX = focal
CALIB_FY = focal

# ---------- 创建桶管道 ----------
if not os.path.exists(PIPE_PATH):
    os.mkfifo(PIPE_PATH)
    print(f"📁 创建桶管道 {PIPE_PATH}")

print("等待 C++ 飞控程序连接桶管道...")
pipe_w = open(PIPE_PATH, 'wb')
fd = pipe_w.fileno()
flags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
print("C++ 飞控程序已连接桶管道")


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
    global running, current_height
    print("推理线程已启动")

    estimator = PhysicalSizeEstimator(CALIB_FX, CALIB_FY)
    frame_count = 0

    while running:
        if raw_queue.empty():
            time.sleep(0.001)
            continue

        frame = raw_queue.get()
        frame_count += 1

        with height_lock:
            height_m = current_height

        results = model(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy()

        target_dets = detections[detections[:, 5] == TARGET_CLASS] if len(detections) > 0 else []

        bucket_list = []
        result_frame = frame.copy()

        for box in target_dets:
            x1, y1, x2, y2, conf, cls = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            pixel_width = x2 - x1
            pixel_height = y2 - y1

            physical_diam, theta = estimator.compute_physical_diameter(
                pixel_width, pixel_height, height_m, cx, cy
            )

            if physical_diam <= 0:
                continue

            bucket_id, final_diam = estimator.classify_bucket(physical_diam)

            if bucket_id is not None:
                bucket_list.append((bucket_id, cx, cy))
                cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(result_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                label = f"桶{bucket_id} ({final_diam*100:.1f}cm)"
                cv2.putText(result_frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                print(f"✅ 桶{bucket_id}: 物理直径={final_diam*100:.1f}cm, 坐标=({cx:.1f}, {cy:.1f})")
            else:
                cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                label = f"? ({physical_diam*100:.1f}cm)"
                cv2.putText(result_frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                print(f"❌ 无法分类: {physical_diam*100:.1f}cm")

        if len(bucket_list) == 0 and frame_count % 30 == 0:
            print("None")

        while not pipe_queue.empty():
            try:
                pipe_queue.get_nowait()
            except queue.Empty:
                break
        pipe_queue.put((len(bucket_list), bucket_list))

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
    while running:
        if pipe_queue.empty():
            time.sleep(0.002)
            continue
        count, bucket_list = pipe_queue.get()
        try:
            pipe_w.write(struct.pack('B', count))
            for bucket_id, cx, cy in bucket_list:
                pipe_w.write(struct.pack('B', bucket_id))
                pipe_w.write(struct.pack('ff', cx, cy))
            pipe_w.flush()
        except Exception as e:
            print(f"管道发送错误: {e}")
            running = False
            break


# ---------- 显示线程 ----------
def display_worker():
    global running
    if not USE_DISPLAY:
        return
    cv2.namedWindow("Drop Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drop Detection", 960, 540)
    delay = 1.0 / DISPLAY_FPS
    while running:
        if not disp_queue.empty():
            frame = disp_queue.get()
            cv2.imshow("Drop Detection", frame)
            if cv2.waitKey(1) == 27:
                running = False
                break
        else:
            time.sleep(delay)
    cv2.destroyAllWindows()


# ---------- 主程序 ----------
def main():
    global running, pipe_w

    # 启动高度读取线程
    t_height = threading.Thread(target=height_reader, daemon=True)
    t_height.start()

    threads = []
    for target in [capture_worker, inference_worker, pipe_sender_worker]:
        t = threading.Thread(target=target, daemon=True)
        t.start()
        threads.append(t)

    if USE_DISPLAY:
        t = threading.Thread(target=display_worker, daemon=True)
        t.start()
        threads.append(t)

    print("\n" + "="*60)
    print("投放区检测（自动标定版）")
    print(f"焦距: {CALIB_FX:.1f}")
    print("输出格式: 桶X: 物理直径=XX.Xcm, 坐标=(cx, cy)")
    print("按 Ctrl+C 退出")
    print("="*60 + "\n")

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("用户中断")
        running = False
    finally:
        for t in threads:
            t.join(timeout=1)
        if pipe_w:
            pipe_w.close()
        print("程序退出")


if __name__ == "__main__":
    main()