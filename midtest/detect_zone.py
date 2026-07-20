#!/usr/bin/env python3
"""
侦察区 - 桶内全色块比例分析系统（加强版）
功能：
  1. 优先 YOLO 检测白色圆桶
  2. 若 YOLO 漏检，自动用颜色定位补全（色块扩展为桶区域）
  3. 自动裁剪桶内区域，统计每种颜色的像素占比（包括所有颜色）
  4. 以表格和进度条形式清晰打印每个桶的颜色比例
  5. 汇总显示所有桶的详细颜色信息
  6. 保存桶 ROI 照片（文件名含颜色信息）
  7. 多线程架构：采集、推理、显示分离，高效流畅
  8. 新增：为每个检测到的桶创建独立小窗口，放大显示ROI并叠加颜色比例信息（便于同时对比观察多个桶）
  9. 新增：独立小窗口按从左到右水平排列，窗口标题为 "Bucket X"，信息栏显示主色和占比
  10. 新增：空桶窗口显示 "Empty"，有标识桶显示颜色占比
  11. 新增：窗口自动更新，每帧刷新显示最新ROI
"""

import cv2
import torch
import os
import sys
import time
import queue
import threading
import numpy as np
import math
from datetime import datetime
from collections import defaultdict

# ================== 用户配置（请根据实际修改）==================
MODEL_PATH = "/home/hy/yolo_test/midtest/best.pt"
YOLOV5_REPO = "/home/hy/yolov5"
IMG_SIZE = 416              # 与训练尺寸一致
CONF_THRESH = 0.4
TARGET_CLASS = 0            # 桶的类别ID

STREAM_URL = "tcp://127.0.0.1:5000"
USE_DISPLAY = True
SAVE_DIR = "/home/hy/marker_captures"
MARKER_MIN_AREA = 120       # 色块最小面积（像素），过滤灰尘和反光
MARGIN = 15                 # 桶内缩进像素（避开桶壁）
MAX_BUCKETS = 5
EXPAND_RATIO = 2.5          # 色块扩展到桶的倍数
# =============================================================

# ---------- HSV颜色阈值（完整覆盖所有常见颜色） ----------
COLOR_RANGES = {
    '红色':   ((0, 20, 20), (10, 255, 255)),
    '红色2':  ((160, 20, 20), (180, 255, 255)),
    '橙色':   ((10, 20, 20), (20, 255, 255)),
    '黄色':   ((25, 20, 20), (35, 255, 255)),
    '绿色':   ((40, 20, 20), (80, 255, 255)),
    '蓝色':   ((100, 20, 20), (130, 255, 255)),
    '紫色':   ((135, 20, 20), (160, 255, 255)),
    '黑色':   ((0, 0, 0), (180, 255, 30)),      # 黑色（低明度）
    '灰色':   ((0, 0, 30), (180, 30, 70)),      # 灰色（低饱和度，中明度）
    '白色':   ((0, 0, 70), (180, 30, 255)),     # 白色（低饱和度，高明度）
}

# 定义需要忽略的颜色（桶本身是白色，背景可能灰色，需忽略）
IGNORE_COLORS = ['白色', '灰色', '黑色']

# ---------- 颜色定位函数（兜底方案） ----------
def find_color_buckets(frame, min_area=80):
    """
    直接在画面中搜索彩色区域，扩展为桶区域
    返回：list of (x1, y1, x2, y2)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    buckets = []
    # 只检测非黑白灰的颜色
    for color_name, (lower, upper) in COLOR_RANGES.items():
        if color_name in IGNORE_COLORS:
            continue
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w//2
            center_y = y + h//2
            bucket_size = int(max(w, h) * EXPAND_RATIO)
            bx = max(0, center_x - bucket_size//2)
            by = max(0, center_y - bucket_size//2)
            bw = min(frame.shape[1] - bx, bucket_size)
            bh = min(frame.shape[0] - by, bucket_size)
            if bw > 30 and bh > 30 and bw < frame.shape[1]*0.8:
                buckets.append((bx, by, bx+bw, by+bh))
    return buckets

def merge_buckets(buckets, iou_thresh=0.3):
    """合并重叠的桶框"""
    if not buckets:
        return []
    buckets = sorted(buckets, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    merged = []
    for b in buckets:
        overlap = False
        for m in merged:
            x1 = max(b[0], m[0])
            y1 = max(b[1], m[1])
            x2 = min(b[2], m[2])
            y2 = min(b[3], m[3])
            if x2 > x1 and y2 > y1:
                inter = (x2-x1)*(y2-y1)
                area_b = (b[2]-b[0])*(b[3]-b[1])
                area_m = (m[2]-m[0])*(m[3]-m[1])
                if inter / min(area_b, area_m) > iou_thresh:
                    overlap = True
                    break
        if not overlap:
            merged.append(b)
    return merged

# ---------- 核心分析函数：统计桶内所有色块比例 ----------
def analyze_color_proportions(roi, min_area=60):
    """
    输入：桶的ROI（BGR）
    输出：字典 {颜色名称: 占比百分比}，按占比降序排列
    自动排除白色/灰色/黑色（因为桶本身是白色底）
    """
    if roi is None or roi.size == 0:
        return {}
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total_pixels = roi.shape[0] * roi.shape[1]
    
    color_area_map = defaultdict(int)
    
    for color_name, (lower, upper) in COLOR_RANGES.items():
        # 跳过白色/灰色/黑色（这些是背景，不是标识）
        if color_name in IGNORE_COLORS:
            continue
            
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # 形态学去噪
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_sum = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                area_sum += area
        
        if area_sum > 0:
            color_area_map[color_name] += area_sum
    
    # 合并红色（HSV中红色分两段）
    if '红色' in color_area_map and '红色2' in color_area_map:
        color_area_map['红色'] += color_area_map['红色2']
        del color_area_map['红色2']
    elif '红色2' in color_area_map:
        color_area_map['红色'] = color_area_map.pop('红色2')
    
    # 计算百分比，过滤小于0.5%的噪点
    final_proportions = {}
    for color, area in color_area_map.items():
        pct = (area / total_pixels) * 100.0
        if pct > 0.5:
            final_proportions[color] = round(pct, 2)
    
    return dict(sorted(final_proportions.items(), key=lambda item: item[1], reverse=True))

def format_proportions(proportions):
    """格式化输出颜色比例，用于打印"""
    if not proportions:
        return "无颜色"
    parts = []
    for color, pct in proportions.items():
        parts.append(f"{color}:{pct:.1f}%")
    return ", ".join(parts)

def print_progress_bars(proportions, total_width=40):
    """打印带进度条的颜色比例"""
    if not proportions:
        print("  无有效颜色")
        return
    max_pct = max(proportions.values())
    for color, pct in proportions.items():
        bar_len = int((pct / max_pct) * total_width) if max_pct > 0 else 0
        bar = "█" * bar_len + "░" * (total_width - bar_len)
        print(f"  {color:<4}  {pct:>5.2f}%  {bar}")

# ---------- 全局队列 ----------
raw_queue = queue.Queue(maxsize=1)
disp_queue = queue.Queue(maxsize=1)
running = True

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if not torch.cuda.is_available():
    print("⚠️ CUDA 不可用，使用 CPU")
else:
    print(f"✅ CUDA 可用，设备: {torch.cuda.get_device_name(0)}")

if not os.path.exists(YOLOV5_REPO) or not os.path.exists(MODEL_PATH):
    print("❌ 模型或仓库路径错误")
    sys.exit(1)

print("正在加载模型...")
try:
    model = torch.hub.load(YOLOV5_REPO, 'custom',
                           path=MODEL_PATH, source='local',
                           device='0', force_reload=False)
    model.conf = CONF_THRESH
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# ---------- 采集线程 ----------
def capture_worker():
    global running
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("❌ 无法连接视频流")
        running = False
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("✅ 采集线程启动")
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
    print("推理线程启动")
    detected_ids = set()

    while running:
        if raw_queue.empty():
            time.sleep(0.001)
            continue

        frame = raw_queue.get()
        result_frame = frame.copy()
        
        # ----- 1. YOLO检测桶 -----
        results = model(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy()
        yolo_boxes = []
        if len(detections) > 0:
            target_dets = detections[detections[:, 5] == TARGET_CLASS]
            for box in target_dets:
                x1, y1, x2, y2, conf, cls = box
                yolo_boxes.append((x1, y1, x2, y2, conf))
        yolo_boxes.sort(key=lambda b: b[0])
        
        # ----- 2. 如果YOLO检测不足5个，用颜色定位补全 -----
        if len(yolo_boxes) < MAX_BUCKETS:
            color_buckets_raw = find_color_buckets(frame, min_area=80)
            color_buckets = merge_buckets(color_buckets_raw, iou_thresh=0.3)
            color_boxes = [(b[0], b[1], b[2], b[3], 0.0) for b in color_buckets]
            all_boxes = yolo_boxes + color_boxes
            all_boxes.sort(key=lambda b: b[0])
            # 去重（基于IoU）
            merged = []
            for box in all_boxes:
                x1, y1, x2, y2, conf = box
                overlap = False
                for m in merged:
                    mx1, my1, mx2, my2, _ = m
                    inter_x1 = max(x1, mx1)
                    inter_y1 = max(y1, my1)
                    inter_x2 = min(x2, mx2)
                    inter_y2 = min(y2, my2)
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter = (inter_x2-inter_x1)*(inter_y2-inter_y1)
                        area1 = (x2-x1)*(y2-y1)
                        area2 = (mx2-mx1)*(my2-my1)
                        if inter / min(area1, area2) > 0.3:
                            overlap = True
                            break
                if not overlap:
                    merged.append((x1, y1, x2, y2, conf))
            final_boxes = merged
        else:
            final_boxes = yolo_boxes
        
        # ----- 3. 分析每个桶 -----
        bucket_results = {}
        for idx, box in enumerate(final_boxes):
            bucket_id = idx + 1
            if bucket_id > MAX_BUCKETS:
                break
            
            x1, y1, x2, y2, conf = box
            x1i = max(0, int(x1))
            y1i = max(0, int(y1))
            x2i = min(frame.shape[1], int(x2))
            y2i = min(frame.shape[0], int(y2))
            
            roi = frame[y1i+MARGIN:y2i-MARGIN, x1i+MARGIN:x2i-MARGIN]
            if roi.size == 0:
                continue

            proportions = analyze_color_proportions(roi, min_area=MARKER_MIN_AREA)
            bucket_results[bucket_id] = {
                'proportions': proportions,
                'roi': roi,
                'bbox': (x1i, y1i, x2i, y2i)
            }
            
            # ----- 打印详细色块比例（每个桶） -----
            if proportions:
                # 保存照片
                timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                color_summary = "_".join(list(proportions.keys())[:2]) if proportions else "empty"
                filename = f"{SAVE_DIR}/bucket_{bucket_id}_{color_summary}_{timestamp}.jpg"
                cv2.imwrite(filename, roi)
                
                print(f"\n{'='*50}")
                print(f"📸 桶 [{bucket_id}] 检测到危险标识！")
                print(f"📁 保存: {filename}")
                print(f"📊 颜色比例 (总像素: {roi.shape[0]*roi.shape[1]}):")
                print_progress_bars(proportions, total_width=40)
                print(f"{'='*50}\n")
                
                # 窗口标注：显示所有颜色比例
                y_offset = y1i - 10
                for color, pct in list(proportions.items())[:3]:
                    text = f"{color}:{pct:.1f}%"
                    cv2.putText(result_frame, text, (x1i, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    y_offset -= 18
                cv2.rectangle(result_frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 3)
                cv2.putText(result_frame, f"B{bucket_id} [标识]", (x1i, y1i-35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 画中画
                roi_display = cv2.resize(roi, (150, 150))
                h, w, _ = result_frame.shape
                result_frame[20:20+150, w-170:w-20] = roi_display
                cv2.rectangle(result_frame, (w-170, 20), (w-20, 170), (255, 255, 255), 2)
                cv2.putText(result_frame, f"Bucket {bucket_id}", (w-160, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # 空桶
                cv2.rectangle(result_frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 1)
                cv2.putText(result_frame, f"B{bucket_id} [Empty]", (x1i, y1i-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

            # ----- 新增：独立小窗口显示每个桶的放大ROI + 颜色信息 -----
            win_name = f"Bucket {bucket_id}"
            # 准备显示内容
            roi_display = cv2.resize(roi, (160, 160))
            # 创建信息栏（白色背景）
            info_bar = np.ones((20, 160, 3), dtype=np.uint8) * 255
            if proportions:
                # 显示占比最大的颜色及百分比
                primary_color = max(proportions.items(), key=lambda x: x[1])
                color_text = f"{primary_color[0]}:{primary_color[1]:.1f}%"
                cv2.putText(info_bar, color_text, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            else:
                cv2.putText(info_bar, "Empty", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            combined = np.vstack([info_bar, roi_display])
            cv2.imshow(win_name, combined)
            cv2.moveWindow(win_name, 50 + (bucket_id-1)*180, 50)  # 水平排列

        # ----- 4. 汇总打印（检测到5个桶时） -----
        if len(bucket_results) == MAX_BUCKETS:
            print("\n" + "="*50)
            print("📋 侦察区 5桶状态汇总 (从左→右):")
            for i in range(1, MAX_BUCKETS+1):
                info = bucket_results.get(i)
                if info and info['proportions']:
                    props_str = format_proportions(info['proportions'])
                    print(f"  桶{i}: ✅ {props_str}")
                else:
                    print(f"  桶{i}: ⚪ 空桶")
            print("="*50 + "\n")
        
        # ----- 送入显示队列 -----
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
    print("显示线程启动")
    cv2.namedWindow("Scout Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scout Detection", 1280, 720)
    delay = 1.0 / 15
    
    while running:
        if not disp_queue.empty():
            frame = disp_queue.get()
            cv2.imshow("Scout Detection", frame)
            if cv2.waitKey(1) == 27:
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
    print("🚁 侦察区 - 色块比例分析系统 (加强版)")
    print("功能：YOLO检测 + 颜色定位兜底 → 精细色块比例分析 + 独立小窗显示")
    print("操作：按 ESC 或 Ctrl+C 退出")
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