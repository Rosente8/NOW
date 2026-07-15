#!/usr/bin/env python3
"""
侦察区 - 桶内颜色标示检测
- 从 TCP 拉流
- YOLO 检测圆桶
- 在每个桶内检测颜色区域
- 每个桶只检测一次，输出颜色、RGB
- 保存纯净的标示区域图片
"""

import cv2
import torch
import os
import numpy as np
from datetime import datetime


MODEL_PATH = "/home/hy/yolo_test/best.pt"
YOLOV5_REPO = "/home/hy/yolov5"
IMG_SIZE = 416
CONF_THRESH = 0.6
SAVE_DIR = "/home/hy/marker_captures"


# ---------- 创建保存目录 ----------
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print("加载 YOLO 模型...")
model = torch.hub.load(YOLOV5_REPO, 'custom', path=MODEL_PATH,
                       source='local', device='0', force_reload=True)
model.conf = CONF_THRESH
model.classes = [0]
print("模型加载完成")

# ---------- HSV 颜色阈值 ----------
COLOR_RANGES = {
    '红色': ((0, 50, 50), (10, 255, 255)),
    '红色2': ((170, 50, 50), (180, 255, 255)),
    '橙色': ((5, 50, 50), (15, 255, 255)),
    '黄色': ((20, 50, 50), (30, 255, 255)),
    '绿色': ((40, 50, 50), (80, 255, 255)),
    '蓝色': ((100, 50, 50), (130, 255, 255)),
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

# ---------- 检测颜色区域 ----------
def detect_color_in_roi(roi):
    if roi is None or roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
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
            # 返回纯净的颜色区域图像（不包含任何标记）
            return {
                'color': color,
                'rgb': (r, g, b),
                'image': roi[y:y+h, x:x+w].copy(),  # 纯净原图
                'bbox': (x, y, w, h)
            }
    return None

# ---------- 主程序 ----------
print("连接视频流...")
cap = cv2.VideoCapture("tcp://127.0.0.1:5000")
if not cap.isOpened():
    print("无法连接视频流，请检查服务端是否启动")
    exit()

print("开始检测，按 q 退出")
detected = [False] * 5  # 每个桶是否已检测到颜色

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取帧失败，重试...")
        continue

    # YOLO 检测（每1帧检测一次，可根据需要调整）
    results = model(frame, size=IMG_SIZE)
    detections = results.xyxy[0].cpu().numpy()
    buckets = detections[detections[:, 5] == 0] if len(detections) > 0 else []

    if len(buckets) > 0:
        # 按 x 坐标从左到右排序
        buckets = sorted(buckets, key=lambda b: b[0])
        for idx, box in enumerate(buckets):
            if idx >= 5:
                break
            if detected[idx]:
                continue

            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 提取桶内区域（缩进一点，避免桶壁干扰）
            margin = 10
            roi = frame[y1+margin:y2-margin, x1+margin:x2-margin]
            if roi.size == 0:
                continue

            # 检测颜色
            result = detect_color_in_roi(roi)
            if result:
                color = result['color']
                r, g, b = result['rgb']
                detected[idx] = True

                # 输出结果
                print(f"\n✅ 桶{idx+1} 检测到颜色: {color}  (RGB: {r:.0f}, {g:.0f}, {b:.0f})")

                # 保存纯净的标示图片（无任何标记）
                timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                filename = f"{SAVE_DIR}/bucket_{idx+1}_{color}_{timestamp}.jpg"
                cv2.imwrite(filename, result['image'])
                print(f"📁 已保存纯净图片: {filename}")

            # 在显示画面上绘制桶框和颜色框（仅用于观察，不影响保存的图片）
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"B{idx+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if result and 'bbox' in result:
                bx, by, bw, bh = result['bbox']
                cv2.rectangle(frame, (x1+bx, y1+by), (x1+bx+bw, y1+by+bh), (0, 0, 255), 2)
                cv2.putText(frame, color, (x1+bx, y1+by-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # 显示画面
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 清理
cap.release()
cv2.destroyAllWindows()
print("\n程序退出")