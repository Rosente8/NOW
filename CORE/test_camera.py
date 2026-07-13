import cv2
import torch
from ultralytics import YOLO

# 加载模型（先用PyTorch版本的 best.pt 测试，不需要TensorRT）
model = YOLO("best.pt", task='detect')
model.conf = 0.5   # 阈值暂时设低一点，方便看到检测
model.classes = [0]   # 只检测circle

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("摄像头打不开")
    exit()

print("开始检测，按ESC退出")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 推理
    results = model(frame, imgsz=416, conf=0.5, verbose=False)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf.item()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"circle {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Detection Test", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()