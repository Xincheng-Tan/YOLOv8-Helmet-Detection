import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = './weights/best.pt'
model = YOLO(MODEL_PATH)

# 0 - person ; 1 - head ; 2 - helmet
TARGET_CLASSES = [0, 1, 2] 

def detect_safety_helmet(image_np):
    if image_np is None:
        return np.zeros((640, 640, 3), dtype=np.uint8)

    results = model.predict(
        source=image_np, 
        classes=TARGET_CLASSES,
        conf=0.3, 
        iou=0.7, 
        save=False,
        verbose=False
    )

    plotted_img_bgr = results[0].plot() 
    # plotted_img_rgb = cv2.cvtColor(plotted_img_bgr, cv2.COLOR_BGR2RGB)
    
    return plotted_img_bgr
    # return plotted_img_rgb

iface = gr.Interface(
    fn=detect_safety_helmet,
    inputs=gr.Image(type="numpy", label="上传图片"),
    outputs=gr.Image(type="numpy", label="检测结果"),
    title="👷 YOLOv8 安全帽检测演示",
    description="上传一张图片，模型将检测图中的安全帽并显示结果"
)

if __name__ == "__main__":
    iface.launch(
        server_port=7860,
        server_name="0.0.0.0"
    )