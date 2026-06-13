import urllib.request
import os
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# Resolve paths relative to backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
# Models are served by FastAPI from backend/static/models/ via /static/models/
static_models_dir = os.path.join(backend_dir, "static", "models")
os.makedirs(static_models_dir, exist_ok=True)

# 1. Export YOLOv8n (if not already copied/exported)
yolo_onnx_path = os.path.join(static_models_dir, "yolov8n.onnx")
local_yolo_onnx = os.path.join(backend_dir, "yolov8n.onnx")

if not os.path.exists(yolo_onnx_path):
    if os.path.exists(local_yolo_onnx):
        print(f"Copying existing yolov8n.onnx to {yolo_onnx_path}...")
        os.rename(local_yolo_onnx, yolo_onnx_path)
    else:
        print("Exporting yolov8n to ONNX...")
        model = YOLO("yolov8n.pt")
        model.export(format="onnx")
        os.rename(local_yolo_onnx, yolo_onnx_path)
        print("yolov8n exported and saved to static/models/")
else:
    print("yolov8n.onnx already exists in static/models/")

# 2. Download and Export Hand Detection model
hand_pt_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt"
hand_pt_path = os.path.join(backend_dir, "hand_yolov8n.pt")
hand_onnx_path = os.path.join(static_models_dir, "hand_yolov8n.onnx")
local_hand_onnx = os.path.join(backend_dir, "hand_yolov8n.onnx")

if not os.path.exists(hand_onnx_path):
    if not os.path.exists(hand_pt_path):
        print("Downloading hand_yolov8n.pt from Hugging Face...")
        urllib.request.urlretrieve(hand_pt_url, hand_pt_path)
        print("Download complete.")
    
    print("Exporting hand_yolov8n to ONNX...")
    model_hand = YOLO(hand_pt_path)
    model_hand.export(format="onnx")
    os.rename(local_hand_onnx, hand_onnx_path)
    print("hand_yolov8n exported and saved to static/models/")
else:
    print("hand_yolov8n.onnx already exists in static/models/")

print("All models exported successfully to backend/static/models/ (served at /static/models/)!")
