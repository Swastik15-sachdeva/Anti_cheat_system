from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import base64
import io
from PIL import Image
from ultralytics import YOLO

import memory_store
import analyzer

app = FastAPI(title="AI Interviewer API")

# Configure CORS so the React frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local dev (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YOLOv8 model at startup
print("Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")

class ChatRequest(BaseModel):
    session_id: str
    message: str

class LogViolationRequest(BaseModel):
    session_id: str
    type: str
    timestamp: str
    screenshot_base64: str

class DetectObjectsRequest(BaseModel):
    screenshot_base64: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Interviewer Backend is running"}

@app.post("/api/chat")
def handle_chat(request: ChatRequest):
    """
    Receive a message from the candidate and return the AI's response.
    """
    ai_response = analyzer.analyze_answer(request.session_id, request.message)
    return {"response": ai_response}

@app.post("/api/log-violation")
def api_log_violation(request: LogViolationRequest):
    """
    Log a new proctoring violation to backend memory.
    """
    memory_store.log_violation(
        request.session_id, 
        request.type, 
        request.timestamp, 
        request.screenshot_base64
    )
    return {"status": "success"}

@app.post("/api/detect-objects")
def api_detect_objects(request: DetectObjectsRequest):
    """
    Receive a base64 image and detect objects using YOLOv8.
    """
    try:
        base64_str = request.screenshot_base64
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        
        results = yolo_model(image, verbose=False)
        
        detected = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                name = yolo_model.names[cls_id]
                detected.append(name)
                
        return {"status": "success", "detected": detected}
    except Exception as e:
        print(f"Error in YOLOv8 detection: {e}")
        return {"status": "error", "message": str(e), "detected": []}

@app.get("/api/get-report")
def api_get_report(session_id: str):
    """
    End the interview and generate a final report incorporating all logged violations.
    """
    report = analyzer.generate_interview_report(session_id)
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

