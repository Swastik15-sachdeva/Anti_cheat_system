from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import base64
import io
from PIL import Image
from ultralytics import YOLO
import logging
import uuid
from datetime import datetime, timedelta
import json

import memory_store
import analyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Interviewer API", version="1.0.0")

# Configure CORS - FIXED SECURITY ISSUE
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3001",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Initialize YOLOv8 model at startup
print("Loading YOLOv8 model...")
try:
    yolo_model = YOLO("yolov8n.pt")
    logger.info("YOLOv8 model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLOv8 model: {e}")
    yolo_model = None

# Pydantic models with validation
class StartSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=5000)

class LogViolationRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1, max_length=200)
    timestamp: str = Field(..., min_length=1)
    screenshot_base64: str = Field(..., max_length=5_000_000)  # 5MB max

class DetectObjectsRequest(BaseModel):
    screenshot_base64: str = Field(..., max_length=5_000_000)

# Error handling
class APIError(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        logger.error(f"API Error ({status_code}): {detail}")

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "AI Interviewer Backend is running",
        "version": "1.0.0"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "yolo_loaded": yolo_model is not None
    }

@app.post("/api/start-session")
def start_session(request: StartSessionRequest):
    """
    Start a new interview session.
    """
    try:
        session_data = {
            "session_id": request.session_id,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
            "messages": [],
            "violations": [],
            "status": "active"
        }
        memory_store.create_session(request.session_id, session_data)
        logger.info(f"Session started: {request.session_id}")
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "message": "Session created successfully"
        }
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise APIError(500, f"Failed to start session: {str(e)}")

@app.post("/api/chat")
def handle_chat(request: ChatRequest):
    """
    Receive a message from the candidate and return the AI's response.
    """
    try:
        # Validate session exists
        session = memory_store.get_session(request.session_id)
        if not session:
            raise APIError(404, "Session not found")
        
        ai_response = analyzer.analyze_answer(request.session_id, request.message)
        logger.info(f"Chat processed for session: {request.session_id}")
        
        return {
            "status": "success",
            "response": ai_response
        }
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise APIError(500, f"Failed to process chat: {str(e)}")

@app.post("/api/log-violation")
def api_log_violation(request: LogViolationRequest):
    """
    Log a new proctoring violation to backend memory.
    """
    try:
        # Validate session exists
        session = memory_store.get_session(request.session_id)
        if not session:
            raise APIError(404, "Session not found")
        
        # Validate violation type
        valid_types = [
            "Face Missing from Frame",
            "Multiple Faces Detected",
            "Cell Phone Detected",
            "Tab Switched / Left Window",
            "Exited Fullscreen",
            "Looked Away from Screen",
            "Eye Shifting / Rapid Eye Movement",
            "Face Partially Hidden"
        ]
        
        if request.type not in valid_types:
            logger.warning(f"Unknown violation type: {request.type}")
        
        memory_store.log_violation(
            request.session_id,
            request.type,
            request.timestamp,
            request.screenshot_base64
        )
        
        logger.info(f"Violation logged: {request.type} for session {request.session_id}")
        
        return {
            "status": "success",
            "message": "Violation logged successfully"
        }
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error logging violation: {e}")
        raise APIError(500, f"Failed to log violation: {str(e)}")

@app.post("/api/detect-objects")
def api_detect_objects(request: DetectObjectsRequest):
    """
    Receive a base64 image and detect objects using YOLOv8.
    """
    try:
        if yolo_model is None:
            logger.warning("YOLOv8 model not loaded, returning empty detection")
            return {
                "status": "success",
                "detected": [],
                "message": "Model not available"
            }
        
        # Decode base64 image
        base64_str = request.screenshot_base64
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run inference with timeout handling
        results = yolo_model(image, verbose=False, conf=0.5)
        
        detected = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                name = yolo_model.names[cls_id]
                confidence = float(box.conf[0].item())
                detected.append({
                    "class": name,
                    "confidence": confidence
                })
        
        logger.debug(f"Objects detected: {len(detected)}")
        
        return {
            "status": "success",
            "detected": [d["class"] for d in detected],
            "detections": detected
        }
    except Exception as e:
        logger.error(f"Error in YOLOv8 detection: {e}")
        return {
            "status": "error",
            "message": str(e),
            "detected": []
        }

@app.get("/api/get-report")
def api_get_report(session_id: str):
    """
    End the interview and generate a final report incorporating all logged violations.
    """
    try:
        # Validate session exists
        session = memory_store.get_session(session_id)
        if not session:
            raise APIError(404, "Session not found")
        
        report = analyzer.generate_interview_report(session_id)
        logger.info(f"Report generated for session: {session_id}")
        
        # Mark session as complete
        memory_store.complete_session(session_id)
        
        return report
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise APIError(500, f"Failed to generate report: {str(e)}")

@app.get("/api/session/{session_id}")
def get_session_info(session_id: str):
    """
    Get current session information.
    """
    try:
        session = memory_store.get_session(session_id)
        if not session:
            raise APIError(404, "Session not found")
        
        return {
            "status": "success",
            "session_id": session_id,
            "violations_count": len(session.get("violations", [])),
            "messages_count": len(session.get("messages", [])),
            "created_at": session.get("created_at"),
            "status": session.get("status", "unknown")
        }
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session info: {e}")
        raise APIError(500, f"Failed to retrieve session: {str(e)}")

@app.exception_handler(APIError)
async def api_exception_handler(request, exc):
    return {"detail": exc.detail, "status": "error"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Interviewer Backend server...")
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")