from typing import List, Dict, Any

_sessions: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "messages": [],
            "violations": []
        }
    return _sessions[session_id]

def append_message(session_id: str, role: str, content: str) -> None:
    session = get_session(session_id)
    session["messages"].append({"role": role, "content": content})

def log_violation(session_id: str, v_type: str, timestamp: str, screenshot_base64: str) -> None:
    session = get_session(session_id)
    session["violations"].append({
        "type": v_type,
        "timestamp": timestamp,
        "screenshot_base64": screenshot_base64
    })

def clear_session(session_id: str) -> None:
    """Clear the session from memory."""
    if session_id in _sessions:
        del _sessions[session_id]
