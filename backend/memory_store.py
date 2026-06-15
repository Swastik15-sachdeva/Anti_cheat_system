from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

_sessions: Dict[str, Dict[str, Any]] = {}

def create_session(session_id: str, session_data: Dict[str, Any]) -> None:
    """Create a new session."""
    if session_id in _sessions:
        logger.warning(f"Session {session_id} already exists, overwriting")
    _sessions[session_id] = session_data
    logger.info(f"Session created: {session_id}")

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get an existing session."""
    if session_id not in _sessions:
        logger.warning(f"Session {session_id} not found")
        return None
    
    session = _sessions[session_id]
    
    # Check if session has expired
    if "expires_at" in session:
        expires_at = datetime.fromisoformat(session["expires_at"])
        if datetime.now() > expires_at:
            logger.warning(f"Session {session_id} has expired")
            del _sessions[session_id]
            return None
    
    return session

def append_message(session_id: str, role: str, content: str) -> None:
    """Append a message to a session."""
    session = get_session(session_id)
    if not session:
        logger.error(f"Cannot append message - session {session_id} not found")
        return
    
    session["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    logger.debug(f"Message added to session {session_id}: {role}")

def log_violation(session_id: str, v_type: str, timestamp: str, screenshot_base64: str) -> None:
    """Log a proctoring violation."""
    session = get_session(session_id)
    if not session:
        logger.error(f"Cannot log violation - session {session_id} not found")
        return
    
    session["violations"].append({
        "type": v_type,
        "timestamp": timestamp,
        "screenshot_base64": screenshot_base64,
        "logged_at": datetime.now().isoformat()
    })
    logger.info(f"Violation logged: {v_type} for session {session_id}")

def complete_session(session_id: str) -> None:
    """Mark a session as complete."""
    session = get_session(session_id)
    if session:
        session["status"] = "completed"
        session["completed_at"] = datetime.now().isoformat()
        logger.info(f"Session completed: {session_id}")

def clear_session(session_id: str) -> None:
    """Clear a session from memory."""
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Session cleared: {session_id}")
    else:
        logger.warning(f"Cannot clear session - {session_id} not found")

def get_all_sessions() -> Dict[str, Dict[str, Any]]:
    """Get all active sessions (for debugging)."""
    # Clean up expired sessions
    expired_sessions = []
    for session_id, session_data in _sessions.items():
        if "expires_at" in session_data:
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.now() > expires_at:
                expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del _sessions[session_id]
        logger.info(f"Cleaned up expired session: {session_id}")
    
    return _sessions

def cleanup_expired_sessions() -> int:
    """Remove all expired sessions and return count."""
    initial_count = len(_sessions)
    expired_sessions = []
    
    for session_id, session_data in _sessions.items():
        if "expires_at" in session_data:
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.now() > expires_at:
                expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del _sessions[session_id]
    
    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    return len(expired_sessions)

def get_session_stats(session_id: str) -> Dict[str, Any]:
    """Get statistics for a session."""
    session = get_session(session_id)
    if not session:
        return {}
    
    violations = session.get("violations", [])
    violation_counts = {}
    for v in violations:
        v_type = v.get("type", "Unknown")
        violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
    
    return {
        "session_id": session_id,
        "total_violations": len(violations),
        "violation_counts": violation_counts,
        "total_messages": len(session.get("messages", [])),
        "status": session.get("status", "unknown"),
        "created_at": session.get("created_at"),
        "completed_at": session.get("completed_at")
    }