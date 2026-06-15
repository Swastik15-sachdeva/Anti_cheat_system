from typing import List, Dict, Any
import memory_store
import logging

logger = logging.getLogger(__name__)

# Violation severity weights (used in scoring)
VIOLATION_WEIGHTS = {
    "Multiple Faces Detected": 8,
    "Cell Phone Detected": 8,
    "Tab Switched / Left Window": 6,
    "Exited Fullscreen": 4,
    "Face Missing from Frame": 3,
    "Face Partially Hidden": 2,
    "Looked Away from Screen": 2,
    "Eye Shifting / Rapid Eye Movement": 1,
}

def analyze_answer(session_id: str, candidate_answer: str) -> str:
    """
    Mock AI logic to analyze a candidate's answer and generate a response.
    In a real scenario, this would call OpenAI/LLM API.
    """
    try:
        # 1. Store candidate's answer
        memory_store.append_message(session_id, "user", candidate_answer)
        logger.info(f"Candidate answer recorded for session {session_id}")
        
        # 2. Mock AI response generation
        # In production, replace with actual LLM call
        ai_response = generate_mock_response(candidate_answer)
        
        # 3. Store AI's response
        memory_store.append_message(session_id, "ai", ai_response)
        logger.debug(f"AI response generated for session {session_id}")
        
        return ai_response
    except Exception as e:
        logger.error(f"Error analyzing answer: {e}")
        return "I encountered an error processing your answer. Please try again."

def generate_mock_response(candidate_answer: str) -> str:
    """Generate a mock AI response based on candidate answer."""
    responses = [
        "That's an interesting approach. Can you elaborate on how you would handle edge cases?",
        "Good thinking. How would you optimize this for performance?",
        "I see. Can you walk me through your problem-solving process?",
        "That makes sense. What challenges do you anticipate with this solution?",
        "Interesting perspective. How would you handle scalability?",
    ]
    
    # Return different responses based on answer length (simple mock)
    if len(candidate_answer) < 50:
        return responses[0]
    elif len(candidate_answer) < 100:
        return responses[1]
    elif len(candidate_answer) < 200:
        return responses[2]
    elif len(candidate_answer) < 300:
        return responses[3]
    else:
        return responses[4]

def generate_interview_report(session_id: str) -> Dict[str, Any]:
    """
    Generate a comprehensive interview report with improved scoring.
    """
    try:
        session = memory_store.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found for report generation")
            return {
                "status": "error",
                "message": "Session not found"
            }
        
        messages = session.get("messages", [])
        violations = session.get("violations", [])
        
        total_messages = len(messages)
        total_violations = len(violations)
        
        # Improved scoring algorithm
        final_score = calculate_score(violations)
        feedback = generate_feedback(final_score, total_violations, total_messages)
        
        # Generate violation details with categorization
        violation_summary = categorize_violations(violations)
        
        logger.info(f"Report generated for session {session_id}: score={final_score}, violations={total_violations}")
        
        return {
            "status": "success",
            "overallScore": final_score,
            "feedback": feedback,
            "violations": violations,
            "totalViolations": total_violations,
            "violationSummary": violation_summary,
            "totalMessages": total_messages,
            "sessionId": session_id,
            "generatedAt": memory_store.datetime.now().isoformat() if hasattr(memory_store, 'datetime') else ""
        }
    except Exception as e:
        logger.error(f"Error generating interview report: {e}")
        return {
            "status": "error",
            "message": f"Failed to generate report: {str(e)}"
        }

def calculate_score(violations: List[Dict[str, Any]]) -> int:
    """
    Calculate interview score based on violations with weighted penalty system.
    
    Scoring Logic:
    - Start with 100 points
    - Deduct points based on violation severity
    - Violations have different weights (more serious = higher penalty)
    - Final score is capped between 0-100
    """
    base_score = 100
    total_penalty = 0
    
    # Calculate weighted penalty
    for violation in violations:
        v_type = violation.get("type", "Unknown")
        weight = VIOLATION_WEIGHTS.get(v_type, 1)
        total_penalty += weight
        logger.debug(f"Violation: {v_type}, Weight: {weight}, Running Penalty: {total_penalty}")
    
    final_score = max(0, min(100, base_score - total_penalty))
    logger.info(f"Score calculated: base={base_score}, penalty={total_penalty}, final={final_score}")
    
    return final_score

def generate_feedback(score: int, total_violations: int, total_messages: int) -> str:
    """
    Generate personalized feedback based on score and violations.
    """
    feedback_parts = []
    
    # Overall performance feedback
    if score >= 90:
        feedback_parts.append("Excellent performance! You demonstrated strong focus and engagement throughout the interview.")
    elif score >= 80:
        feedback_parts.append("Good performance. You showed solid focus with minimal distractions during the interview.")
    elif score >= 70:
        feedback_parts.append("Satisfactory performance. There were some minor distractions, but overall you maintained reasonable focus.")
    elif score >= 60:
        feedback_parts.append("Fair performance. Multiple distractions were detected during the interview.")
    elif score >= 50:
        feedback_parts.append("Below average performance. Significant distractions affected your interview quality.")
    else:
        feedback_parts.append("Poor performance. Multiple serious violations were detected during the interview.")
    
    # Violation-specific feedback
    if total_violations > 0:
        feedback_parts.append(f"\nDuring the interview, {total_violations} proctoring violation(s) were recorded.")
        
        if total_violations > 3:
            feedback_parts.append("Please ensure a distraction-free environment for future interviews.")
    else:
        feedback_parts.append("No proctoring violations were detected. Excellent test-taking environment!")
    
    # Engagement feedback
    if total_messages > 5:
        feedback_parts.append(f"\nYou engaged well with {total_messages} messages exchanged during the interview.")
    
    return " ".join(feedback_parts)

def categorize_violations(violations: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Categorize violations by type and count occurrences.
    """
    violation_counts = {}
    
    for violation in violations:
        v_type = violation.get("type", "Unknown")
        violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
    
    logger.debug(f"Violation summary: {violation_counts}")
    return violation_counts