from typing import List, Dict, Any
import memory_store

def analyze_answer(session_id: str, candidate_answer: str) -> str:
    """
    Mock AI logic to analyze a candidate's answer and generate a response.
    In a real scenario, this would call OpenAI/LLM API.
    """
    # 1. Store candidate's answer
    memory_store.append_message(session_id, "user", candidate_answer)
    
    # 2. Mock AI response generation
    ai_response = "That's an interesting approach. Can you elaborate on how you would handle edge cases?"
    
    # 3. Store AI's response
    memory_store.append_message(session_id, "ai", ai_response)
    
    return ai_response

def generate_interview_report(session_id: str) -> Dict[str, Any]:
    session = memory_store.get_session(session_id)
    history = session["messages"]
    violations = session["violations"]
    
    total_messages = len(history)
    total_violations = len(violations)
    
    base_score = 90
    penalty = total_violations * 5
    final_score = max(0, base_score - penalty)
    
    if final_score >= 80:
        feedback = "The candidate demonstrated a strong understanding of the topics discussed. The answers were well-structured and clear."
    elif final_score >= 60:
        feedback = "The candidate showed adequate knowledge, but there were some hesitations and areas lacking depth."
    else:
        feedback = "The candidate struggled with several core concepts and accumulated multiple proctoring violations during the session."
        
    if penalty > 0:
        feedback += f" Note: The score was penalized by {penalty} points due to {total_violations} proctoring violations."

    return {
        "overallScore": final_score,
        "feedback": feedback,
        "violations": violations,
        "totalViolations": total_violations
    }
