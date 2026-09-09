from __future__ import annotations
import json
import urllib.error
import urllib.request
import re
from typing import Any
from skillproof.models import Question, SkillCandidate, ScoredAssessment
from skillproof.report import evidence_status, gap_priority, gap_reason, learning_plan_rows

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_MODEL = "gemini-2.5-flash"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/auto"
DEFAULT_ENDPOINT = GEMINI_ENDPOINT
DEFAULT_MODEL = GEMINI_MODEL

def assessment_payload(scored: ScoredAssessment, learning_style: str, weekly_hours: int) -> dict[str, Any]:
    return {
        "overall_score": scored.overall_score,
        "recommendation": scored.recommendation,
        "skills": [
            {
                "name": r.name,
                "score": r.total_score,
                "level": r.level,
                "gap_reason": gap_reason(r),
                "evidence": r.evidence[:3]
            } for r in scored.skill_results
        ]
    }

def call_gemini_native(api_key: str, system_message: str, user_prompt: str, history: list[dict[str, str]] = None, model: str = None) -> str:
    # Convert history (role/content) to Gemini contents (role/parts)
    contents = []
    if history:
        for turn in history:
            role = "user" if turn["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    
    # Add the current prompt
    contents.append({"role": "user", "parts": [{"text": user_prompt}]})

    body = {
        "system_instruction": {"parts": [{"text": system_message}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048}
    }
    
    active_model = model or GEMINI_MODEL or "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        # Fallback to a very simple prompt if complex one fails
        raise RuntimeError(f"Gemini Error: {str(e)}")

def parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except: pass
    return {}

def generate_assessment_question(skill: SkillCandidate, question: Question, api_key: str, endpoint: str, model: str, seniority: str = "Mid-Level", prior_turns: list[dict[str, str]] = None) -> dict[str, str]:
    system = (
        f"You are a Technical Interviewer for a {seniority} role. "
        "STRICT RULE: Avoid repetitive patterns. Do NOT always start with 'Can you tell me...'. "
        "VARY YOUR STYLE: Mix between scenario-based questions, troubleshooting challenges, and architectural tradeoffs. "
        "Your goal is to verify technical claims with ONE deep-dive question. "
        "Return JSON: {\"question\": \"...\", \"interviewer_intent\": \"...\"}"
    )
    user = f"Seniority Level: {seniority}\nSkill: {skill.name}\nJD Mentions: {skill.jd_mentions[:2]}\nResume Claims: {skill.resume_evidence[:2]}"
    
    res = call_gemini_native(api_key, system, user, prior_turns, model=model)
    data = parse_json_object(res)
    return {"question": data.get("question", res), "interviewer_intent": data.get("interviewer_intent", f"Verify {seniority} level expertise")}

def generate_adaptive_follow_up(skill: SkillCandidate, question: Question, displayed_question: str, answer_text: str, api_key: str, endpoint: str, model: str, seniority: str = "Mid-Level") -> dict[str, str]:
    system = (
        f"You are a Senior Technical Architect auditing a candidate's answer for a {seniority} role. "
        "STRICT RULE: Be direct and conversational. No generic fluff. "
        "If the answer is vague: Challenge them on specific metrics or implementation steps. "
        "If the answer is strong: Drill into a specific 'What if?' scenario, a tradeoff, or a failure state they encountered. "
        "Return JSON: {\"response_feedback\": \"...\", \"follow_up\": \"...\"}"
    )
    user = f"Role Seniority: {seniority}\nSkill: {skill.name}\nQuestion Asked: {displayed_question}\nAnswer: {answer_text}"
    
    res = call_gemini_native(api_key, system, user, model=model)
    data = parse_json_object(res)
    return {"response_feedback": data.get("response_feedback", "Got it."), "follow_up": data.get("follow_up", res)}

def generate_personalized_learning_plan(scored: ScoredAssessment, learning_style: str, weekly_hours: int, api_key: str, endpoint: str, model: str) -> list[dict[str, Any]]:
    system = (
        "You are a Technical Mentor. Create a structured upskilling roadmap.\n"
        "STRICT RULES:\n"
        "1. GAP ANALYSIS: State which technical signal was missing in the interview.\n"
        "2. PROOF ARTIFACT: Define a technical project to prove mastery.\n"
        "3. ACCURACY: Base everything on the provided assessment data.\n"
        "Return JSON: {\"plans\": [{\"skill\": \"...\", \"priority\": \"...\", \"gap_analysis\": \"...\", \"target_level\": \"...\", \"estimated_hours\": \"...\", \"course_path\": [], \"proof_artifact\": \"...\", \"retest_prompt\": \"...\"}]}"
    )
    user = json.dumps(assessment_payload(scored, learning_style, weekly_hours))
    
    res = call_gemini_native(api_key, system, user, model=model)
    data = parse_json_object(res)
    plans = data.get("plans", [])
    for p in plans:
        p.setdefault("why_now", p.get("gap_analysis", "Critical gap identified"))
    return plans

def generate_ai_review(scored: ScoredAssessment, learning_style: str, weekly_hours: int, api_key: str, endpoint: str, model: str) -> str:
    system = (
        "You are a Technical Lead. Audit this candidate's assessment.\n"
        "1. IDENTIFY: Which resume claims were successfully proven?\n"
        "2. EXPOSE: Which claims were identified as risks/gaps?\n"
        "3. ROI: What is the business impact of hiring vs. upskilling this specific candidate?\n"
        "Write 2 professional paragraphs."
    )
    user = json.dumps(assessment_payload(scored, learning_style, weekly_hours))
    return call_gemini_native(api_key, system, user, model=model)

def generate_ai_answer_audit(skill: SkillCandidate, question: Question, displayed_question: str, answer_text: str, api_key: str, endpoint: str, model: str, seniority: str = "Mid-Level") -> dict[str, Any]:
    system = (
        f"You are a Technical Subject Matter Expert auditing a {seniority} level candidate's answer for the skill '{skill.name}'.\n"
        "Your task is to provide an objective technical score (0-100) based on accuracy, depth, and evidence.\n"
        "SCORING CRITERIA:\n"
        "1. ACCURACY (0-45): Is the technical explanation correct?\n"
        "2. DEPTH (0-25): Does the candidate show senior-level reasoning (tradeoffs, constraints)?\n"
        "3. CONFIDENCE (0-10): Is the answer concrete and evidence-backed?\n"
        "Return JSON: {\"assessment_score\": int, \"depth_score\": int, \"confidence_score\": int, \"reason_codes\": [\"code1\", \"code2\"]}"
    )
    user = (
        f"Skill: {skill.name}\n"
        f"Context (JD): {skill.jd_mentions[:1]}\n"
        f"Context (Resume): {skill.resume_evidence[:1]}\n"
        f"Question Asked: {displayed_question}\n"
        f"Candidate Answer: {answer_text}"
    )
    
    res = call_gemini_native(api_key, system, user, model=model)
    data = parse_json_object(res)
    return {
        "assessment_score": data.get("assessment_score", 0),
        "depth_score": data.get("depth_score", 0),
        "confidence_score": data.get("confidence_score", 0),
        "reason_codes": data.get("reason_codes", ["ai_audit_complete"])
    }

