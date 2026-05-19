from __future__ import annotations

import re

from skillproof.extraction import evidence_quality, extract_candidates, normalize
from skillproof.models import Assessment, Question, ScoredAssessment, SkillCandidate, SkillResult


WEAK_PHRASES = (
    "i don't know",
    "not sure",
    "no experience",
    "never used",
    "just basic",
    "only theoretical",
)

DEPTH_SIGNALS = (
    "audit",
    "tradeoff",
    "edge case",
    "constraint",
    "latency",
    "cost",
    "scale",
    "security",
    "validation",
    "validated",
    "monitor",
    "debug",
    "debugging",
    "decision",
    "duplicate",
    "fallback",
    "grain",
    "impact",
    "metric",
    "baseline",
    "failure",
    "risk",
    "threshold",
)

CONTEXT_SIGNALS = (
    "artifact",
    "candidate",
    "component",
    "dashboard",
    "database",
    "endpoint",
    "fixture",
    "pipeline",
    "project",
    "query",
    "report",
    "resume",
    "service",
    "stakeholder",
    "table",
    "validation",
)

GENERIC_PHRASES = (
    "reliable output",
    "speed versus correctness",
    "document assumptions",
    "business impact to the stakeholder",
)

SPECIFICITY_RE = re.compile(r"(\d+%|\d+\s*(users|hours|mins|minutes|seconds|x|tables|apis|models|projects))", re.I)


def detect_seniority(jd_text: str) -> str:
    text = jd_text.lower()
    if any(w in text for w in ("senior", "architect", "lead", "principal", "staff", "manager", "head")):
        return "Senior"
    if any(w in text for w in ("junior", "fresher", "intern", "associate", "entry")):
        return "Junior"
    return "Mid-Level"


def build_assessment(jd_text: str, resume_text: str) -> Assessment:
    skills = extract_candidates(jd_text, resume_text)
    seniority = detect_seniority(jd_text)
    return Assessment(
        jd_text=jd_text, 
        resume_text=resume_text, 
        skills=skills, 
        seniority=seniority
    )


def answer_key(skill_name: str, prompt: str) -> str:
    return f"{skill_name}::{prompt}"


def follow_up_key(skill_name: str, prompt: str) -> str:
    return f"{answer_key(skill_name, prompt)}::adaptive_follow_up"


def short_context(text: str, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def contextual_question_prompt(skill: SkillCandidate, question: Question) -> str:
    jd_context = short_context(skill.jd_mentions[0]) if skill.jd_mentions else ""
    resume_context = short_context(skill.resume_evidence[0]) if skill.resume_evidence else ""
    skill_name = skill.name
    signal_text = ", ".join(question.signals[:3])

    if resume_context and jd_context:
        return (
            f"For {skill_name}, use this resume claim: \"{resume_context}\". "
            f"Explain the exact artifact or decision you owned, the {signal_text or skill_name} constraint you handled, "
            f"and the proof a reviewer could inspect to confirm it worked."
        )
    if resume_context:
        return (
            f"Your resume mentions {skill_name}: \"{resume_context}\". "
            f"Describe the implementation detail you owned, one tradeoff you made, "
            f"and the measurable or observable outcome."
        )
    if jd_context:
        return (
            f"The JD asks for {skill_name} in this context: \"{jd_context}\". "
            f"Describe the closest real task you have done, the first {skill_name} artifact you would build or analyze, "
            f"and the quality signal you would check in week one."
        )
    return (
        f"For {skill_name}, give me one real example that proves {signal_text}. "
        "What did you do, what went wrong or could have gone wrong, and how did you know it worked?"
    )


def contextual_follow_up_prompt(skill: SkillCandidate, question: Question, answer_text: str) -> str:
    cleaned = normalize(answer_text)
    answer_context = short_context(answer_text, 150)
    if not cleaned:
        return f"Give me one concrete {skill.name} example from a real project, internship, class project, or work task."

    missing_signals = [
        signal
        for signal in question.signals
        if signal.lower() not in cleaned and len(signal) > 2
    ][:2]
    has_metric = SPECIFICITY_RE.search(answer_text) is not None
    has_depth = any(signal in cleaned for signal in DEPTH_SIGNALS)
    has_action = any(
        word in cleaned
        for word in (
            "analyzed",
            "built",
            "debugged",
            "designed",
            "implemented",
            "measured",
            "optimized",
            "tested",
            "validated",
            "wrote",
        )
    )
    has_generic_phrase = any(phrase in cleaned for phrase in GENERIC_PHRASES)
    has_weak_phrase = any(phrase in cleaned for phrase in WEAK_PHRASES)

    if has_weak_phrase:
        return (
            f"What is the closest adjacent experience you have to {skill.name}, and what proof artifact "
            "could you build in one week to reduce this gap?"
        )
    if has_generic_phrase or len(cleaned.split()) < 22:
        return (
            f"You said: \"{answer_context}\". Give me one specific example with the tool, dataset, component, "
            "stakeholder, or deliverable you actually handled."
        )
    if not has_action:
        return (
            f"In that {skill.name} example, what did you personally implement, analyze, debug, test, or decide?"
        )
    if missing_signals:
        missing = " and ".join(missing_signals)
        return f"Where did {missing} show up in that work, and how did you handle it?"
    if not has_metric:
        return (
            "What metric, output, user impact, review result, or business signal showed that your work succeeded?"
        )
    if not has_depth:
        return (
            "What failed or almost failed, what tradeoff did you choose, and how did you validate the final decision?"
        )
    return (
        f"If I asked a hiring manager to verify this {skill.name} claim, what artifact would you show them?"
    )


def follow_up_prompt(
    skill_name: str,
    prompt: str,
    answer_text: str,
    signals: tuple[str, ...],
) -> str:
    cleaned = normalize(answer_text)
    if not cleaned:
        return f"Give me one concrete {skill_name} example from a real project, internship, class project, or work task."

    missing_signals = [
        signal
        for signal in signals
        if signal.lower() not in cleaned and len(signal) > 2
    ][:2]
    has_metric = SPECIFICITY_RE.search(answer_text) is not None
    has_depth = any(signal in cleaned for signal in DEPTH_SIGNALS)

    if missing_signals:
        missing = " and ".join(missing_signals)
        return f"Where did {missing} show up in that work, and how did you handle it?"
    if not has_metric:
        return (
            "What metric, output, user impact, review result, or business signal showed that your work succeeded?"
        )
    if not has_depth:
        return (
            "What failed or almost failed, what tradeoff did you choose, and how did you validate the final decision?"
        )
    return (
        f"If I asked a hiring manager to verify this {skill_name} claim, what artifact would you show them?"
    )


def answer_signature(text: str) -> str:
    cleaned = normalize(text)
    if len(cleaned.split()) < 18:
        return ""
    return re.sub(r"\d+", "#", cleaned)


def answer_reuse_count(text: str, answers: dict[str, str]) -> int:
    signature = answer_signature(text)
    if not signature:
        return 0
    return sum(1 for answer in answers.values() if answer_signature(answer) == signature)


def score_answer(text: str, signals: tuple[str, ...]) -> tuple[int, int, int, list[str]]:
    cleaned = normalize(text)
    if not cleaned:
        return 0, 0, 0, ["no_answer"]

    words = cleaned.split()
    reason_codes: list[str] = []
    # Length score: Fair baseline for 30+ words. Max 8 points.
    length_score = min(8, max(0, len(words) // 5))
    signal_hits = sum(1 for signal in signals if signal.lower() in cleaned)
    # Signal score: More flexible, rewards any 2 technical keywords significantly.
    signal_score = min(20, signal_hits * 10)
    # Technical depth bonus: If you use technical markers but miss exact signals
    depth_hits = sum(1 for signal in DEPTH_SIGNALS if signal in cleaned)
    if signal_hits == 0 and depth_hits >= 2:
        signal_score = 12
        reason_codes.append("uses_general_technical_language")

    specificity_score = min(10, len(SPECIFICITY_RE.findall(text)) * 5)
    # Practical score: High reward for action
    practical_score = 10 if any(
        word in cleaned
        for word in (
            "analyzed",
            "built",
            "compared",
            "debugged",
            "deployed",
            "designed",
            "explained",
            "fixed",
            "implemented",
            "measured",
            "modeled",
            "optimized",
            "presented",
            "profile",
            "profiled",
            "tested",
            "validated",
            "wrote",
        )
    ) else 0
    # Context score: High reward for artifacts/processes
    context_hits = sum(1 for signal in CONTEXT_SIGNALS if signal in cleaned)
    context_score = min(10, context_hits * 3)
    
    weak_penalty = 12 if any(phrase in cleaned for phrase in WEAK_PHRASES) else 0
    generic_penalty = 0
    if len(words) >= 50 and signal_hits <= 1 and context_hits <= 1:
        generic_penalty = 10

    assessment = max(
        0,
        min(
            45,
            length_score
            + signal_score
            + specificity_score
            + practical_score
            + context_score
            - weak_penalty
            - generic_penalty,
        ),
    )

    depth = min(
        25, # Increased max depth from 20 to 25
        depth_hits * 4
        + min(8, signal_hits * 3)
        + (4 if "because" in cleaned else 0)
        + (4 if practical_score else 0),
    )
    confidence = min(
        10,
        3 # Higher baseline confidence
        + min(5, signal_hits * 2)
        + (2 if specificity_score else 0)
        + (2 if practical_score else 0)
        - (6 if weak_penalty else 0),
    )
    confidence = max(0, confidence)

    if signal_hits >= 1:
        reason_codes.append("uses_skill_specific_language")
    if specificity_score:
        reason_codes.append("mentions_measurable_outcome")
    if practical_score:
        reason_codes.append("gives_practical_example")
    if context_hits >= 2:
        reason_codes.append("gives_contextual_detail")
    if depth >= 10:
        reason_codes.append("discusses_tradeoffs_or_failures")
    if weak_penalty:
        reason_codes.append("low_confidence_language")

    return assessment, depth, confidence, reason_codes


def classify(score: int) -> str:
    if score >= 80:
        return "Strong"
    if score >= 65:
        return "Ready with checks"
    if score >= 45:
        return "Developing"
    return "Gap"


def learning_plan(skill: SkillCandidate, level: str) -> str:
    if level == "Strong":
        return "No immediate learning plan needed. Use this skill as assessment evidence and ask one advanced calibration question."

    if level == "Ready with checks":
        time = "3-5 focused hours"
        priority = "Low"
    elif level == "Developing":
        time = "1-2 weeks"
        priority = "Medium"
    else:
        time = "2-4 weeks"
        priority = "High"

    adjacent = ", ".join(skill.adjacent_skills[:2]) if skill.adjacent_skills else "related fundamentals"
    resources = "; ".join(skill.resources[:2])
    return (
        f"Priority: {priority}. Estimated time to job-ready: {time}. "
        f"Start from adjacent strengths in {adjacent}. "
        f"Resources: {resources}. "
        f"Proof task: build a small role-relevant artifact demonstrating {skill.name} and document the tradeoffs."
    )


def score_skill(skill: SkillCandidate, answers: dict[str, str]) -> SkillResult:
    resume_score = evidence_quality(skill.resume_evidence)
    answer_scores: list[int] = []
    depth_scores: list[int] = []
    confidence_scores: list[int] = []
    reason_codes = []

    for question in skill.questions:
        key = answer_key(skill.name, question.prompt)
        answer_text = answers.get(key, "")
        follow_up_text = answers.get(follow_up_key(skill.name, question.prompt), "")
        combined_answer = "\n".join(part for part in (answer_text, follow_up_text) if part.strip())
        assessment, depth, confidence, reasons = score_answer(combined_answer, question.signals)
        if follow_up_text.strip():
            reasons.append("answered_adaptive_follow_up")
        if answer_reuse_count(combined_answer, answers) > 1:
            assessment = max(0, assessment - 10)
            depth = max(0, depth - 4)
            confidence = max(0, confidence - 3)
            reasons.append("reused_answer_pattern")
        answer_scores.append(assessment)
        depth_scores.append(depth)
        confidence_scores.append(confidence)
        reason_codes.extend(reasons)

    assessment_score = round(sum(answer_scores) / max(1, len(answer_scores)))
    depth_score = round(sum(depth_scores) / max(1, len(depth_scores)))
    confidence_score = round(sum(confidence_scores) / max(1, len(confidence_scores)))
    total = min(100, resume_score + assessment_score + depth_score + confidence_score)
    level = classify(total)

    if resume_score >= 18:
        reason_codes.append("strong_resume_evidence")
    elif resume_score == 0:
        reason_codes.append("missing_resume_evidence")
    else:
        reason_codes.append("partial_resume_evidence")

    risk_flags = []
    if skill.criticality == "High" and total < 70:
        risk_flags.append("Critical JD skill below readiness threshold")
    if resume_score == 0 and skill.criticality in {"High", "Medium"}:
        risk_flags.append("Required skill has no resume evidence")
    if assessment_score < 18:
        risk_flags.append("Assessment answers lack concrete detail")
    if "reused_answer_pattern" in reason_codes:
        risk_flags.append("Answers appear reused across skills")
    if "generic_answer_pattern" in reason_codes:
        risk_flags.append("Assessment answers may be too generic")

    return SkillResult(
        name=skill.name,
        category=skill.category,
        criticality=skill.criticality,
        total_score=total,
        level=level,
        resume_evidence_score=resume_score,
        assessment_score=assessment_score,
        depth_score=depth_score,
        confidence_score=confidence_score,
        reason_codes=sorted(set(reason_codes)),
        risk_flags=risk_flags,
        evidence=skill.resume_evidence[:3],
        learning_plan=learning_plan(skill, level),
        adjacent_skills=skill.adjacent_skills[:3],
        resources=skill.resources[:3],
    )


def recommendation(overall: int, results: list[SkillResult]) -> str:
    high_risks = [result for result in results if result.criticality == "High" and result.total_score < 70]
    if overall >= 82 and not high_risks:
        return "Candidate shows strong proficiency across the required skills. Use one advanced calibration question per core skill."
    if overall >= 60:
        return "Candidate has usable foundations but needs targeted upskilling on the flagged skill gaps."
    return "Candidate needs foundation work before being considered ready for this JD. Follow the learning plan and reassess."


def score_assessment(assessment: Assessment, answers: dict[str, str]) -> ScoredAssessment:
    results = [score_skill(skill, answers) for skill in assessment.skills]
    if not results:
        return ScoredAssessment([], 0, "No relevant skills found. Recheck input quality.", [], [])

    weighted_total = 0
    weight_sum = 0
    weights = {"High": 3, "Medium": 2, "Resume-only": 1}
    for result in results:
        weight = weights.get(result.criticality, 1)
        weighted_total += result.total_score * weight
        weight_sum += weight
    overall = round(weighted_total / weight_sum)

    strongest = [result.name for result in sorted(results, key=lambda item: item.total_score, reverse=True)[:3]]
    risks = []
    for result in results:
        risks.extend([f"{result.name}: {flag}" for flag in result.risk_flags])

    return ScoredAssessment(
        skill_results=results,
        overall_score=overall,
        recommendation=recommendation(overall, results),
        strongest_skills=strongest,
        highest_risks=risks[:5],
    )
