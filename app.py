from __future__ import annotations

import importlib
import os

import streamlit as st

from skillproof.ai_assist import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    GEMINI_ENDPOINT,
    GEMINI_MODEL,
    OPENROUTER_ENDPOINT,
    OPENROUTER_MODEL,
    generate_adaptive_follow_up,
    generate_ai_review,
    generate_assessment_question,
    generate_personalized_learning_plan,
)
import skillproof.assessment as assessment_engine
import skillproof.extraction as extraction_engine
import skillproof.file_readers as file_readers
import skillproof.report as report_engine
import skillproof.sample_data as sample_data


file_readers = importlib.reload(file_readers)
assessment_engine = importlib.reload(assessment_engine)
extraction_engine = importlib.reload(extraction_engine)
report_engine = importlib.reload(report_engine)
sample_data = importlib.reload(sample_data)
answer_key = assessment_engine.answer_key
build_assessment = assessment_engine.build_assessment
contextual_question_prompt = assessment_engine.contextual_question_prompt
contextual_follow_up_prompt = assessment_engine.contextual_follow_up_prompt
follow_up_key = assessment_engine.follow_up_key
follow_up_prompt = assessment_engine.follow_up_prompt
score_assessment = assessment_engine.score_assessment
accuracy_lift_estimate = report_engine.accuracy_lift_estimate
build_markdown_report = report_engine.build_markdown_report
evidence_status = report_engine.evidence_status
executive_summary = report_engine.executive_summary
gap_reason = report_engine.gap_reason
gap_priority = report_engine.gap_priority
learning_plan_rows = report_engine.learning_plan_rows
proof_ledger_rows = report_engine.proof_ledger_rows
roi_projection = report_engine.roi_projection
readiness_summary = report_engine.readiness_summary
skill_choice_reason = report_engine.skill_choice_reason
SAMPLE_ANSWERS = sample_data.SAMPLE_ANSWERS
SAMPLE_JD = sample_data.SAMPLE_JD
SAMPLE_RESUME = sample_data.SAMPLE_RESUME
UPLOAD_TYPES = file_readers.supported_upload_types()
FORMAT_LABEL = "PDF, DOCX, TXT/MD, CSV, or XLSX"
MAX_INTERVIEW_SKILLS = 5
QUESTIONS_PER_INTERVIEW_SKILL = 1
INTERVIEW_CRITICALITY_ORDER = {"High": 0, "Medium": 1, "Resume-only": 2}


st.set_page_config(
    page_title="SkillProof AI",
    page_icon="SP",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; }
    .hero {
        border: 1px solid #d0d7e2;
        background: #f8fafc;
        padding: 18px 20px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .hero h1 {
        margin: 0 0 6px 0;
        font-size: 2rem;
        line-height: 1.15;
    }
    .hero p {
        margin: 0;
        color: #475467;
        font-size: 1rem;
    }
    .metric-card {
        border: 1px solid #d0d7e2;
        border-radius: 8px;
        padding: 14px 16px;
        background: white;
        min-height: 104px;
    }
    .metric-card .label {
        color: #667085;
        font-size: .82rem;
        margin-bottom: 8px;
    }
    .metric-card .value {
        color: #101828;
        font-size: 1.45rem;
        font-weight: 720;
        margin-bottom: 5px;
    }
    .metric-card .note {
        color: #475467;
        font-size: .86rem;
    }
    .section-note {
        color: #475467;
        font-size: .92rem;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_state() -> None:
    defaults = {
        "jd_text": "",
        "resume_text": "",
        "assessment": None,
        "answers": {},
        "scored": None,
        "ai_review": "",
        "chat_messages": [],
        "conversation_started": False,
        "conversation_complete": False,
        "conversation_skill_index": 0,
        "conversation_question_index": 0,
        "conversation_waiting_follow_up": False,
        "learning_style": "Project-based",
        "weekly_hours": 6,
        "candidates": 60,
        "manual_minutes": 55,
        "agent_minutes": 14,
        "hourly_cost": 1000,
        "llm_question_cache": {},
        "llm_follow_up_cache": {},
        "llm_question_notes": {},
        "ai_learning_plan": [],
        "ai_learning_plan_error": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_conversation() -> None:
    st.session_state.chat_messages = []
    st.session_state.conversation_started = False
    st.session_state.conversation_complete = False
    st.session_state.conversation_skill_index = 0
    st.session_state.conversation_question_index = 0
    st.session_state.conversation_waiting_follow_up = False
    st.session_state.llm_question_cache = {}
    st.session_state.llm_follow_up_cache = {}
    st.session_state.llm_question_notes = {}


def limit_assessment_for_interview() -> None:
    assessment = st.session_state.assessment
    if not assessment:
        return

    def rank_skill(skill) -> tuple[int, int, int, int, str]:
        return (
            INTERVIEW_CRITICALITY_ORDER.get(skill.criticality, 3),
            0 if skill.jd_mentions else 1,
            -len(skill.resume_evidence),
            0 if skill.category != "Role-specific skill" else 1,
            skill.name.lower(),
        )

    ordered_skills = sorted(assessment.skills, key=rank_skill)
    core_skills = [skill for skill in ordered_skills if skill.criticality in {"High", "Medium"}]
    fallback_skills = [skill for skill in ordered_skills if skill.criticality not in {"High", "Medium"}]
    selected_skills = (core_skills + fallback_skills)[:MAX_INTERVIEW_SKILLS]

    for skill in selected_skills:
        skill.questions = skill.questions[:QUESTIONS_PER_INTERVIEW_SKILL]

    st.session_state.assessment.skills = selected_skills


def load_sample() -> None:
    st.session_state.jd_text = SAMPLE_JD
    st.session_state.resume_text = SAMPLE_RESUME
    st.session_state.answers = SAMPLE_ANSWERS.copy()
    st.session_state.assessment = build_assessment(SAMPLE_JD, SAMPLE_RESUME)
    limit_assessment_for_interview()
    st.session_state.scored = None
    clear_conversation()


def run_sample() -> None:
    load_sample()
    st.session_state.scored = score_assessment(
        st.session_state.assessment,
        st.session_state.answers,
    )


def analyze_inputs() -> None:
    api_key, _, _ = gemini_ai_config()
    if api_key:
        with st.spinner("🔍 Analyzing Skills with Gemini..."):
             skills = extraction_engine.extract_candidates_ai(
                st.session_state.jd_text,
                st.session_state.resume_text,
                api_key
            )
             # Wrap the results in a proper Assessment object
             from skillproof.models import Assessment
             st.session_state.assessment = Assessment(
                 jd_text=st.session_state.jd_text,
                 resume_text=st.session_state.resume_text,
                 skills=skills,
                 seniority=assessment_engine.detect_seniority(st.session_state.jd_text)
             )
    else:
        st.session_state.assessment = build_assessment(
            st.session_state.jd_text,
            st.session_state.resume_text,
        )
    
    limit_assessment_for_interview()
    st.session_state.scored = None
    st.session_state.answers = {}
    clear_conversation()


def score_current_assessment() -> None:
    st.session_state.scored = score_assessment(
        st.session_state.assessment,
        st.session_state.answers,
    )
    st.session_state.ai_review = ""
    st.session_state.ai_learning_plan = []
    st.session_state.ai_learning_plan_error = ""


def secret_value(*names: str) -> str:
    for name in names:
        try:
            value = st.secrets.get(name, "")
        except Exception:
            value = ""
        if value:
            return str(value)
        value = os.getenv(name, "")
        if value:
            return value
    return ""


def apply_upload(target_key: str, uploaded_file) -> None:
    if uploaded_file is None:
        return
    text = file_readers.read_uploaded_file(uploaded_file)
    if text:
        st.session_state[target_key] = text
        st.caption(f"Parsed {len(text):,} characters from {uploaded_file.name}.")
    else:
        st.warning(f"Could not read {uploaded_file.name}. Please paste the text manually.")


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def skill_matrix_rows() -> list[dict[str, str | int]]:
    assessment = st.session_state.assessment
    if not assessment:
        return []
    return [
        {
            "Skill": skill.name,
            "Category": skill.category,
            "JD priority": skill.criticality,
            "Resume evidence snippets": len(skill.resume_evidence),
            "Questions": len(skill.questions),
            "Why chosen": (
                "Required or role-relevant skill from the JD"
                if skill.criticality in {"High", "Medium"}
                else "Resume skill that may support adjacent learning"
            ),
        }
        for skill in assessment.skills
    ]


def current_conversation_item():
    assessment = st.session_state.assessment
    if not assessment or st.session_state.conversation_skill_index >= len(assessment.skills):
        return None, None
    skill = assessment.skills[st.session_state.conversation_skill_index]
    if st.session_state.conversation_question_index >= len(skill.questions):
        return None, None
    return skill, skill.questions[st.session_state.conversation_question_index]


def add_assistant_message(content: str) -> None:
    st.session_state.chat_messages.append({"role": "assistant", "content": content})


def add_user_message(content: str) -> None:
    st.session_state.chat_messages.append({"role": "user", "content": content})


def question_cache_key(skill, question) -> str:
    return answer_key(skill.name, question.prompt)


def gemini_ai_config() -> tuple[str, str, str]:
    return secret_value("GEMINI_API_KEY", "GOOGLE_API_KEY"), GEMINI_ENDPOINT, GEMINI_MODEL


    return f"AI mode: Gemini active ({model})"


def recent_interview_turns(limit: int = 8) -> list[dict[str, str]]:
    turns = []
    for message in st.session_state.chat_messages[-limit:]:
        turns.append(
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", ""))[:900],
            }
        )
    return turns


def get_display_question(skill, question) -> str:
    # DISABLE CACHE: Force a fresh, unique question from Gemini every time
    fallback = contextual_question_prompt(skill, question)
    api_key, endpoint, model = gemini_ai_config()
    
    if not api_key:
        st.session_state.llm_question_notes[answer_key(skill.name, question.prompt)] = "Gemini not configured"
        return fallback

    seniority = st.session_state.assessment.seniority if st.session_state.assessment else "Mid-Level"

    try:
        generated = generate_assessment_question(
            skill,
            question,
            api_key,
            endpoint,
            model,
            seniority,
            recent_interview_turns(),
        )
        return generated["question"]
    except Exception as error:
        return fallback


def get_display_follow_up(skill, question, answer_text: str) -> str:
    key = f"{question_cache_key(skill, question)}::{len(answer_text)}::{hash(answer_text)}"
    if key in st.session_state.llm_follow_up_cache:
        return st.session_state.llm_follow_up_cache[key]

    displayed_question = get_display_question(skill, question)
    fallback = contextual_follow_up_prompt(skill, question, answer_text)

    api_key, endpoint, model = gemini_ai_config()
    if not api_key:
        return fallback

    seniority = st.session_state.assessment.seniority if st.session_state.assessment else "Mid-Level"

    try:
        generated = generate_adaptive_follow_up(
            skill,
            question,
            displayed_question,
            answer_text,
            api_key,
            endpoint,
            model,
            seniority
        )
        feedback = generated.get("response_feedback", "").strip()
        follow_up = generated["follow_up"].strip()
        message = f"{feedback}\n\n{follow_up}" if feedback else follow_up
        st.session_state.llm_follow_up_cache[key] = message
        return message
    except Exception:
        return f"Thanks. I want to make that answer easier to verify.\n\n{fallback}"


def main_question_message(skill, question) -> str:
    evidence = skill.resume_evidence[0] if skill.resume_evidence else "No direct resume example found yet."
    # Global step number
    current_step = st.session_state.conversation_skill_index + 1
    total_steps = len(st.session_state.assessment.skills)
    
    displayed_question = get_display_question(skill, question)
    seniority = st.session_state.assessment.seniority if st.session_state.assessment else "Mid-Level"
    
    return (
        f"### Question {current_step} of {total_steps} ({seniority} Level)\n"
        f"**Verifying Skill:** `{skill.name}`\n"
        f"**Resume Claim:** \"_{evidence}_\"\n\n"
        f"**Architect Scenario:**\n"
        f"{displayed_question}"
    )


def conversation_progress() -> str:
    if not st.session_state.assessment:
        return "No assessment in progress"
    if st.session_state.conversation_complete:
        return "✅ Verification Complete"
    
    current = st.session_state.conversation_skill_index + 1
    total = len(st.session_state.assessment.skills)
    return f"Current Progress: Question {current} of {total} (Focusing on top technical skills)"


def start_live_conversation() -> None:
    clear_conversation()
    if not st.session_state.assessment:
        return
    api_key, _, _ = gemini_ai_config()
    if not api_key:
        return

    limit_assessment_for_interview()
    st.session_state.conversation_started = True
    skill, question = current_conversation_item()
    if skill and question:
        add_assistant_message(main_question_message(skill, question))


def move_to_next_question() -> None:
    assessment = st.session_state.assessment
    if not assessment:
        return

    # Force move to next skill (1 question per skill)
    st.session_state.conversation_skill_index += 1
    st.session_state.conversation_question_index = 0

    if st.session_state.conversation_skill_index >= len(assessment.skills):
        st.session_state.conversation_complete = True
        add_assistant_message(
            "The technical assessment is complete. Click **Score conversation** to see your verified results and ROI analysis."
        )
        return

    skill, question = current_conversation_item()
    if skill and question:
        add_assistant_message(main_question_message(skill, question))


def handle_chat_reply(reply: str) -> None:
    reply = reply.strip()
    if not reply or st.session_state.conversation_complete:
        return
    skill, question = current_conversation_item()
    if not skill or not question:
        return

    add_user_message(reply)
    
    # Record answer and move to next skill immediately
    st.session_state.answers[answer_key(skill.name, question.prompt)] = reply
    move_to_next_question()


def render_skill_conversation() -> None:
    st.subheader("🛡️ Skill Assessment")
    assessment = st.session_state.assessment
    if not assessment:
        st.info("Extract required skills from the Inputs tab first.")
        return

    limit_assessment_for_interview()
    assessment = st.session_state.assessment

    st.info(f"⚡ **Current Protocol:** {conversation_progress()}. Answer with technical proof to unlock your validated roadmap.")

    st.markdown(
        '<div class="section-note">SkillProof interviews one skill at a time. Questions are generated based on the JD, resume evidence, and previous answers.</div>',
        unsafe_allow_html=True,
    )
    live_api_key, live_endpoint, live_model = gemini_ai_config()
    gemini_ready = bool(live_api_key)
    with st.expander("Question engine", expanded=False):
        st.write("Engine: Gemini")

    st.caption(conversation_progress())
    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
    with col_a:
        if st.button("Start live assessment", type="primary", disabled=not gemini_ready):
            start_live_conversation()
            st.rerun()
    with col_b:
        if st.button("Skip to next question"):
            move_to_next_question()
            st.rerun()
    with col_c:
        if st.button("Restart chat"):
            start_live_conversation()
            st.rerun()
    with col_d:
        score_disabled = not st.session_state.conversation_started
        if st.button("Score conversation", type="primary", disabled=score_disabled):
            score_current_assessment()
            st.success("Assessment scored. Reports are unlocked.")

    if not st.session_state.conversation_started:
        st.info("Click **Start live assessment** to begin the back-and-forth interview.")
    else:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state.conversation_complete:
        st.success("Conversation complete. Score it to generate the report.")

    reply = st.chat_input("Reply to SkillProof AI...", disabled=not gemini_ready)
    if reply:
        if not st.session_state.conversation_started:
            start_live_conversation()
        handle_chat_reply(reply)
        st.rerun()

    with st.expander("Extracted skills and evidence", expanded=False):
        st.dataframe(skill_matrix_rows(), hide_index=True, use_container_width=True)


def render_ai_learning_plan(plans: list[dict]) -> None:
    for plan in plans:
        title = plan.get("skill") or "Skill"
        priority = plan.get("priority") or "Priority"
        hours = plan.get("estimated_hours") or "time estimate"
        
        # High-impact card UI
        st.markdown(f"""
        <div style="border-left: 5px solid #2563eb; padding: 15px; background: #f1f5f9; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
            <h3 style="margin:0; color:#1e3a8a;">🚀 {title} Roadmap</h3>
            <p style="margin:5px 0; font-weight:bold; color:#1e40af;">{priority} Priority | Estimated: {hours}</p>
            <div style="background:white; padding:15px; border-radius:5px; margin-top:10px; border: 1px solid #e2e8f0;">
                <p style="margin-bottom:8px; color:#b91c1c;"><b>⚠️ Gap Analysis:</b> {plan.get('gap_analysis', 'Technical verification required.')}</p>
                <p style="margin-bottom:8px;"><b>🎯 Target Level:</b> {plan.get('target_level')}</p>
                <hr style="margin:15px 0; border:0; border-top: 1px solid #e2e8f0;">
                <p style="margin-bottom:12px;"><b>🛠️ Proof Artifact (Mandatory Project):</b> <br><code style="color:#d97706; font-size:1.1em; background:#fff7ed; padding:4px 8px; border-radius:4px; display:block; margin-top:5px; border:1px dashed #fed7aa;">{plan.get('proof_artifact')}</code></p>
                <p style="margin-bottom:8px;"><b>📚 Recommended Resources:</b><br>{' • '.join(plan.get('course_path', []))}</p>
                <div style="background:#f8fafc; padding:10px; border-radius:4px; border: 1px solid #f1f5f9;">
                    <p style="margin:0; font-size:0.9em; color:#64748b;"><b>📝 Final Verification Challenge:</b><br><i>"{plan.get('retest_prompt')}"</i></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


ensure_state()

with st.sidebar:
    st.markdown("### 🤖 Engine Status")
    api_key, _, model = gemini_ai_config()
    if api_key:
        st.success(f"● Gemini Active")
        st.caption(f"Model: {model}")
        st.info("The AI is dynamically analyzing your resume to generate unique questions.")
    else:
        st.warning("○ Static Fallback")
        st.caption("No API Key detected.")
        st.info("Add GEMINI_API_KEY to see the AI generate custom questions.")
    st.markdown("---")

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ SkillProof AI: Claim-to-Proof Agent</h1>
        <p>Assess real proficiency from a JD and resume, identify gaps, and generate a personalized <b>Claim-to-Proof</b> learning roadmap.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

scored = st.session_state.scored
summary = readiness_summary(scored) if scored else None
learning_style = st.session_state.learning_style
weekly_hours = st.session_state.weekly_hours
roi = roi_projection(
    st.session_state.candidates,
    st.session_state.manual_minutes,
    st.session_state.agent_minutes,
    st.session_state.hourly_cost,
)

tabs = st.tabs(
    [
        "Inputs",
        "Live Assessment",
        "Personalized Learning Plan",
        "Gap Analysis",
        "Claim-to-Proof Ledger",
        "ROI Dashboard",
        "Export",
        "Executive Summary",
    ]
)

with tabs[0]:
    st.subheader("Inputs")
    st.markdown(
        f'<div class="section-note">Upload {FORMAT_LABEL} files or paste text manually. Spreadsheets are flattened with row and column labels so ATS exports and skill matrices still become assessment evidence.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Job Description**")
        jd_file = st.file_uploader(f"Upload JD as {FORMAT_LABEL}", type=UPLOAD_TYPES, key="jd_upload")
        apply_upload("jd_text", jd_file)
        st.session_state.jd_text = st.text_area(
            "Paste or edit job description",
            value=st.session_state.jd_text,
            height=340,
        )
    with right:
        st.markdown("**Candidate Resume**")
        resume_file = st.file_uploader(f"Upload resume as {FORMAT_LABEL}", type=UPLOAD_TYPES, key="resume_upload")
        apply_upload("resume_text", resume_file)
        st.session_state.resume_text = st.text_area(
            "Paste or edit candidate resume",
            value=st.session_state.resume_text,
            height=340,
        )

    if st.button("Extract required skills", type="primary"):
        if not st.session_state.jd_text.strip() or not st.session_state.resume_text.strip():
            st.warning("Paste both the JD and resume first.")
        else:
            analyze_inputs()
            score_current_assessment() # Initial scoring based on resume evidence
            st.success("Skills extracted! Reports and Personalized Learning Plan are now available. You can also start the Live Assessment for a deeper verification.")

    with st.expander("Quick demo controls", expanded=False):
        st.button("Load sample inputs", use_container_width=True, on_click=load_sample)
        st.button("Run sample end-to-end", use_container_width=True, on_click=run_sample)

    with st.expander("Why this is not a toy workflow", expanded=False):
        st.dataframe(
            [
                {
                    "Business workflow": "Recruiter receives JD/resume documents",
                    "Covered by": "PDF, DOCX, TXT/MD upload or paste",
                    "Impact": "Less manual copying before assessment",
                },
                {
                    "Business workflow": "ATS or spreadsheet export",
                    "Covered by": "CSV/XLSX ingestion with row and column context",
                    "Impact": "Batch-style evidence can be converted into auditable text",
                },
                {
                    "Business workflow": "Manager needs proof, not claims",
                    "Covered by": "Question answers, evidence snippets, score breakdown, reason codes",
                    "Impact": "Higher accuracy and fewer unstructured review cycles",
                },
            ],
            hide_index=True,
            use_container_width=True,
        )

    if st.session_state.assessment:
        st.markdown("**Extracted skill matrix**")
        st.dataframe(skill_matrix_rows(), hide_index=True, use_container_width=True)

with tabs[1]:
    render_skill_conversation()

with tabs[3]:
    st.subheader("Gap Analysis")
    scored = st.session_state.get("scored")
    if not scored:
        st.info("Extract skills in the Inputs tab to see the Gap Analysis.")
    else:
        if not st.session_state.conversation_complete:
            st.warning("⚠️ This analysis is based on Resume Claims only. Complete the Live Assessment for verified proof.")
        rows = []
        for result in scored.skill_results:
            rows.append(
                {
                    "Skill": result.name,
                    "Score": result.total_score,
                    "Level": result.level,
                    "JD priority": result.criticality,
                    "Evidence": evidence_status(result),
                    "Gap priority": gap_priority(result),
                    "Why chosen": skill_choice_reason(result),
                    "Why gap was detected": gap_reason(result),
                    "Notes": "; ".join(result.risk_flags) if result.risk_flags else "No major gap",
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)

        st.markdown("**Score breakdown**")
        for result in scored.skill_results:
            with st.expander(f"{result.name}: {result.total_score}/100 ({result.level})"):
                st.write("Resume evidence:", f"{result.resume_evidence_score}/25")
                st.write("Answer quality:", f"{result.assessment_score}/45")
                st.write("Practical depth:", f"{result.depth_score}/20")
                st.write("Confidence:", f"{result.confidence_score}/10")
                st.write("Reason codes:", ", ".join(result.reason_codes))
                st.write("Why this skill was chosen:", skill_choice_reason(result))
                st.write("Why this gap was detected:", gap_reason(result))

with tabs[4]:
    st.subheader("🛡️ Claim-to-Proof Ledger")
    scored = st.session_state.get("scored")
    if not scored:
        st.info("Extract skills in the Inputs tab to see the Claim-to-Proof Ledger.")
    else:
        st.markdown(
            '<div class="section-note">Audit trail: Analyzing the gap between resume claims and verified technical signals.</div>',
            unsafe_allow_html=True,
        )
        
        api_key, _, _ = gemini_ai_config()
        if api_key:
            with st.spinner("🕵️ Running Audit with Gemini..."):
                try:
                    audit = generate_ai_review(scored, st.session_state.learning_style, st.session_state.weekly_hours, api_key, GEMINI_ENDPOINT, GEMINI_MODEL)
                    st.markdown("### 📝 Auditor Verdict")
                    st.info(audit)
                except:
                    st.warning("Summary generation delayed. Showing granular ledger.")

        st.divider()
        ledger = proof_ledger_rows(scored)
        for row in ledger:
            status_color = "green" if "Verified" in row['Audit status'] or "backed" in row['Audit status'] else "orange"
            with st.expander(f"Skill: {row['Skill claim']} — {row['Audit status']}"):
                st.write(f"**Verification Status:** :{status_color}[{row['Audit status']}]")
                st.markdown(f"**Resume Claim:** \n> {row['Resume proof']}")
                st.markdown(f"**Assessment Proof:** \n {row['Assessment proof']}")
                st.markdown(f"**Verification Protocol:** \n {row['Proof task']}")

with tabs[2]:
    st.subheader("Personalized Learning Plan")
    scored = st.session_state.get("scored")
    if not scored:
        st.info("Extract skills in the Inputs tab to see the Learning Plan.")
    else:
        if not st.session_state.conversation_complete:
            st.warning("⚠️ This roadmap is based on Resume Claims. Complete the Live Assessment to refine your gaps.")
        
        # AUTO-TRIGGER GEMINI FOR LEARNING PLAN
        api_key, endpoint, model = gemini_ai_config()
        if api_key and not st.session_state.ai_learning_plan and not st.session_state.ai_learning_plan_error:
            with st.spinner("🧠 Gemini is architecting your personalized roadmap..."):
                try:
                    st.session_state.ai_learning_plan = generate_personalized_learning_plan(
                        scored,
                        st.session_state.learning_style,
                        st.session_state.weekly_hours,
                        api_key,
                        endpoint,
                        model,
                    )
                except Exception as e:
                    st.session_state.ai_learning_plan_error = str(e)

        st.markdown(
            '<div class="section-note">Turns assessment gaps into a practical roadmap: adjacent skills, courses, weekly effort, drills, and proof artifacts.</div>',
            unsafe_allow_html=True,
        )
        settings_left, settings_right, ai_plan_col = st.columns([1, 1, 1.2])
        with settings_left:
            learning_style = st.selectbox(
                "Preferred learning style",
                ["Project-based", "Video-first", "Docs-first", "Practice drills"],
                index=["Project-based", "Video-first", "Docs-first", "Practice drills"].index(st.session_state.learning_style),
                key="learning_style_control",
            )
            st.session_state.learning_style = learning_style
        with settings_right:
            weekly_hours = st.slider(
                "Available learning hours / week",
                2,
                20,
                value=int(st.session_state.weekly_hours),
                step=1,
                key="weekly_hours_control",
            )
            st.session_state.weekly_hours = weekly_hours
        with ai_plan_col:
            if st.button(
                "Generate AI-personalized roadmap",
                type="primary",
                use_container_width=True,
            ):
                api_key, endpoint, model = gemini_ai_config()
                if not api_key:
                    st.session_state.ai_learning_plan_error = (
                        "Add GEMINI_API_KEY or GOOGLE_API_KEY in Streamlit secrets to generate the AI-personalized roadmap."
                    )
                    st.session_state.ai_learning_plan = []
                else:
                    try:
                        st.session_state.ai_learning_plan = generate_personalized_learning_plan(
                            scored,
                            learning_style,
                            weekly_hours,
                            api_key,
                            endpoint,
                            model,
                        )
                        st.session_state.ai_learning_plan_error = ""
                    except RuntimeError as error:
                        st.session_state.ai_learning_plan = []
                        st.session_state.ai_learning_plan_error = str(error)

        st.markdown(f"Learning style: **{learning_style}** | Available time: **{weekly_hours} hrs/week**")

        if st.session_state.ai_learning_plan_error:
            st.warning(st.session_state.ai_learning_plan_error)
        if st.session_state.ai_learning_plan:
            st.markdown("**AI-personalized roadmap**")
            render_ai_learning_plan(st.session_state.ai_learning_plan)
            st.markdown("**Rubric-backed local plan**")

        plans = learning_plan_rows(scored, learning_style, weekly_hours)
        if not plans:
            st.success("No major learning gaps found.")
        for row in plans:
            with st.expander(
                f"{row['Skill']} | {row['Priority']} priority | {row['Timeline']}",
                expanded=not bool(st.session_state.ai_learning_plan),
            ):
                st.write(row["Goal"])
                st.write(row["Why this gap"])
                st.write(
                    "Adjacent skills:",
                    ", ".join(row["Adjacent skills"]) if row["Adjacent skills"] else "related fundamentals",
                )
                st.write("Learning style adaptation:", row["Style tip"])
                st.markdown("**Recommended course path**")
                for resource in row["Course path"]:
                    st.write(f"- {resource}")
                st.markdown("**Practice drill**")
                st.write(row["Practice drill"])
                st.markdown("**5-day sprint**")
                for step in row["Sprint plan"]:
                    st.write(f"- {step}")
                st.markdown("**Proof artifact**")
                st.write(row["Proof artifact"])
                st.write(row["Plan"])

with tabs[5]:
    st.subheader("ROI Dashboard")
    st.markdown(
        '<div class="section-note">Estimated business impact from replacing manual claim-checking with structured skill assessment.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("ROI assumptions", expanded=False):
        roi_a, roi_b, roi_c, roi_d = st.columns(4)
        with roi_a:
            st.session_state.candidates = st.slider(
                "Candidates / month",
                10,
                500,
                value=int(st.session_state.candidates),
                step=10,
                key="candidates_control",
            )
        with roi_b:
            st.session_state.manual_minutes = st.slider(
                "Manual minutes",
                20,
                90,
                value=int(st.session_state.manual_minutes),
                step=5,
                key="manual_minutes_control",
            )
        with roi_c:
            st.session_state.agent_minutes = st.slider(
                "SkillProof minutes",
                5,
                30,
                value=int(st.session_state.agent_minutes),
                step=1,
                key="agent_minutes_control",
            )
        with roi_d:
            st.session_state.hourly_cost = st.slider(
                "Reviewer cost / hour (INR)",
                300,
                3000,
                value=int(st.session_state.hourly_cost),
                step=100,
                key="hourly_cost_control",
            )
    roi = roi_projection(
        st.session_state.candidates,
        st.session_state.manual_minutes,
        st.session_state.agent_minutes,
        st.session_state.hourly_cost,
    )
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        metric_card("Cost saved", f"INR {roi['monthly_cost_saved']:,}", "Estimated monthly review cost saved.")
    with r2:
        metric_card("Throughput gain", f"{roi['throughput_gain']}x", "More candidates assessed in the same time.")
    with r3:
        metric_card("Time saved", f"{roi['monthly_hours_saved']} hrs", "Estimated monthly hours saved.")
    with r4:
        metric_card("Accuracy improved", f"+{accuracy_lift_estimate(scored)}%", "Estimated lift from evidence and answer coverage.")

    st.dataframe(
        [
            {
                "Metric": "Manual process",
                "Value": f"{st.session_state.manual_minutes} minutes per candidate",
                "Meaning": "Resume claims checked manually.",
            },
            {
                "Metric": "SkillProof process",
                "Value": f"{st.session_state.agent_minutes} minutes per candidate",
                "Meaning": "Structured assessment plus explainable scoring.",
            },
            {
                "Metric": "Monthly cost saved",
                "Value": f"INR {roi['monthly_cost_saved']:,}",
                "Meaning": "Based on candidate volume, time saved, and hourly cost assumptions.",
            },
            {
                "Metric": "Estimated accuracy improved",
                "Value": f"+{accuracy_lift_estimate(scored)}%",
                "Meaning": "Heuristic lift from evidence coverage, answers, and reason codes.",
            },
        ],
        hide_index=True,
        use_container_width=True,
    )

with tabs[6]:
    st.subheader("Export")
    scored = st.session_state.get("scored")
    if not scored:
        st.info("Score the assessment first.")
    else:
        report = build_markdown_report(scored)
        st.markdown("**Report preview**")
        st.markdown(report)
        st.download_button(
            "Download assessment report",
            data=report,
            file_name="skillproof-assessment-report.md",
            mime="text/markdown",
        )

        with st.expander("Optional AI reviewer"):
            st.caption(
                "The app works without an API key. This optional reviewer sends only score summaries, reason codes, and learning-plan rows to Gemini/OpenRouter for calibration notes."
            )
            provider = st.selectbox(
                "Provider",
                ["Gemini", "OpenRouter", "Custom OpenAI-compatible"],
                key="export_provider_select"
            )
            if provider == "Gemini":
                default_endpoint = GEMINI_ENDPOINT
                default_model = GEMINI_MODEL
                key_names = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
            elif provider == "OpenRouter":
                default_endpoint = OPENROUTER_ENDPOINT
                default_model = OPENROUTER_MODEL
                key_names = ("OPENROUTER_API_KEY",)
            else:
                default_endpoint = DEFAULT_ENDPOINT
                default_model = DEFAULT_MODEL
                key_names = ("OPENAI_COMPATIBLE_API_KEY",)

            api_key = st.text_input(
                "API key",
                type="password",
                value="",
                placeholder="Optional; not stored in the repo",
            )
            endpoint = st.text_input(
                "Endpoint",
                value=secret_value("OPENAI_COMPATIBLE_URL", "OPENROUTER_URL", "GEMINI_URL") or default_endpoint,
            )
            model = st.text_input(
                "Model",
                value=secret_value("OPENAI_COMPATIBLE_MODEL", "OPENROUTER_MODEL", "GEMINI_MODEL") or default_model,
            )
            if st.button("Generate AI reviewer notes"):
                resolved_key = api_key or secret_value(*key_names, "OPENAI_COMPATIBLE_API_KEY")
                if not resolved_key:
                    st.warning("Add an API key here or in Streamlit secrets to use the optional AI reviewer.")
                else:
                    try:
                        st.session_state.ai_review = generate_ai_review(
                            scored,
                            learning_style,
                            weekly_hours,
                            resolved_key,
                            endpoint,
                            model,
                        )
                    except RuntimeError as error:
                        st.error(str(error))

            if st.session_state.ai_review:
                st.markdown("**AI reviewer notes**")
                st.markdown(st.session_state.ai_review)

with tabs[7]:
    st.subheader("Executive Summary")
    scored = st.session_state.get("scored")
    if not scored:
        st.info("Extract skills in the Inputs tab to see the Executive Summary.")
    else:
        # Recalculate summary locally for safety
        summary = readiness_summary(scored)
        if not summary:
            st.error("Could not generate readiness summary. Please try extracting skills again.")
        else:
            st.markdown(
                '<div class="section-note">Judge-facing snapshot: decision, proof coverage, main risk, next action, and business ROI in one view.</div>',
                unsafe_allow_html=True,
            )
            
            top_cols = st.columns(4)
            with top_cols[0]:
                metric_card("Readiness status", str(summary.get("status", "Unknown")), "Final assessment status.")
            with top_cols[1]:
                metric_card("Overall score", f"{scored.overall_score}%", "Weighted by JD priority.")
            with top_cols[2]:
                metric_card("Proof-backed skills", str(summary.get("proof_backed_skills", 0)), "Verified evidence exists.")
            with top_cols[3]:
                metric_card("Skill gaps", str(summary.get("gap_count", 0)), "Below readiness threshold.")

        executive = executive_summary(scored)
        st.dataframe(
            [
                {"Signal": label, "What it means": value}
                for label, value in executive.items()
            ]
            + [
                {
                    "Signal": "Estimated monthly cost saved",
                    "What it means": f"INR {roi['monthly_cost_saved']:,} using current assumptions.",
                },
                {
                    "Signal": "Estimated throughput gain",
                    "What it means": f"{roi['throughput_gain']}x more candidates assessed.",
                },
                {
                    "Signal": "Estimated accuracy lift",
                    "What it means": f"+{accuracy_lift_estimate(scored)}% from evidence coverage.",
                },
            ],
            hide_index=True,
            use_container_width=True,
        )
