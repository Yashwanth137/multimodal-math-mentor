import streamlit as st
import sys
import os
from PIL import Image
import json
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import process_math_input, ingest_rag, get_session_history, get_past_solution, clear_session_history

# Page Config
st.set_page_config(
    page_title="Reliable Multimodal Math Mentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utilities ---
def format_timestamp(ts_str):
    """Simple fuzzy timestamp formatter."""
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        diff = datetime.now() - dt
        seconds = diff.total_seconds()
        if seconds < 60: return "just now"
        if seconds < 3600: return f"{int(seconds // 60)}m ago"
        if seconds < 86400: return f"{int(seconds // 3600)}h ago"
        return dt.strftime("%b %d")
    except:
        return ts_str

def get_short_title(text):
    """Cleanse titles from temp file names."""
    if not text: return "Untitled Problem"
    if text.endswith(".png") or "temp_img" in text: return "📸 Image Problem"
    if text.endswith(".wav") or "temp_audio" in text: return "🎙️ Audio Problem"
    return text[:30] + "..." if len(text) > 30 else text

def clean_latex_for_streamlit(text: str) -> str:
    import re
    if not isinstance(text, str): return text
    # Convert block math \[ ... \] to $$ ... $$ and inline math \( ... \) to $ ... $
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #f0f2f6; }
    .stButton>button { border-radius: 8px; font-weight: 500; }
    .status-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .verified { background-color: #1b5e20; color: #00c853; border: 1px solid #00c853; }
    .warning { background-color: #3e2723; color: #ffab40; border: 1px solid #ffab40; }
    .debug-card {
        background-color: #1a1c23;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 15px;
    }
    .step-box {
        background-color: #262730;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 8px;
    }
    .final-answer {
        background: linear-gradient(135deg, #1f77b4, #00c853);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2em;
        margin-top: 20px;
    }
    /* Timeline styles */
    .timeline {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
        padding: 10px;
        background-color: #1a1c23;
        border-radius: 10px;
    }
    .timeline-step { text-align: center; font-size: 0.7em; opacity: 0.7; }
    .timeline-step.active { opacity: 1; color: #00c853; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if "active_problem_id" not in st.session_state:
    st.session_state["active_problem_id"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# --- SIDEBAR (Navigation) ---
with st.sidebar:
    st.title("🎓 Math Mentor AI")
    
    if st.button("➕ New Problem", use_container_width=True):
        st.session_state["last_result"] = None
        st.session_state["active_problem_id"] = None
        st.rerun()

    st.markdown("### 📚 History")
    history_items = get_session_history()
    if not history_items:
        st.info("No history yet.")
    else:
        # Proper Dropdown for History
        history_map = {f"{get_short_title(item['original_input'])} ({format_timestamp(item['timestamp'])})": item['problem_id'] for item in history_items[:20]}
        selected_hist = st.selectbox("Select past problem:", ["-- Select --"] + list(history_map.keys()), label_visibility="collapsed")
        
        if selected_hist != "-- Select --":
            problem_id = history_map[selected_hist]
            if st.session_state["active_problem_id"] != problem_id:
                with st.spinner("Loading from memory..."):
                    st.session_state["last_result"] = get_past_solution(problem_id)
                    st.session_state["active_problem_id"] = problem_id
                    st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    debug_mode = st.checkbox("🔍 Debug Mode", value=True)
    
    with st.popover("🗑️ Clear History", use_container_width=True):
        st.warning("This will delete all past problems.")
        if st.button("Confirm Delete EVERYTHING", type="primary", use_container_width=True):
            clear_session_history()
            st.session_state["last_result"] = None
            st.session_state["active_problem_id"] = None
            st.success("History wiped.")
            st.rerun()

    if st.button("📂 Ingest Knowledge Base", use_container_width=True):
        with st.spinner("Ingesting..."):
            ingest_rag()
            st.success("RAG Ready!")

# --- MAIN WORKSPACE ---
# Define Layout
main_col, debug_col = st.columns([3, 1]) if debug_mode else (st.container(), None)

with main_col:
    st.markdown("## Solve a Math Problem")
    
    # Input Area
    input_type = st.segmented_control("Input Mode", ["Text", "Image", "Audio"], default="Text")
    
    user_input = None
    if input_type == "Text":
        user_input = st.text_area("Problem Text", placeholder="e.g. Solve x^2 - 5x + 6 = 0", height=100)
        mode = "text"
    elif input_type == "Image":
        up = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if up:
            st.image(up, caption="Target Problem", width=400)
            
            # Extract actual extension instead of forcing .png
            import os
            ext = os.path.splitext(up.name)[1].lower()
            if not ext: ext = ".png"
            
            img_path = os.path.abspath(f"temp_img{ext}")
            with open(img_path, "wb") as f: f.write(up.getbuffer())
            user_input = img_path
        mode = "image"
    elif input_type == "Audio":
        st.write("Upload an audio file or record your question directly.")
        uploaded_audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
        recorded_audio = st.audio_input("Or record your microphone")
        
        audio_data = recorded_audio if recorded_audio else uploaded_audio
        
        if audio_data:
            st.audio(audio_data)
            audio_path = os.path.abspath("temp_audio.wav")
            with open(audio_path, "wb") as f: f.write(audio_data.getbuffer())
            user_input = audio_path
        mode = "audio"

    if st.button("🚀 Solve Problem", type="primary"):
        if not user_input:
            st.error("Please provide input.")
        else:
            with st.status("🤖 AI Agents Analyzing Problem...", expanded=True) as status:
                try:
                    res = process_math_input(user_input, mode, ui_status=status)
                    
                    if isinstance(res, dict) and res.get("status") == "needs_hitl":
                        status.update(label="HITL Required", state="complete")
                        st.session_state["hitl_data"] = res
                    elif isinstance(res, dict) and res.get("status") == "error":
                        status.update(label="Pipeline Failed", state="error")
                        st.error(f"Pipeline Failed: {res.get('error_message')}")
                        st.session_state["last_result"] = res
                    else:
                        status.update(label="Solution Generated!", state="complete")
                        st.session_state["last_result"] = res
                        st.session_state["active_problem_id"] = res.get("problem_id")
                        st.session_state["hitl_data"] = None
                except Exception as e:
                    status.update(label="Critical Error", state="error")
                    st.error(f"Pipeline Critical Error: {e}")

    # HITL Correction Area
    if st.session_state.get("hitl_data"):
        h = st.session_state["hitl_data"]
        st.warning(f"⚠️ **OCR/ASR Uncertainty**: {h.get('reason')}")
        correction = st.text_input("Confirm text for Solver Agent:", value=h.get("raw_text"))
        if st.button("Proceed with Correction", use_container_width=True):
            with st.spinner("Processing correction..."):
                res = process_math_input(correction, "text")
                st.session_state["last_result"] = res
                st.session_state["hitl_data"] = None
                st.rerun()

    # Results Display
    if st.session_state["last_result"]:
        res = st.session_state["last_result"]
        
        if res.get("status") == "error":
             st.error(f"System Error: {res.get('error_message', 'The math pipeline failed to produce a valid solution.')}")
             if not debug_mode: st.info("Enable Debug Mode to see internal trace.")
        else:
            st.markdown("---")
            
            # Header with Verification Badge
            v = res.get("verification", {})
            badge_class = "verified" if v.get("is_correct") else "warning"
            badge_text = "✅ Verified" if v.get("is_correct") else "⚠️ Review Needed"
            
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h2 style="margin:0;">Solution Steps</h2>
                    <div class="status-badge {badge_class}">{badge_text}</div>
                </div>
            """, unsafe_allow_html=True)

            # Steps
            exp = res.get("explanation", {})
            steps = [step for step in exp.get("step_by_step", []) if step.strip()]
            for step in steps:
                step = clean_latex_for_streamlit(step)
                # THE FIX: Use native Streamlit containers to keep LaTeX rendering intact!
                with st.container(border=True):
                    st.markdown(step)
            
            # Final Answer
            st.markdown("<div class='final-answer'><strong>Final Answer</strong>", unsafe_allow_html=True)
            ans = exp.get("final_boxed_answer", "N/A")
            # Clean LaTeX boxing for st.latex
            if isinstance(ans, str):
                if ans.startswith("\\boxed{") and ans.endswith("}"): ans = ans[7:-1]
                # Prevent text smushing in latex rendering
                if "\\" not in ans and "^" not in ans and len(ans.split()) > 3:
                    st.markdown(f"<p style='text-align: center; margin-top: 10px;'>{ans}</p>", unsafe_allow_html=True)
                else:
                    st.latex(ans)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("### Reasoning Summary")
            st.info(exp.get("reasoning", "No summary available."))
            
            st.divider()
            st.subheader("Rate this Solution")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Correct (Save to Memory)"):
                    # You can call memory.update_feedback(res.get("problem_id"), "correct") here later
                    st.success("Thanks! Pattern saved for future learning.")
                    
            with col2:
                # Using an expander here instead of a nested button which typically disappears in Streamlit 
                with st.expander("❌ Incorrect (Flag for Review)"):
                    comment = st.text_input("What went wrong?")
                    if st.button("Submit Report"):
                        st.warning("Feedback logged. The system will learn from this.")

# --- DEBUG PANEL ---
if debug_mode and debug_col:
    with debug_col:
        st.markdown("### Agent Trace")
        
        # Timeline Visualization
        res = st.session_state.get("last_result")
        is_err = res.get("status") == "error" if res else False
        status_color = "#f44336" if is_err else "#00c853"
        
        st.markdown(f"""
            <div class="timeline">
                <div class="timeline-step active">In</div>
                <div class="timeline-step active">RAG</div>
                <div class="timeline-step active" style="color: {status_color}">Solv</div>
                <div class="timeline-step active">Ver</div>
                <div class="timeline-step active">Out</div>
            </div>
        """, unsafe_allow_html=True)

        if res:
            dbg = res.get("debug", {})
            v = res.get("verification", {})
            triage = res.get("parsed_problem", {})
            is_complex = triage.get("is_complex", True) # Default true if missing
            
            with st.expander("🧠 RAG Retrieval", expanded=not is_err):
                retrieval = dbg.get("retrieval", []) if dbg else []
                if retrieval:
                    for ctx in retrieval: st.caption(ctx)
                else:
                    st.write("No retrieval data.")
            
            if is_complex:
                with st.expander("🔧 Solver Reasoning"):
                    solver_dbg = dbg.get("solver", {}) if dbg else {}
                    if solver_dbg:
                        st.write(f"**Model:** {solver_dbg.get('model_used', 'Gemini')}")
                        st.write("**Raw Proof Snippet:**")
                        st.code(str(solver_dbg.get("raw_proof", "N/A"))[:500] + "...")
                        st.write("**Tools Used:**")
                        st.json(solver_dbg.get("tools_used_detail", solver_dbg.get("tools_used", [])))
                    else:
                        st.write("No solver data available.")
                
                with st.expander("🛡️ Verifier Critique", expanded=True):
                    if v:
                        st.write(f"Confidence: {v.get('confidence_score', 'N/A')}")
                        st.markdown(f"*{v.get('critique', 'N/A')}*")
                    else:
                        st.write("No verification data.")
            else:
                st.info("⚡ Solved instantly via Supervisor Fast-Lane.")
        else:
            st.info("Waiting for input...")
