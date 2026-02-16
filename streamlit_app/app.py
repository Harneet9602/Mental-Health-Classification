import streamlit as st
import pandas as pd
from inference import predict_text
from support_ai import get_support_message

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="MindSight | Mental Health NLP",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# THEME TOGGLE
# ======================================================
theme_mode = st.toggle("🌙 Dark Mode", value=True)

if theme_mode:
    st.markdown("""
    <style>
    body {background-color:#0e1117;color:white;}
    .stTextArea textarea {background:#1a1f2b;color:white;border-radius:12px;}
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    body {background-color:#F7F9FC;color:black;}
    .stTextArea textarea {background:white;color:black;border-radius:12px;}
    </style>
    """, unsafe_allow_html=True)

# ======================================================
# HERO HEADER
# ======================================================
st.markdown("""
<div style="
background: linear-gradient(135deg,#667eea,#764ba2);
padding:30px;
border-radius:20px;
color:white;
margin-bottom:25px;
">
<h1>🧠 MindSight Mental Health Analyzer</h1>
<p>AI-powered detection of Anxiety, Depression, Stress and Suicidal ideation signals.</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# CONTROL PANEL
# ======================================================
c1, c2, c3 = st.columns(3)

with c1:
    sensitivity = st.slider("🎯 Sensitivity", 0.3, 0.9, 0.6)

with c2:
    show_chart = st.toggle("📊 Show Insights", value=True)

with c3:
    compact_mode = st.toggle("⚡ Compact Mode", value=False)

# ======================================================
# INPUT AREA
# ======================================================
col1, col2 = st.columns([1.6,1])

with col1:
    st.subheader("📝 Input Text")

    user_text = st.text_area(
        "Paste text below:",
        height=260,
        placeholder="e.g. I feel exhausted and overwhelmed lately..."
    )

    analyze_btn = st.button("🔍 Analyze Text", use_container_width=True)

# ======================================================
# INFERENCE
# ======================================================
if analyze_btn:

    if user_text.strip()=="":
        st.error("Please enter some text.")

    else:

        with st.spinner("Running MentalBERT inference..."):
            label, results, explanation = predict_text(user_text)

        sorted_preds = sorted(results.items(), key=lambda x:x[1], reverse=True)

        top1_label, top1_prob = sorted_preds[0]
        top2_label, top2_prob = sorted_preds[1]

        high_risk = False
        if label == "Suicidal":
            high_risk = True
        elif label in ["Depression","Stress","Anxiety"]:
            if results[label] > sensitivity:
                high_risk = True

        badge_colors = {
            "Suicidal":"#ff4b4b",
            "Depression":"#7F8C8D",
            "Anxiety":"#3498DB",
            "Stress":"#F1C40F",
            "Normal":"#2ECC71"
        }

        # ==================================================
        # RESULT CARD
        # ==================================================
        with col2:

            st.markdown(f"""
            <span style="
            background:{badge_colors.get(top1_label,'#888')};
            padding:6px 12px;
            border-radius:12px;
            color:white;
            font-weight:bold;">
            {top1_label}
            </span>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(12px);
            padding:25px;
            border-radius:18px;
            box-shadow:0px 8px 24px rgba(0,0,0,0.3);
            margin-top:10px;
            ">
            <h2>🥇 {top1_label}</h2>
            <h3>{top1_prob*100:.1f}% confidence</h3>
            <hr>
            <p>🥈 Secondary: {top2_label} — {top2_prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # ==================================================
        # CONFIDENCE BARS
        # ==================================================
        if show_chart:

            df_probs = pd.DataFrame(
                list(results.items()),
                columns=["Condition","Probability"]
            ).sort_values("Probability", ascending=False)
        
            for cond, prob in df_probs.values:
        
                color = "#ff4b4b" if cond=="Suicidal" else "#4facfe"
        
                html = f"""
                <div style="margin-bottom:14px;">
        
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span>{cond}</span>
                        <span>{prob*100:.1f}%</span>
                    </div>
        
                    <div style="
                        width:100%;
                        background:#e0e0e0;
                        border-radius:8px;
                        height:10px;
                    ">
                        <div style="
                            width:{prob*100}%;
                            background:{color};
                            height:10px;
                            border-radius:8px;
                        "></div>
                    </div>
        
                </div>
                """
        
                st.markdown(html, unsafe_allow_html=True)
        # ==================================================
        # SUPPORT AI
        # ==================================================
        if high_risk:

            support_msg = get_support_message(user_text, label, results)

            st.markdown(f"""
            <div style="
            background: linear-gradient(135deg,#ffecd2,#fcb69f);
            padding:20px;
            border-radius:16px;
            margin-top:20px;">
            <h4>🤝 Support Assistant</h4>
            <p>{support_msg}</p>
            </div>
            """, unsafe_allow_html=True)

            if label=="Suicidal":
                st.error("⚠️ The text suggests possible serious distress.")

                st.info("""
If you are in India and need immediate support:

• Kiran Mental Health Helpline: 1800-599-0019  
• Sneha Foundation: 044-24640050
""")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")

st.markdown("""
### Try my Agentic AI Companion

GitHub:
https://github.com/Harneet9602/Sehaj-Holistic-AI-Companion

Live App:
https://sehaj-holistic-ai-companion-mxmfmefmmhgtxcweevkx7e.streamlit.app/
""")
