import streamlit as st
import pandas as pd
import altair as alt
from inference import predict_text
from support_ai import get_support_message


# ======================================================
# 1. PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="MindSight | Mental Health NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# 2. GLOBAL DARK MODE + CSS
# ======================================================
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    body {
        background-color: #0e1117;
        color: white;
    }

    .stTextArea textarea {
    background-color: #f8f9fa !important;
    color: #111111 !important;   /* 👈 FIX */
    border-radius: 12px;
    font-size: 16px;
}


    .stButton button {
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
    }

    .result-card {
        padding: 20px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# 3. SIDEBAR
# ======================================================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3062/3062634.png",
        width=80
    )
    st.title("MindSight")
    st.caption("Mental Health NLP Analyzer")
    st.markdown("---")

    st.info(
        "This app uses **MentalBERT**, a transformer model fine-tuned "
        "to detect mental health signals from text."
    )

    st.warning(
        "⚠️ **DISCLAIMER**\n\n"
        "For academic & research use only. "
        "Not a medical diagnostic tool."
    )

# ======================================================
# 4. MAIN LAYOUT
# ======================================================
st.title("🧠 Mental Health Text Analyzer")
st.markdown(
    "Analyze text for **Anxiety, Depression, Stress, Suicidal ideation**, and more."
)

col1, col2 = st.columns([1.6, 1])

# ======================================================
# 5. INPUT
# ======================================================
with col1:
    st.subheader("📝 Input Text")
    user_text = st.text_area(
        "Paste text below:",
        height=260,
        placeholder="e.g. I feel exhausted and overwhelmed lately..."
    )

    analyze_btn = st.button(
        "🔍 Analyze Text",
        use_container_width=True,
        type="primary"
    )

# ======================================================
# 6. INFERENCE + RESULTS
# ======================================================
if analyze_btn:
    if user_text.strip() == "":
        st.error("Please enter some text.")
    else:
        with st.spinner("Running MentalBERT inference..."):
            label, results, explanation = predict_text(user_text)

            sorted_preds = sorted(
                results.items(),
                key=lambda x: x[1],
                reverse=True
            )

            top1_label, top1_prob = sorted_preds[0]
            top2_label, top2_prob = sorted_preds[1]

        # ==================================================
        # COLOR MAP
        # ==================================================
        color_map = {
            "Normal": "#2ECC71",
            "Anxiety": "#3498DB",
            "Stress": "#F1C40F",
            "Depression": "#7F8C8D",
            "Bipolar": "#9B59B6",
            "Suicidal": "#C0392B",
            "Personality disorder": "#E67E22"
        }

        bg_color = color_map.get(top1_label, "#34495E")

        # ==================================================
        # RESULTS CARD
        # ==================================================
        with col2:
            st.markdown(
                f"""
                <div class="result-card" style="background-color:{bg_color};">
                    <h2>🥇 {top1_label}</h2>
                    <h4>{top1_prob*100:.1f}% confidence</h4>
                    <hr>
                    <h4>🥈 Secondary: {top2_label}</h4>
                    <p>{top2_prob*100:.1f}% confidence</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ==================================================
            # BAR CHART (ALTAIR)
            # ==================================================
            df_probs = pd.DataFrame(
                list(results.items()),
                columns=["Condition", "Probability"]
            ).sort_values("Probability", ascending=False)

            chart = alt.Chart(df_probs).mark_bar().encode(
                x=alt.X(
                    "Probability",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])
                ),
                y=alt.Y("Condition", sort="-x"),
                color=alt.Color(
                    "Probability",
                    scale=alt.Scale(scheme="blues"),
                    legend=None
                ),
                tooltip=[
                    "Condition",
                    alt.Tooltip("Probability", format=".2%")
                ]
            ).properties(height=300)

            st.altair_chart(chart, use_container_width=True)

        # ==================================================
        # OPTIONAL EXPLANATION
        # ==================================================
        if explanation:
            with st.expander("ℹ️ Model Explanation"):
                st.write(explanation)

st.markdown("---")

st.markdown(
"""
### Try my Agentic AI Companion

Holistic AI assistant for guided emotional support:

GitHub:
https://github.com/Harneet9602/Sehaj-Holistic-AI-Companion

Live App:
https://sehaj-holistic-ai-companion-mxmfmefmmhgtxcweevkx7e.streamlit.app/
"""
)


