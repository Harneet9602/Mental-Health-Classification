import streamlit as st
import pandas as pd
import altair as alt
from inference import predict_text

# 1. Page Configuration (Must be first)
st.set_page_config(
    page_title="MindSight | Mental Health NLP",
    page_icon="🧠",
    layout="wide", # 'wide' layout looks more professional for dashboards
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for a "Pinterest-clean" look
st.markdown("""
    <style>
    /* Remove the default main menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom title font styling */
    .title-text {
        font-family: 'Sans-serif';
        font-weight: 700;
        color: #2C3E50;
    }
    
    /* Card-like styling for containers */
    .stTextArea textarea {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar - Context & Disclaimer
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=80)
    st.title("MindSight")
    st.caption("Advanced NLP for Mental Health Detection")
    st.markdown("---")
    
    st.markdown("**About the Model**")
    st.info(
        """
        This tool uses **MentalBERT**, a fine-tuned transformer model 
        designed to classify mental health signals in social media text.
        """
    )
    
    st.markdown("---")
    st.warning(
        "⚠️ **DISCLAIMER:**\n"
        "This tool is for **academic research purposes only** (Capstone Project). "
        "It is not a substitute for professional medical diagnosis. "
        "If you or someone you know is in crisis, please contact a helpline."
    )

# 4. Main Content Area
st.title("🧠 Mental Health Text Analyzer")
st.markdown("##### Detect patterns of Anxiety, Depression, and Stress in text using Deep Learning.")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### 📝 Input Text")
    user_text = st.text_area(
        "Paste a social media post, diary entry, or text snippet here:",
        height=250,
        placeholder="e.g., 'I feel overwhelming sadness and I don't want to get out of bed...'"
    )
    
    analyze_btn = st.button("🔍 Analyze Text", use_container_width=True, type="primary")

# 5. Prediction Logic
if analyze_btn:
    if user_text.strip() == '':
        st.toast("⚠️ Please enter some text to analyze.", icon="⚠️")
    else:
        with st.spinner('Running inference on MentalBERT...'):
            # Call your inference function
            label, results, explanation = predict_text(user_text)

        # --- DISPLAY RESULTS IN COLUMN 2 ---
        with col2:
            st.markdown("### 📊 Analysis Results")
            
            # Dynamic Color Coding for the Result
            # 'Normal' gets green, others get warning colors
            color_map = {
                "Normal": "green",
                "Anxiety": "orange",
                "Depression": "red",
                "Bipolar": "red",
                "Stress": "orange",
                "Suicidal": "inverse", # Black/Dark Red
                "Personality disorder": "orange"
            }
            status_color = color_map.get(label, "blue")

            # Display the Top Prediction as a Metric
            st.subheader(f"Prediction: :{status_color}[{label}]")
            
            # --- CREATE A BAR CHART ---
            # Convert dictionary results to DataFrame for the chart
            df = pd.DataFrame(list(results.items()), columns=['Condition', 'Probability'])
            
            # Create a beautiful Altair chart
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Probability', axis=alt.Axis(format='%')),
                y=alt.Y('Condition', sort='-x'),
                color=alt.Color('Probability', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=['Condition', alt.Tooltip('Probability', format='.1%')]
            ).properties(
                height=300
            )
            
            st.altair_chart(chart, use_container_width=True)

        # --- OPTIONAL: EXPLANATION SECTION (Full Width) ---
        if explanation:
            with st.expander("ℹ️ Model Explanation"):
                st.write(explanation)
