import streamlit as st
import pandas as pd
import tempfile
import os

# Import the core functions from your existing files
from clean_data import clean_csv_pipeline
from engine import evaluate_all_predictors, CV_FOLDS

# Import the new Google GenAI SDK
from google import genai
from google.genai import types

# --- UI Configuration ---
st.set_page_config(page_title="Data Cleaning & Regression UI", layout="wide")
st.title("⚙️ Automated Regression & Cleaning Engine")

# --- Initialize Gemini Client ---
# It automatically picks up the GEMINI_API_KEY from your environment variables
try:
    client = genai.Client()
    api_connected = True
except Exception as e:
    api_connected = False
    st.sidebar.warning("⚠️ Gemini API Key not found. AI Interpretation will be disabled. Set GEMINI_API_KEY as an environment variable.")

# --- Session State Management ---
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# --- SECTION 1: File Upload & Cleaning ---
st.header("1. Upload & Clean Data")
uploaded_file = st.file_uploader("Upload your raw CSV file", type=["csv"])

if uploaded_file is not None:
    if st.button("🧹 Clean Data"):
        with st.spinner("Cleaning in progress..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            result = clean_csv_pipeline(tmp_path)
            os.remove(tmp_path)

            if result["data"] is not None:
                st.session_state.cleaned_df = result["data"]
                st.session_state.report = result["report"]
                st.session_state.results_df = None # Reset results if new data uploaded
                st.success("Data successfully cleaned and factorized!")
            else:
                st.error("Failed to clean the data. Check the logs.")

if st.session_state.cleaned_df is not None:
    with st.expander("View Cleaning Report & Data Preview"):
        st.text(st.session_state.report)
        st.dataframe(st.session_state.cleaned_df.head(10))

    st.divider()

    # --- SECTION 2: Regression Engine ---
    st.header("2. Regression Analysis")
    
    df = st.session_state.cleaned_df
    
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Select Dependent Variable (Target)", options=df.columns)
    
    with col2:
        use_cv = len(df) >= 1000
        calc_method = f"{CV_FOLDS}-fold Out-of-Sample (CV)" if use_cv else "In-Sample (Standard Fit)"
        st.info(f"**Rows:** {len(df)} | **Evaluation Method:** {calc_method}")

    if st.button("🚀 Start Regression"):
        with st.spinner(f"Evaluating Best Subsets and Multivariate models for '{target}'..."):
            # Run the engine and save to session state so it survives re-renders
            st.session_state.results_df = evaluate_all_predictors(df, dependent_var=target, use_cv=use_cv)
            st.success("Regression Complete!")
            
    # Display results if they exist
    if st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        winner = results_df.iloc[0]
        
        st.subheader("🏆 Best Predictor")
        st.metric(label=f"Variables: {winner['independent']}", 
                  value=f"{winner['score']:.4f} {winner['metric']}",
                  delta=f"AIC: {winner['aic']:.2f}" if pd.notna(winner['aic']) else None,
                  delta_color="inverse")
        
        st.subheader("Leaderboard (Top 20)")
        st.dataframe(results_df.head(20), use_container_width=True)

        st.divider()

        # --- SECTION 3: AI Interpretation ---
        st.header("3. AI Interpretation")
        
        if not api_connected:
             st.info("Set your GEMINI_API_KEY environment variable to enable AI insights.")
        else:
             if st.button("🧠 Generate AI Insight"):
                 with st.spinner("Gemini is analyzing the winning model..."):
                     
                     # Construct the prompt using the winning model's details
                     prompt = f"""
                     You are an expert data scientist explaining regression results to a stakeholder.
                     
                     Context: I ran an automated regression engine to predict '{target}'.
                     Dataset size: {len(df)} rows.
                     
                     The winning model has the following attributes:
                     - Independent Variable(s): {winner['independent']}
                     - Regression Type: {winner['regression']}
                     - Score: {winner['score']:.4f} ({winner['metric']})
                     
                     Please provide a short, professional, 2-3 paragraph summary explaining:
                     1. What these results actually mean in plain English.
                     2. Whether this score is considered strong or weak.
                     3. Why the specific regression type (e.g., Polynomial, Ridge, Lasso) might have won for these specific variables.
                     """
                     
                     try:
                         # Make the API call
                         response = client.models.generate_content(
                             model='gemini-2.5-flash',
                             contents=prompt,
                         )
                         
                         # Display the result
                         st.markdown("### 🤖 Gemini's Analysis")
                         st.write(response.text)
                         
                     except Exception as e:
                         st.error(f"Error calling Gemini API: {e}")