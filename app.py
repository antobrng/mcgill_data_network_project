import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import altair as alt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, FunctionTransformer

from clean_data import clean_csv_pipeline
from engine import evaluate_all_predictors, CV_FOLDS

from google import genai

st.set_page_config(page_title="Data Cleaning & Regression UI", layout="wide")
st.title("⚙️ Automated Regression & Cleaning Engine")

try:
    client = genai.Client()
    api_connected = True
except Exception as e:
    api_connected = False
    st.sidebar.warning("⚠️ Gemini API Key not found. AI Interpretation will be disabled. Set GEMINI_API_KEY as an environment variable.")

if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

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
                st.session_state.results_df = None
                st.success("Data successfully cleaned and factorized!")
            else:
                st.error("Failed to clean the data. Check the logs.")

if st.session_state.cleaned_df is not None:
    with st.expander("View Cleaning Report & Data Preview"):
        st.text(st.session_state.report)
        st.dataframe(st.session_state.cleaned_df.head(10))

    st.divider()

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
            st.session_state.results_df = evaluate_all_predictors(df, dependent_var=target, use_cv=use_cv)
            st.success("Regression Complete!")
            
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

        st.header("3. Model Visualization & AI Insight")
        
        if winner['independent'] == "[ALL VARIABLES]":
            feat_cols = [c for c in df.columns if c != target and c.lower() not in ['id', 'index', 'unnamed: 0']]
        else:
            feat_cols = winner['independent'].split(' + ')
            
        X_sub = pd.get_dummies(df[feat_cols], drop_first=True)
        y_sub = df[target]

        reg_name = winner['regression']
        if "Log-Linear" in reg_name:
            model = Pipeline([("log", FunctionTransformer(np.log)), ("scaler", StandardScaler()), ("reg", LinearRegression())])
        elif "Poly" in reg_name:
            deg = int(reg_name.split("deg ")[1].replace(")", ""))
            algo = Ridge(alpha=1.0) if "Ridge" in reg_name else Lasso(alpha=0.1, max_iter=10000) if "Lasso" in reg_name else LinearRegression()
            model = Pipeline([("poly", PolynomialFeatures(degree=deg, include_bias=False)), ("scaler", StandardScaler()), ("reg", algo)])
        elif "Logistic" in reg_name:
            model = Pipeline([("scaler", StandardScaler()), ("reg", LogisticRegression(max_iter=2000))])
        else: 
            model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])

        model.fit(X_sub, y_sub)
        y_pred = model.predict(X_sub)
        
        coef_str = ""
        is_classification = "Logistic" in reg_name
        
        if not is_classification:
            reg_step = model.named_steps["reg"]
            feature_names = model.named_steps["poly"].get_feature_names_out(X_sub.columns) if "poly" in model.named_steps else X_sub.columns
            coefs = reg_step.coef_
            
            coef_dict = {name: val for name, val in zip(feature_names, coefs) if abs(val) > 0.0001}
            sorted_coefs = sorted(coef_dict.items(), key=lambda item: abs(item[1]), reverse=True)
            
            coef_str += f"Base Intercept: {reg_step.intercept_:.4f}\n"
            for name, val in sorted_coefs[:10]: # Top 10 to not crash the AI prompt
                coef_str += f"- {name}: {val:.4f}\n"
            if len(sorted_coefs) > 10:
                coef_str += f"... plus {len(sorted_coefs)-10} minor terms.\n"

        col_chart, col_ai = st.columns([1.2, 1])
        
        with col_chart:
            st.markdown("##### 📈 Line of Best Fit")
            plot_df = pd.DataFrame({'Actual': y_sub, 'Predicted': y_pred})
            
            if len(feat_cols) == 1 and not is_classification:
                x_col = feat_cols[0]
                plot_df[x_col] = df[x_col]
                
                scatter = alt.Chart(plot_df).mark_circle(size=60, opacity=0.6).encode(
                    x=alt.X(x_col, title=x_col, scale=alt.Scale(zero=False)), 
                    y=alt.Y('Actual', title=target), 
                    tooltip=[x_col, 'Actual']
                )
                
                x_min, x_max = df[x_col].min(), df[x_col].max()
                x_smooth = pd.DataFrame({x_col: np.linspace(x_min, x_max, 300)})
                
                y_smooth = model.predict(x_smooth)
                curve_df = pd.DataFrame({x_col: x_smooth[x_col], 'Predicted': y_smooth})
                
                line = alt.Chart(curve_df).mark_line(color='red', size=3).encode(
                    x=x_col, y='Predicted'
                )
                
                st.altair_chart(scatter + line, use_container_width=True)
                st.caption(f"Red line shows the '{reg_name}' fitted curve over your data points.")
                
            elif not is_classification:

                min_val = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
                max_val = max(plot_df['Actual'].max(), plot_df['Predicted'].max())
                
                scatter = alt.Chart(plot_df).mark_circle(size=60, opacity=0.6).encode(
                    x=alt.X('Predicted', title=f'Predicted {target}', scale=alt.Scale(domain=[min_val, max_val])),
                    y=alt.Y('Actual', title=f'Actual {target}', scale=alt.Scale(domain=[min_val, max_val])),
                    tooltip=['Predicted', 'Actual']
                )
                line_df = pd.DataFrame({'val': [min_val, max_val]})
                line = alt.Chart(line_df).mark_line(color='red', strokeDash=[5, 5]).encode(x='val', y='val')
                
                st.altair_chart(scatter + line, use_container_width=True)
                st.caption("Red dashed line represents perfect prediction accuracy (Actual = Predicted).")

        with col_ai:
            st.markdown("##### 🤖 Executive Summary")
            if not api_connected:
                 st.info("Set your GEMINI_API_KEY environment variable to enable AI insights.")
            else:
                 if st.button("Generate Insight"):
                     with st.spinner("Analyzing coefficients..."):
                         
                         prompt = f"""
                         You are an executive dashboard AI. Output your analysis directly. Do NOT include any greetings or conversational filler.
                         
                         Target Variable: '{target}'
                         Winning Predictors: {winner['independent']}
                         Model Type: {winner['regression']}
                         Score: {winner['score']:.4f} ({winner['metric']})
                         Dataset Size: {len(df)} rows
                         
                         Coefficients (Note: Features were standardized using Z-scores, so a coefficient of 5 means a 1 Standard Deviation increase in the feature changes the target by 5):
                         {coef_str}
                         
                         Provide a straightforward, 3-bullet executive summary:
                         * **The Drivers:** Explain which specific variables are driving the target up or down based on the highest coefficients provided. Keep it simple.
                         * **The Reliability:** A blunt assessment of the score (e.g., "Moderate: explains X% of variance, leaving the rest to hidden factors").
                         * **The Shape & Algorithm:** Explain exactly WHY {winner['regression']} fit best. If it is a Polynomial (Degree 2), explicitly state that it discovered a non-linear curve—such as a "U-shape" (high at the extremes, lower in the middle) or a curve of "diminishing returns"—and explain why that mathematical shape makes logical real-world sense for these specific variables.
                         """
                         
                         try:
                             response = client.models.generate_content(
                                 model='gemini-2.5-flash',
                                 contents=prompt,
                             )
                             st.write(response.text)
                             
                         except Exception as e:
                             st.error(f"Error calling Gemini API: {e}")