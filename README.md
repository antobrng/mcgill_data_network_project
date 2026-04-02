# Project: The Auto-Regression Engine
**Repository:** [antobrng/mcgill_data_network_project](https://github.com/antobrng/mcgill_data_network_project)
> *From Raw Data to Interpretable Insights in Seconds.*

## Our Goal
Our goal is to give access to Statistical Insight to anyone. Running a robust regression model shouldn’t take long or require a bachelor in Statistics, but the reality of data science is fraught with friction.

### The Problem
* **For Beginners:** The learning curve is overwhelming. Choosing the right algorithm (Linear, Logistic, Ridge, Lasso), properly evaluating combinations of variables, and translating metrics like Adjusted $R^2$ or AIC into actionable business logic blocks non-technical users from finding insights.
* **For Data Scientists:** Prototyping is incredibly tedious. Analysts waste hours writing complex code to clean datasets, build preprocessing pipelines, and manually loop through feature combinations just to establish a baseline model.

## The Solution: An End-to-End Automated Engine
We built a unified platform consisting of an intelligent data-cleaning backend (`clean_data.py`), an exhaustive model-evaluation core (`engine.py`), and an intuitive Streamlit UI (`app.py`). Simply upload a raw CSV, select your dependent variable, and the engine handles the math, the selection, and the final interpretation.

### Why It Changes the Game for Beginners
* **Zero-Code Interface:** The Streamlit app provides a seamless, click-to-run experience. If you know what variable you want to predict, the UI does the rest.
* **Automatic Algorithm Routing:** The engine dynamically detects if the target variable is continuous (e.g., price) or categorical (e.g., house vs. apartment) and automatically routes the data to the correct algorithms, swapping smoothly between Adjusted $R^2$ and Accuracy metrics.
* **AI-Powered Interpretation:** Through our Gemini API integration, the UI doesn't just hand the user a wall of math. It generates a plain-English summary explaining what the winning model means, why a specific algorithm was chosen, and whether the predictive score is strong or weak.

### Why Data Scientists Will Love It
* **Exhaustive "Best Subset" Selection:** `engine.py` doesn't just guess. It rigorously loops through subset combinations (testing combinations of 1 to 3 features) alongside a full multivariate baseline to find the perfect balance of simplicity and accuracy.
* **Advanced Regularization Built-In:** It automatically tests standard Linear/Logistic models against Log-Linear and Polynomial transformations (Degree 2). It even applies Ridge and Lasso penalties to polynomial regressions to capture complex non-linear relationships without overfitting.
* **Statistically Sound Validation:** The engine is smart enough to scale its evaluation methodology based on data volume. It automatically triggers strict 5-fold Cross-Validation (Out-of-Sample) for large datasets ($\ge$ 1000 rows), while using In-Sample fitting penalized by the Akaike Information Criterion (AIC) for smaller datasets.
* **The "Clean" Pipeline:** The cleaning module automatically standardizes strings, coerces dates, maps categorical variables to regression-ready integers, and handles missing data thresholds—saving hours of data prep before the regression even starts.

## The Value Proposition
We aren't just automating code; we are automating the *methodology* of a data scientist. By combining rigorous econometric principles, scalable software architecture, and an AI-interpreted UI, our project bridges the gap between a raw CSV and actionable business intelligence.

Getting Started: How to Execute the Code
1. Install Dependencies

This project is built using Python. To run the engine and the UI, you need to install the following core data science and web libraries.

Run this command in your terminal:

Bash
pip install pandas numpy scipy scikit-learn streamlit altair google-genai
2. Set Up the Gemini API Key (Required for AI Insights)

To enable the "AI Executive Summary" feature, you need a free Google Gemini API key.

Generate an API key from Google AI Studio.

Set the key as an environment variable in the terminal window you plan to run the app from:

For Mac/Linux:

Bash
export GEMINI_API_KEY="your_api_key_here"
For Windows (Command Prompt):

DOS
set GEMINI_API_KEY="your_api_key_here"
3. Launch the Application

Once your dependencies are installed and your API key is set, launch the Streamlit interface directly from your terminal:

Bash
streamlit run app.py
(Note: If you encounter a "command not found" error, try running python3 -m streamlit run app.py instead).

# Getting started: How to execute the code

## Install Dependencies

`pip install pandas numpy scipy scikit-learn streamlit altair google-genai`

## Set Up the Gemini API Key

To enable the "AI Executive Summary" feature, you need a free Google Gemini API key.
Generate an API key from Google AI Studio.

Set the key as an environment variable in the terminal window you plan to run the app from:

`export GEMINI_API_KEY="your_api_key_here`

## Launch the application

From the terminal launch the steamlit terminal

`streamlit run app.py`

Note: If you encounter a "command not found" error, try running python3 -m streamlit run app.py instead).

The UI will automatically open in your default web browser (typically at http://localhost:8501). From there, simply upload your CSV file, clean the data, select your target variable, and start generating insights.


