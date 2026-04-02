import sys
import warnings
import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model    import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing   import PolynomialFeatures, LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline        import Pipeline

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  SECTION 3 – REGRESSION EVALUATORS
# ══════════════════════════════════════════════════════════════

CV_FOLDS = 5

def _eval_regressor(pipeline, X, y, use_cv: bool) -> dict:
    n = len(y)
    try:
        if use_cv:
            scores = cross_val_score(pipeline, X, y, cv=CV_FOLDS, scoring="r2")
            r2 = float(max(scores.mean(), 0.0))
        else:
            pipeline.fit(X, y)
            r2 = float(max(pipeline.score(X, y), 0.0))

        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        rss = np.sum((y - y_pred) ** 2)
        
        model = pipeline.named_steps['reg']
        p = np.size(model.coef_) 
        k = p + 1 
        
        if rss > 0:
            aic = n * np.log(rss / n) + 2 * k
        else:
            aic = float('-inf')

        if n > p + 1:
            adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
        else:
            adj_r2 = 0.0
            
        return {"score": adj_r2, "metric": "Adj R²", "aic": aic}
    except Exception:
        return {"score": 0.0, "metric": "Adj R²", "aic": float('inf')}


def _eval_classifier(pipeline, X, y, use_cv: bool) -> dict:
    try:
        if use_cv:
            cv = StratifiedKFold(CV_FOLDS)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
            acc = float(max(scores.mean(), 0.0))
        else:
            pipeline.fit(X, y)
            acc = float(max(pipeline.score(X, y), 0.0))
        return {"score": acc, "metric": "Accuracy", "aic": None}
    except Exception:
        return {"score": 0.0, "metric": "Accuracy", "aic": None}


def eval_linear(X: np.ndarray, y: np.ndarray, use_cv: bool) -> dict:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])
    res = _eval_regressor(pipeline, X, y, use_cv)
    return {"regression": "Linear", **res}


def eval_log_linear(X: np.ndarray, y: np.ndarray, use_cv: bool) -> dict:
    if np.min(X) <= 0:
        return {"regression": "Log-Linear", "score": 0.0, "metric": "Adj R²", "aic": float('inf')}
        
    pipeline = Pipeline([
        ("log", FunctionTransformer(np.log, validate=True)),
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])
    res = _eval_regressor(pipeline, X, y, use_cv)
    return {"regression": "Log-Linear", **res}


def eval_polynomial(X: np.ndarray, y: np.ndarray, degree: int, use_cv: bool) -> dict:
    pipeline = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg",    LinearRegression()), 
    ])
    res = _eval_regressor(pipeline, X, y, use_cv)
    return {"regression": f"Poly OLS (deg {degree})", **res}


def eval_ridge_poly(X: np.ndarray, y: np.ndarray, degree: int, use_cv: bool) -> dict:
    pipeline = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1.0)), 
    ])
    res = _eval_regressor(pipeline, X, y, use_cv)
    return {"regression": f"Poly Ridge (deg {degree})", **res}


def eval_lasso_poly(X: np.ndarray, y: np.ndarray, degree: int, use_cv: bool) -> dict:
    pipeline = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg",    Lasso(alpha=0.1, max_iter=10000)), 
    ])
    res = _eval_regressor(pipeline, X, y, use_cv)
    return {"regression": f"Poly Lasso (deg {degree})", **res}


def eval_logistic(X: np.ndarray, y_labels: np.ndarray, use_cv: bool) -> dict:
    X_scaled = StandardScaler().fit_transform(X)
    res = _eval_classifier(LogisticRegression(max_iter=2000), X_scaled, y_labels, use_cv)
    return {"regression": "Logistic", **res}


# ══════════════════════════════════════════════════════════════
#  SECTION 4 – EVALUATION LOOP 
# ══════════════════════════════════════════════════════════════

def evaluate_all_predictors(df: pd.DataFrame, dependent_var: str, use_cv: bool) -> pd.DataFrame:
    y_col = df[dependent_var]
    is_num = pd.api.types.is_numeric_dtype(y_col)
    n_uniq = y_col.nunique()
    
    is_classification = False
    
    if not is_num and 2 <= n_uniq <= 10:
        is_classification = True
        y = LabelEncoder().fit_transform(y_col)
    elif is_num and n_uniq == 2:
        is_classification = True
        y = y_col.values.astype(float)
    elif is_num:
        y = y_col.values.astype(float)
    else:
        print(f"Target '{dependent_var}' is not suitable.")
        sys.exit(1)

    rows = []

    # Identify valid features (drop dependent var and ID columns)
    valid_features = [c for c in df.columns if c != dependent_var and c.lower() not in ['id', 'index', 'unnamed: 0']]
    
    # Cap combinations to max 3 variables to prevent exponential memory crash
    MAX_COMBO_SIZE = min(len(valid_features), 3)

    # --- PART 1: EXHAUSTIVE BEST SUBSET SELECTION (Combinations of 1 to 3) ---
    for r in range(1, MAX_COMBO_SIZE + 1):
        for combo in itertools.combinations(valid_features, r):
            combo_list = list(combo)
            
            # Format label for the UI Table
            label = " + ".join(combo_list)
            if len(label) > 36:
                label = label[:33] + "..."

            # Extract subset and convert categories to dummies
            X_subset_df = df[combo_list]
            X_subset = pd.get_dummies(X_subset_df, drop_first=True).values.astype(float)

            if not is_classification:
                rows.append({"independent": label, **eval_linear(X_subset, y, use_cv)})
                rows.append({"independent": label, **eval_log_linear(X_subset, y, use_cv)})
                
                # Only test polynomials if the feature matrix is small enough to avoid memory crashing
                if X_subset.shape[1] <= 10:
                    for deg in [2]: # Restrict to deg 2 for multivariate stability
                        rows.append({"independent": label, **eval_polynomial(X_subset, y, deg, use_cv)})
                        rows.append({"independent": label, **eval_ridge_poly(X_subset, y, deg, use_cv)})
                        rows.append({"independent": label, **eval_lasso_poly(X_subset, y, deg, use_cv)})
            else:
                rows.append({"independent": label, **eval_logistic(X_subset, y, use_cv)})

    # --- PART 2: MULTIVARIATE REGRESSION (All Variables Baseline) ---
    X_mult_df = df[valid_features]
    X_mult = pd.get_dummies(X_mult_df, drop_first=True).values.astype(float)
    label_all = "[ALL VARIABLES]"

    if not is_classification:
        rows.append({"independent": label_all, **eval_linear(X_mult, y, use_cv)})
        rows.append({"independent": label_all, **eval_log_linear(X_mult, y, use_cv)})
        
        if X_mult.shape[1] <= 15:
            for deg in [2]:
                rows.append({"independent": label_all, **eval_polynomial(X_mult, y, deg, use_cv)})
                rows.append({"independent": label_all, **eval_ridge_poly(X_mult, y, deg, use_cv)})
                rows.append({"independent": label_all, **eval_lasso_poly(X_mult, y, deg, use_cv)})
    else:
        rows.append({"independent": label_all, **eval_logistic(X_mult, y, use_cv)})

    result_df = (
        pd.DataFrame(rows, columns=["independent", "regression", "score", "metric", "aic"])
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return result_df


# ══════════════════════════════════════════════════════════════
#  SECTION 5 – DISPLAY
# ══════════════════════════════════════════════════════════════

# Increased terminal width to fit combination names
W = 100

def divider(char="─"): print(char * W)

def header(text):
    divider("=")
    print(f"  {text}")
    divider("=")

def sub_header(text):
    divider()
    print(f"  {text}")
    divider()

ICONS = {
    "Log-Linear": "~", 
    "Linear": "o", 
    "Poly OLS": "*", 
    "Poly Ridge": "§", 
    "Poly Lasso": "$", 
    "Logistic": "^"
}

def print_table(results: pd.DataFrame, top_n: int = 20):
    sub_header(f"Top {top_n} regressions  (Dependent <- Independent)")
    print(f"  {'#':<3} {'Independent Variables':<36} {'Regression':<22} {'Score':>8}  {'Metric':<7} {'AIC':>10}")
    divider()
    for rank, (_, row) in enumerate(results.head(top_n).iterrows(), 1):
        icon = "."
        for key, val in ICONS.items():
            if key in row["regression"]:
                icon = val
                break
        
        aic_val = row.get("aic")
        aic_str = f"{aic_val:>10.2f}" if pd.notna(aic_val) and aic_val != float('inf') else "       N/A"

        print(
            f"  {rank:<3} {row['independent']:<36} "
            f"{icon} {row['regression']:<20} {row['score']:>8.4f}  {row['metric']:<7} {aic_str}"
        )
    divider()
    print("  o Linear   ~ Log-Linear   * OLS Poly   § Ridge Poly   $ Lasso Poly   ^ Logistic")

def print_winner(row: pd.Series, dependent_var: str):
    print()
    print("  +-- BEST RESULT " + "-" * (W - 16) + "+")
    print(f"  |  Dependent var   : {dependent_var}")
    print(f"  |  Independent var : {row['independent']}")
    print(f"  |  Regression type : {row['regression']}")
    print(f"  |  Score ({row['metric']:<7}) : {row['score']:.4f}")
    if pd.notna(row['aic']):
        print(f"  |  AIC Score       : {row['aic']:.2f}")
    print("  +" + "-" * (W - 2) + "+")


# ══════════════════════════════════════════════════════════════
#  SECTION 6 – ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    header("AUTO REGRESSION ENGINE  v8.0")

    filepath = sys.argv[1] if len(sys.argv) > 1 else input("  CSV file path : ").strip().strip('"\'')
    print(f"\n  Loading -> {filepath}")
    try:
        try:
            df = pd.read_csv(filepath)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="latin-1")
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        print(f"  File not found: {filepath}")
        sys.exit(1)

    print(f"  {len(df)} rows  {len(df.columns)} columns")

    sub_header("Available variables")
    for i, col in enumerate(df.columns):
        dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        print(f"  [{i:>2}]  {col:<30}  ({dtype})")

    print()
    raw = input("  Enter dependent variable (target) : ").strip()
    if raw.lstrip("-").isdigit():
        dependent_var = df.columns[int(raw)]
    else:
        match = {c.lower(): c for c in df.columns}
        dependent_var = match.get(raw.lower(), raw)

    if dependent_var not in df.columns:
        print(f"  '{dependent_var}' not found.")
        sys.exit(1)

    print(f"\n  Dependent variable : '{dependent_var}'")
    
    use_cv = len(df) >= 1000
    if use_cv:
        print(f"  Dataset size >= 1000 rows. Running {CV_FOLDS}-fold CV (Out-of-Sample) ...\n")
    else:
        print(f"  Dataset size < 1000 rows. Running standard fit (In-Sample) ...\n")

    results = evaluate_all_predictors(df, dependent_var, use_cv)
    print_table(results, top_n=20)  # Expanded to show top 20 since we have many more models
    print_winner(results.iloc[0], dependent_var)

    print()
    if input("  Export full results to CSV? [y/N] : ").strip().lower() == "y":
        out = "regression_results.csv"
        results.to_csv(out, index=False)
        print(f"  Saved -> {out}")

    print()
    divider("=")
    print("  Done.")
    divider("=")

if __name__ == "__main__":
    main()