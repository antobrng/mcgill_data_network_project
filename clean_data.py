import pandas as pd
import numpy as np
from scipy.stats import zscore
import datetime
import re
import argparse
import sys


def stamp(log_list, msg):
    """Appends a timestamped message to the log list."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_list.append(f"[{timestamp}] {msg}")


def handle_missing_data(df, log, drop_col_thresh=0.8, drop_row_thresh=0.8):
    """Drops mostly empty rows/cols and imputes remaining NAs intelligently."""
    df_clean = df.copy()
    
    # 2.1a Drop columns that are mostly empty
    col_miss_rate = df_clean.isna().mean()
    cols_to_drop = col_miss_rate[col_miss_rate > drop_col_thresh].index.tolist()
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        stamp(log, f"Dropped {len(cols_to_drop)} column(s) with >{drop_col_thresh*100}% missing: {', '.join(cols_to_drop)}")

    # 2.1b Drop rows that are mostly empty
    row_miss_rate = df_clean.isna().mean(axis=1)
    n_rows_dropped = (row_miss_rate > drop_row_thresh).sum()
    if n_rows_dropped > 0:
        df_clean = df_clean[row_miss_rate <= drop_row_thresh]
        stamp(log, f"Dropped {n_rows_dropped} row(s) with >{drop_row_thresh*100}% missing values")

    # 2.1c Fill remaining numeric NAs with column median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        n_na = df_clean[col].isna().sum()
        if n_na > 0:
            med = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(med)
            stamp(log, f"Filled {n_na} NA(s) in '{col}' with median ({med:.4g})")

    # 2.1d Fill remaining character NAs with "Unknown"
    chr_cols = df_clean.select_dtypes(include=['object', 'string']).columns
    for col in chr_cols:
        n_na = df_clean[col].isna().sum()
        if n_na > 0:
            df_clean[col] = df_clean[col].fillna("Unknown")
            stamp(log, f"Filled {n_na} NA(s) in '{col}' with 'Unknown'")

    return df_clean

def flag_outliers(df, log, outlier_z=3.0):
    """Flags outliers based on Z-score without removing them."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].std() > 0:  # Avoid division by zero
            # Calculate absolute Z-scores, ignoring NAs just in case
            z_scores = np.abs(zscore(df[col], nan_policy='omit'))
            n_out = (z_scores > outlier_z).sum()
            if n_out > 0:
                stamp(log, f"Flagged {n_out} outlier(s) in '{col}' (|Z| > {outlier_z}) — kept in data")
    return df

def standardize_strings(df, log):
    """Trims whitespace, lowercases, and forces UTF-8 encoding."""
    df_clean = df.copy()
    chr_cols = df_clean.select_dtypes(include=['object', 'string']).columns
    
    for col in chr_cols:
        before = df_clean[col].copy()
        
        # Collapse multiple spaces, trim edges, and lowercase
        df_clean[col] = df_clean[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x).strip().lower())
        
        # Fix encoding (simulate iconv by encoding and decoding to utf-8)
        df_clean[col] = df_clean[col].apply(lambda x: x.encode('utf-8', 'ignore').decode('utf-8'))
        
        n_changed = (before != df_clean[col]).sum()
        if n_changed > 0:
            stamp(log, f"Standardized {n_changed} string(s) in '{col}' (trim/lowercase/encoding)")
            
    return df_clean

def auto_factorize(df, log, ref_levels=None, min_freq=2):
    """
    Automatically detects string/object columns, removes rare categories, 
    encodes them into integers, and returns a mapping dictionary.
    """
    df_clean = df.copy()
    ref_levels = ref_levels or {}
    
    # This dictionary will store our factor mappings
    encoding_dict = {}
    
    # Automatically detect columns that should be categorical
    cat_cols = df_clean.select_dtypes(include=['object', 'string']).columns
    
    if len(cat_cols) == 0:
        stamp(log, "No categorical columns detected to factorize.")
        return df_clean, encoding_dict
        
    for col in cat_cols:
        counts = df_clean[col].value_counts(dropna=False)
        rare_cats = counts[counts < min_freq].index
        
        if len(rare_cats) > 0:
            df_clean[col] = df_clean[col].replace(rare_cats, np.nan)
            stamp(log, f"[{col}] Replaced {len(rare_cats)} rare categories (freq < {min_freq}) with NA.")
            
        if df_clean[col].nunique() < 2:
            stamp(log, f"[{col}] Skipped encoding: not enough valid categories left after filtering.")
            continue
            
        df_clean[col] = df_clean[col].astype('category')
        
        if col in ref_levels:
            ref = ref_levels[col]
            if ref in df_clean[col].cat.categories:
                new_order = [ref] + [c for c in df_clean[col].cat.categories if c != ref]
                df_clean[col] = df_clean[col].cat.reorder_categories(new_order)
                stamp(log, f"Factorized '{col}' with reference level: '{ref}'")
        
        categories = list(df_clean[col].cat.categories)
        
        # Save the mapping to our dictionary
        col_mapping = {cat: i for i, cat in enumerate(categories)}
        encoding_dict[col] = col_mapping
        
        # Apply the encoding
        df_clean[col] = df_clean[col].cat.codes.replace(-1, np.nan)
        
        # Print the mapping summary
        print(f"\n--- Encoding Summary for: {col} ---")
        print("Integer Mapping:")
        for cat, val in col_mapping.items():
            print(f"  {cat} -> {val}")
        print("-" * 40)
        stamp(log, f"Encoded '{col}' into integers (0 to {len(categories)-1}).")
        
    return df_clean, encoding_dict

def coerce_dates(df, log):
    """Attempts to parse string columns into DateTime objects."""
    df_clean = df.copy()
    chr_cols = df_clean.select_dtypes(include=['object', 'string']).columns
    
    for col in chr_cols:
        # Prevent "Unknown" from crashing the parser
        sample_vals = df_clean[col].replace("unknown", np.nan).dropna()
        if len(sample_vals) == 0:
            continue
            
        converted = pd.to_datetime(df_clean[col], errors='coerce')
        success_rate = converted.notna().sum() / len(df_clean)
        
        if success_rate >= 0.7:
            df_clean[col] = converted
            stamp(log, f"Coerced '{col}' to DateTime")
            
    return df_clean

def handle_duplicates(df, log, strategy="keep_first"):
    """Removes duplicate rows based on the chosen strategy."""
    df_clean = df.copy()
    n_dups = df_clean.duplicated().sum()
    
    if n_dups > 0:
        keep_rule = 'last' if strategy == "keep_last" else 'first'
        df_clean = df_clean.drop_duplicates(keep=keep_rule)
        stamp(log, f"Removed {n_dups} duplicate row(s) (strategy: {strategy})")
        
    return df_clean

def check_invalid_numeric(df, log):
    """Fixes impossible negative values in columns like 'price' or 'age'."""
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    non_neg_keywords = ['age', 'price', 'amount', 'count', 'qty', 'quantity', 
                        'duration', 'hours', 'minutes', 'revenue', 'sales', 'score', 'rate']
    
    for col in num_cols:
        if any(keyword in col.lower() for keyword in non_neg_keywords):
            n_neg = (df_clean[col] < 0).sum()
            if n_neg > 0:
                # Replace negatives with NaN
                df_clean.loc[df_clean[col] < 0, col] = np.nan
                stamp(log, f"Replaced {n_neg} impossible negative(s) in '{col}' with NA")
                
                # Re-fill with median
                med = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(med)
                stamp(log, f"Re-filled those NA(s) in '{col}' with median ({med:.4g})")
                
    return df_clean


# =============================================================================
# Master Controller
# =============================================================================

def clean_csv_pipeline(filepath, 
                       na_strings=["", "NA", "#NA", "N/A", "n/a", "None", "none", "NULL", "null", "NaN", "Inf", "-Inf", "Not Applicable", "not applicable", "0"],
                       outlier_z=3.0, 
                       drop_col_thresh=0.8, 
                       drop_row_thresh=0.8, 
                       dedup_strategy="keep_first"):
    """
    Main function to execute the modular cleaning pipeline and generate a report.
    """
    log = []
    stamp(log, f"Loading file: {filepath}")
    
    # Load data
    try:
        # First, try standard UTF-8
        df = pd.read_csv(filepath, na_values=na_strings, keep_default_na=True, encoding='utf-8')
    except UnicodeDecodeError:
        stamp(log, "UTF-8 decoding failed. Retrying with 'latin-1' encoding...")
        try:
            # Fallback for special characters and accents
            df = pd.read_csv(filepath, na_values=na_strings, keep_default_na=True, encoding='latin-1')
        except Exception as e:
            stamp(log, f"Failed to load file even with latin-1 fallback: {e}")
            return {"data": None, "report": "\n".join(log), "log": log}
    except Exception as e:
        stamp(log, f"Failed to load file: {e}")
        return {"data": None, "report": "\n".join(log), "log": log}
        
    original_dims = df.shape
    stamp(log, f"Loaded: {original_dims[0]} rows x {original_dims[1]} columns")
    
    missing_before = df.isna().sum().sum()
    stamp(log, f"Missing values on load: {missing_before}")

    # Pipeline execution
    df = handle_missing_data(df, log, drop_col_thresh, drop_row_thresh)
    df = flag_outliers(df, log, outlier_z)
    df = standardize_strings(df, log)
    df, factor_mappings = auto_factorize(df, log)
    df = coerce_dates(df, log)
    df = handle_duplicates(df, log, dedup_strategy)
    df = check_invalid_numeric(df, log)

    # Final summary
    final_dims = df.shape
    missing_after = df.isna().sum().sum()
    
    stamp(log, "--- CLEANING COMPLETE ---")
    stamp(log, f"Rows:    {original_dims[0]} -> {final_dims[0]}  (removed {original_dims[0] - final_dims[0]})")
    stamp(log, f"Columns: {original_dims[1]} -> {final_dims[1]}  (removed {original_dims[1] - final_dims[1]})")
    stamp(log, f"Missing values: {missing_before} -> {missing_after}")

    # Build printable report
    report = "\n".join([
        "=======================================================",
        "                 DATA CLEANING REPORT                  ",
        "=======================================================",
        *log,
        "======================================================="
    ])

    return {
        "data": df, 
        "report": report, 
        "log": log,
        "mappings": factor_mappings
    }


# =============================================================================
# Command Line Interface (Argparse)
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Modular Data Cleaning Pipeline")
    # I changed this to --i (lowercase) to match what you typed in your terminal!
    parser.add_argument("--i", required=True, help="Path to the input CSV dataset")
    parser.add_argument("--o", default="~/Desktop/hackathon/data/cleaned_dataset.csv", help="Path to save the cleaned CSV (default: cleaned_dataset.csv)")
    
    args = parser.parse_args()
    
    result = clean_csv_pipeline(args.i)
    
    if result["data"] is not None:
        # Print the detailed cleaning report to the terminal
        print(result["report"])
        
        # Save the resulting DataFrame to the specified output file
        try:
            result["data"].to_csv(args.o, index=False)
            print(f"\n[Success] Cleaned data successfully saved to: {args.o}")
        except Exception as e:
            print(f"\n[Error] Could not save the file: {e}")
            sys.exit(1)
    else:
        # This will now print EXACTLY why it failed (e.g., FileNotFoundError)
        print(result["report"])
        sys.exit(1)