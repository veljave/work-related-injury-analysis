import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset as pandas dataframe 
print("Loading the ITA OSHA Combined dataset...")
df = pd.read_csv('../data/raw/ITA_OSHA_Combined.csv')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("\n" + "="*50)

# Basic inspection
print("BASIC DATASET INFORMATION")
print("="*50)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names and types:")
print(df.dtypes)

print("\nDataset info:")
df.info()

print("\nBasic statistics:")
print(df.describe())

print("\n" + "="*50)
print("DATA QUALITY ASSESSMENT")
print("="*50)

# Check for missing values
print("\nMissing values per column:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
}).sort_values('Missing Count', ascending=False)

print(missing_summary[missing_summary['Missing Count'] > 0])

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Unique values in key columns
print("\nUnique values in key columns:")
key_columns = ['company_name', 'state', 'naics_code', 'industry_description']
for col in key_columns:
    if col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

print("\n" + "="*50)
print("COLUMN-SPECIFIC INSPECTION")
print("="*50)

# Look at specific columns that might need cleaning
for col in df.columns:
    print(f"\n--- {col} ---")
    if df[col].dtype == 'object':  # String columns
        print(f"Sample values: {df[col].dropna().head(3).tolist()}")
        if df[col].nunique() < 20:  # If few unique values, show them all
            print(f"All unique values: {df[col].unique()}")
    else:  # Numeric columns
        print(f"Range: {df[col].min()} to {df[col].max()}")
        print(f"Sample values: {df[col].dropna().head(5).tolist()}")

print("\n" + "="*50)
print("POTENTIAL ISSUES TO ADDRESS")
print("="*50)

issues = []

# Check for obvious data quality issues
for col in df.columns:
    # Check for weird values in numeric columns
    if df[col].dtype in ['int64', 'float64']:
        if (df[col] < 0).any():
            issues.append(f"{col}: Contains negative values")
        if df[col].isnull().any():
            issues.append(f"{col}: Contains missing values")
    
    # Check string columns for inconsistencies
    elif df[col].dtype == 'object':
        if df[col].isnull().any():
            issues.append(f"{col}: Contains missing values")
        
        # Check for leading/trailing spaces
        if df[col].dropna().astype(str).str.strip().ne(df[col].dropna().astype(str)).any():
            issues.append(f"{col}: Contains leading/trailing spaces")

if issues:
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
else:
    print("No obvious data quality issues detected!")

print("\n" + "="*50)
print("SAVE INSPECTION RESULTS")
print("="*50)


# Save summary statistics
summary_stats = df.describe(include='all')
summary_stats.to_csv('../data/processed/summary_statistics.csv')
print("Summary statistics saved to: ../data/processed/summary_statistics.csv")

# Save missing data report
missing_summary.to_csv('../data/processed/missing_data_report.csv')
print("Missing data report saved to: ../data/processed/missing_data_report.csv")

# Save sample of data for quick reference
df.sample(100).to_csv('../data/processed/data_sample_100.csv', index=False)
print("Data sample (100 rows) saved to: ../data/processed/data_sample_100.csv")

print("\n" + "="*50)
print("INSPECTION COMPLETE!")
print("="*50)
print("Review the output above to understand what cleaning is needed.")
print("Common next steps:")
print("1. Handle missing values")
print("2. Standardize text fields (company names, states, etc.)")
print("3. Convert data types where needed")
print("4. Remove duplicates if any")
print("5. Create derived variables (injury rates, etc.)")