import pandas as pd
import numpy as np

print("Loading ITA OSHA Combined dataset...")
df = pd.read_csv('../data/raw/ITA_OSHA_Combined.csv', low_memory=False)
df_clean = df.copy()

print("Starting data cleaning process...\n")

# Track cleaning actions
cleaning_log = []

# 1. Drop administrative columns
drop_cols = ['id', 'ein', 'establishment_id', 'created_timestamp', 'change_reason', 'source', 'delete', 'zip_code', 'industry_description']
df_clean = df_clean.drop(columns=drop_cols)
cleaning_log.append(f"Dropped {len(drop_cols)} administrative columns")

# 2. Fix establishment_type mixed types
df_clean = df_clean[df_clean['establishment_type'] != 'Executive and Legislative Offices']
df_clean['establishment_type'] = pd.to_numeric(df_clean['establishment_type'], errors='coerce')
cleaning_log.append("Fixed establishment_type mixed data types")

# 3. Remove rows with negative values
negative_cols = ['total_hours_worked', 'total_other_cases', 'total_dafw_days', 'total_djtr_days', 'total_injuries']
initial_rows = len(df_clean)
for col in negative_cols:
    df_clean = df_clean[df_clean[col] >= 0]
negative_removed = initial_rows - len(df_clean)
cleaning_log.append(f"Removed {negative_removed} rows with negative values")

# 4. Statistical outlier detection and handling
print("Performing outlier analysis...")

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

# Analyze numeric columns for outliers
numeric_cols = ['annual_average_employees', 'total_hours_worked', 'total_injuries', 
                'total_dafw_days', 'total_djtr_days']

initial_outlier_count = len(df_clean)

print(f"Analyzing outliers using IQR method...")
for col in numeric_cols:
    outliers = detect_outliers_iqr(df_clean, col)
    outlier_count = outliers.sum()
    print(f"  {col}: {outlier_count:,} outliers ({outlier_count/len(df_clean)*100:.1f}%)")

# Identify impossible data patterns
print(f"\nIdentifying impossible data patterns...")
df_clean['hours_per_employee'] = np.where(
    df_clean['annual_average_employees'] > 0,
    df_clean['total_hours_worked'] / df_clean['annual_average_employees'],
    0
)

impossible_patterns = {
    'excessive_hours_per_employee': df_clean['hours_per_employee'] > 4000,
    'zero_employees_with_hours': (df_clean['annual_average_employees'] == 0) & (df_clean['total_hours_worked'] > 0),
    'injuries_exceed_employees': df_clean['total_injuries'] > df_clean['annual_average_employees'] * 2,
    'unrealistically_low_hours': (df_clean['total_hours_worked'] < 2000) & (df_clean['total_injuries'] > 0),  # <2000 hours with injuries
    'extreme_injury_density': (df_clean['total_injuries'] / df_clean['total_hours_worked']) > 0.1,  # More than 10% injury rate per hour
    'part_time_seasonal_bias': df_clean['hours_per_employee'] < 1500,  # Remove part-time/seasonal companies
}

total_impossible = 0
for pattern_name, pattern_mask in impossible_patterns.items():
    count = pattern_mask.sum()
    total_impossible += count
    print(f"  {pattern_name}: {count:,} records")

# Remove impossible records
print(f"\nRemoving {total_impossible:,} impossible records...")
for pattern_mask in impossible_patterns.values():
    df_clean = df_clean[~pattern_mask]

# Apply winsorization (99th percentile capping)
print(f"Applying winsorization...")
for col in numeric_cols:
    p99 = df_clean[col].quantile(0.99)
    outliers_capped = (df_clean[col] > p99).sum()
    if outliers_capped > 0:
        df_clean[col] = df_clean[col].clip(upper=p99)
        print(f"  {col}: Capped {outliers_capped:,} values at {p99:.0f}")

df_clean = df_clean.drop('hours_per_employee', axis=1)
outliers_removed = initial_outlier_count - len(df_clean)
cleaning_log.append(f"Removed {outliers_removed:,} impossible records and applied 99th percentile capping")

# 5. Clean text fields
text_columns = ['company_name', 'establishment_name', 'street_address', 'city']
total_whitespace = 0
for col in text_columns:
    whitespace_count = (df_clean[col].str.strip() != df_clean[col]).sum()
    total_whitespace += whitespace_count
    df_clean[col] = df_clean[col].astype(str).str.strip()
cleaning_log.append(f"Cleaned whitespace from {total_whitespace} text field entries")

# 6. Standardize state codes
df_clean['state'] = df_clean['state'].str.upper().str.strip()
cleaning_log.append(f"Standardized state codes ({df_clean['state'].nunique()} unique values)")

# 7. Handle missing values
df_clean = df_clean.dropna(subset=['no_injuries_illnesses'])
missing_est_type = df_clean['establishment_type'].isnull().sum()
df_clean['establishment_type'] = df_clean['establishment_type'].fillna(1.0)
empty_companies = (df_clean['company_name'] == '').sum()
df_clean = df_clean[df_clean['company_name'] != '']
cleaning_log.append(f"Handled missing values: imputed {missing_est_type} establishment_type, dropped {empty_companies + 2} rows")

# 8. Clean NAICS codes
print("Cleaning NAICS codes...")
initial_rows = len(df_clean)
valid_naics_sectors = ['11', '21', '22', '23', '31', '32', '33', '42', '44', '45', 
                      '48', '49', '51', '52', '53', '54', '55', '56', '61', '62', 
                      '71', '72', '81', '92']
df_clean['naics_sector'] = df_clean['naics_code'].astype(str).str[:2]
df_clean = df_clean[df_clean['naics_sector'].isin(valid_naics_sectors)]
invalid_removed = initial_rows - len(df_clean)
cleaning_log.append(f"Removed {invalid_removed} records with invalid NAICS codes")

# 9. Create derived variables
print("Creating derived variables...")
df_clean['injury_rate_per_100_employees'] = np.where(
    df_clean['annual_average_employees'] > 0,
    (df_clean['total_injuries'] / df_clean['annual_average_employees']) * 100,
    0
)

df_clean['trir'] = np.where(
    df_clean['total_hours_worked'] > 0,
    (df_clean['total_injuries'] / df_clean['total_hours_worked']) * 200000,
    0
)

# NAICS sector mapping
naics_mapping = {
    '11': 'Agriculture, Forestry, Fishing', '21': 'Mining, Quarrying, Oil/Gas',
    '22': 'Utilities', '23': 'Construction', '31': 'Manufacturing',
    '32': 'Manufacturing', '33': 'Manufacturing', '42': 'Wholesale Trade',
    '44': 'Retail Trade', '45': 'Retail Trade', '48': 'Transportation & Warehousing',
    '49': 'Transportation & Warehousing', '51': 'Information', '52': 'Finance & Insurance',
    '53': 'Real Estate & Rental', '54': 'Professional Services', '55': 'Management of Companies',
    '56': 'Administrative Support', '61': 'Educational Services', '62': 'Health Care & Social Assistance',
    '71': 'Arts, Entertainment, Recreation', '72': 'Accommodation & Food Services',
    '81': 'Other Services', '92': 'Public Administration'
}

df_clean['naics_sector_name'] = df_clean['naics_sector'].map(naics_mapping)
cleaning_log.append("Added derived variables: injury rates and industry sector names")

# 10. Save cleaned dataset
import os
os.makedirs('../data/processed', exist_ok=True)

output_file = '../data/processed/ita_osha_enhanced.csv'
df_clean.to_csv(output_file, index=False)

sample_file = '../data/processed/ita_osha_sample_10k.csv'
df_clean.sample(10000, random_state=42).to_csv(sample_file, index=False)

# Generate cleaning report
print("="*60)
print("DATA CLEANING & PREPARATION REPORT")
print("="*60)
print(f"Original dataset shape: {df.shape}")
print(f"Final dataset shape: {df_clean.shape}")
print(f"Rows removed: {len(df) - len(df_clean):,} ({((len(df) - len(df_clean)) / len(df) * 100):.1f}%)")
print(f"Columns remaining: {len(df_clean.columns)}")

print("\nCleaning actions performed:")
for i, action in enumerate(cleaning_log, 1):
    print(f"{i}. {action}")

print(f"\nData quality summary:")
print(f"Missing values remaining: {df_clean.isnull().sum().sum():,}")
print(f"Valid NAICS sectors: {df_clean['naics_sector'].nunique()}")
print(f"Years covered: {df_clean['year_filing_for'].min()}-{df_clean['year_filing_for'].max()}")

print(f"\nOutput files:")
print(f"✓ Enhanced dataset: {output_file}")
print(f"✓ Sample dataset: {sample_file}")

print("\n" + "="*60)
print("DATA PREPARATION COMPLETED - Ready for Phase 2: Exploratory Analysis")
print("="*60)

# ==================== DIAGNOSTIC ANALYSIS ====================
print("\n" + "="*60)
print("DIAGNOSTIC: INVESTIGATING TRIR COMPONENT VARIABLES")
print("="*60)

# Analyze total_injuries and total_hours_worked by sector
sector_analysis = df_clean.groupby('naics_sector_name').agg({
    'total_injuries': ['count', 'sum', 'mean', 'median', 'min', 'max'],
    'total_hours_worked': ['mean', 'median', 'min', 'max'],
    'annual_average_employees': ['mean', 'median'],
    'company_name': 'count'
}).round(2)

sector_analysis.columns = ['_'.join(col).strip() for col in sector_analysis.columns]

print(f"\nSECTOR ANALYSIS (sorted by injury count):")
print(f"{'Sector':<35} {'Companies':<10} {'Tot Injuries':<12} {'Med Hours':<12} {'Med Employees':<12} {'Hours/Employee':<15}")
print(f"{'-'*110}")

for sector in sector_analysis.sort_values('total_injuries_sum', ascending=False).index:
    companies = sector_analysis.loc[sector, 'company_name_count']
    total_inj = sector_analysis.loc[sector, 'total_injuries_sum']
    med_hours = sector_analysis.loc[sector, 'total_hours_worked_median']
    med_employees = sector_analysis.loc[sector, 'annual_average_employees_median']
    hours_per_emp = med_hours / med_employees if med_employees > 0 else 0
    
    print(f"{sector:<35} {companies:<10.0f} {total_inj:<12.0f} {med_hours:<12.0f} {med_employees:<12.0f} {hours_per_emp:<15.0f}")

print(f"\nSUSPICIOUS PATTERNS:")

# Check for sectors with extremely low hours
low_hours_sectors = sector_analysis[sector_analysis['total_hours_worked_median'] < 2000]
if len(low_hours_sectors) > 0:
    print(f"\nSectors with median hours < 2000:")
    for sector in low_hours_sectors.index:
        med_hours = low_hours_sectors.loc[sector, 'total_hours_worked_median']
        companies = low_hours_sectors.loc[sector, 'company_name_count']
        print(f"  {sector}: {med_hours:.0f} hours (from {companies:.0f} companies)")

# Check for sectors with extremely high injury rates
high_injury_sectors = sector_analysis[sector_analysis['total_injuries_mean'] > 5]
if len(high_injury_sectors) > 0:
    print(f"\nSectors with mean injuries > 5:")
    for sector in high_injury_sectors.index:
        mean_inj = high_injury_sectors.loc[sector, 'total_injuries_mean']
        med_inj = high_injury_sectors.loc[sector, 'total_injuries_median']
        print(f"  {sector}: Mean {mean_inj:.1f}, Median {med_inj:.1f}")

# Sample problematic companies
print(f"\nSAMPLE PROBLEMATIC COMPANIES:")
problematic = df_clean[(df_clean['total_hours_worked'] < 2000) & (df_clean['total_injuries'] > 0)]
if len(problematic) > 0:
    print(f"Companies with <2000 hours but injuries:")
    sample = problematic[['company_name', 'naics_sector_name', 'total_hours_worked', 'total_injuries', 'annual_average_employees']].head(10)
    for idx, row in sample.iterrows():
        calc_trir = (row['total_injuries'] / row['total_hours_worked']) * 200000
        print(f"  {row['company_name'][:30]:<30} | {row['naics_sector_name']:<25} | Hours: {row['total_hours_worked']:>6.0f} | Inj: {row['total_injuries']:>2.0f} | TRIR: {calc_trir:>8.1f}")