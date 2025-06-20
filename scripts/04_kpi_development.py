import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

print("Loading enhanced dataset...")
df = pd.read_csv('../data/processed/ita_osha_enhanced.csv')

print("="*60)
print("KPI & KRI DEVELOPMENT REPORT")
print("="*60)

# Create output directory for KPI reports
os.makedirs('../outputs/kpi_dashboards', exist_ok=True)

# Set professional color palette
colors = px.colors.qualitative.Set2

# ==================== SECTION 1: SAFETY KPIs ====================
print("\n1. CALCULATING CORE SAFETY KPIs...")

# Calculate standard safety KPIs
df['trir'] = np.where(
    df['total_hours_worked'] > 0,
    (df['total_injuries'] / df['total_hours_worked']) * 200000,
    0
)

df['ltifr'] = np.where(
    df['total_hours_worked'] > 0,
    (df['total_dafw_cases'] / df['total_hours_worked']) * 1000000,
    0
)

df['dart_rate'] = np.where(
    df['total_hours_worked'] > 0,
    ((df['total_dafw_cases'] + df['total_djtr_cases']) / df['total_hours_worked']) * 200000,
    0
)

df['severity_rate'] = np.where(
    df['total_injuries'] > 0,
    (df['total_dafw_days'] + df['total_djtr_days']) / df['total_injuries'],
    0
)

df['fatality_rate'] = np.where(
    df['annual_average_employees'] > 0,
    (df['total_deaths'] / df['annual_average_employees']) * 100000,
    0
)

print("✓ Calculated TRIR, LTIFR, DART Rate, Severity Rate, Fatality Rate")

print("Cleaning extreme TRIR values...")
initial_count = len(df)
df = df[df['trir'] <= 50]  # TRIR over 50 is unrealistic
trir_removed = initial_count - len(df)
print(f"Removed {trir_removed:,} records with TRIR > 50")

# ==================== SECTION 2: INDUSTRY BENCHMARKING ====================
print("\n2. DEVELOPING INDUSTRY BENCHMARKS...")

# Filter to only companies with injuries (TRIR > 0) for meaningful comparison
df_with_injuries = df[df['trir'] > 0].copy()
print(f"Filtered to {len(df_with_injuries):,} companies with injuries (from {len(df):,} total)")

# Industry KPI benchmarks for companies WITH injuries (USE MEAN - official standard)
industry_kpis = df_with_injuries.groupby('naics_sector_name').agg({
    'trir': ['mean', 'median', 'std', 'count'],  # Put mean first - official standard
    'ltifr': ['mean', 'median'], 
    'dart_rate': ['mean', 'median'],
    'severity_rate': ['mean', 'median'],
    'fatality_rate': ['mean', 'median'],
    'total_injuries': 'sum',
    'annual_average_employees': 'sum'
}).round(2)

# Flatten column names
industry_kpis.columns = ['_'.join(col).strip() for col in industry_kpis.columns]

# Calculate industry risk scores using MEAN (official standard)
industry_kpis['risk_score'] = (
    industry_kpis['trir_mean'] * 0.4 +         # Use mean!
    industry_kpis['dart_rate_mean'] * 0.3 +    # Use mean!
    industry_kpis['fatality_rate_mean'] * 0.3  # Use mean!
).round(2)

# Risk classification
industry_kpis['risk_level'] = pd.cut(
    industry_kpis['risk_score'],
    bins=[0, 2, 5, float('inf')],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

print("✓ Created industry benchmarks and risk classifications")

# ==================== SECTION 3: COMPANY SIZE ANALYSIS ====================
print("\n3. ANALYZING PERFORMANCE BY COMPANY SIZE...")

size_labels = {1: 'Small (<20)', 2: 'Medium (20-249)', 3: 'Large (250+)'}
size_kpis = df.groupby('size').agg({
    'trir': ['mean', 'median'],
    'ltifr': ['mean', 'median'],
    'dart_rate': ['mean', 'median'],
    'severity_rate': ['mean', 'median'],
    'injury_rate_per_100_employees': ['mean', 'median']
}).round(2)

size_kpis.columns = ['_'.join(col).strip() for col in size_kpis.columns]
size_kpis['size_label'] = [size_labels[idx] for idx in size_kpis.index]

print("✓ Analyzed KPIs by company size")

# ==================== SECTION 4: KRI DEVELOPMENT ====================
print("\n4. DEVELOPING KEY RISK INDICATORS...")

# Year-over-year trend analysis
yearly_trends = df.groupby(['naics_sector_name', 'year_filing_for']).agg({
    'trir': 'mean',
    'ltifr': 'mean'
}).reset_index()

# Calculate trend direction (improving/worsening)
sector_trends = yearly_trends.groupby('naics_sector_name').apply(
    lambda x: np.polyfit(x['year_filing_for'], x['trir'], 1)[0]
).reset_index()
sector_trends.columns = ['naics_sector_name', 'trend_slope']
sector_trends['trend_direction'] = np.where(
    sector_trends['trend_slope'] > 0.1, 'Worsening',
    np.where(sector_trends['trend_slope'] < -0.1, 'Improving', 'Stable')
)

print("✓ Calculated risk trend indicators")

# ==================== SECTION 5: FIXED VISUALIZATIONS ====================
print("\n5. CREATING PNG-FRIENDLY KPI DASHBOARDS...")

# 5.1 SIMPLE FIX: Industry Risk Assessment - just add text to existing scatter
top_industries = industry_kpis.nlargest(12, 'trir_count')

fig = px.scatter(
    top_industries.reset_index(),
    x='trir_mean',
    y='dart_rate_mean',
    size='trir_count',
    color='risk_level',
    hover_name='naics_sector_name',
    text='naics_sector_name',
    title='Industry Risk Assessment: TRIR vs DART Rate',
    labels={'trir_mean': 'TRIR (Average)', 'dart_rate_mean': 'DART Rate (Average)'},
    color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
)
fig.update_traces(textposition='top center', textfont_size=8)
fig.update_layout(title_x=0.5, height=600)
fig.write_html('../outputs/kpi_dashboards/industry_risk_assessment.html')
fig.write_image('../outputs/kpi_dashboards/industry_risk_assessment.png', width=1200, height=600, scale=2)
fig.show()

# 5.2 FIXED: Performance Benchmarking with better text positioning
best_performers = industry_kpis.nsmallest(8, 'trir_mean').sort_values('trir_mean')
worst_performers = industry_kpis.nlargest(8, 'trir_mean').sort_values('trir_mean', ascending=False)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Best Safety Performers (Lowest TRIR)', 'Highest Risk Sectors (Highest TRIR)'),
    horizontal_spacing=0.15
)

# Truncate long industry names for better display
best_names = [name[:25] + '...' if len(name) > 25 else name for name in best_performers.index]
worst_names = [name[:25] + '...' if len(name) > 25 else name for name in worst_performers.index]

fig.add_trace(
    go.Bar(
        x=best_performers['trir_mean'],
        y=best_names,
        orientation='h',
        marker_color='darkgreen',
        text=[f'{val:.2f}' for val in best_performers['trir_mean']],
        textposition='inside',
        textfont=dict(size=11, color='black'),
        name='Best',
        hovertemplate='<b>%{customdata}</b><br>TRIR: %{x:.2f}<extra></extra>',
        customdata=best_performers.index
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        x=worst_performers['trir_mean'],
        y=worst_names,
        orientation='h',
        marker_color='darkred',
        text=[f'{val:.2f}' for val in worst_performers['trir_mean']],
        textposition='inside',
        textfont=dict(size=11, color='black'),
        name='Worst',
        hovertemplate='<b>%{customdata}</b><br>TRIR: %{x:.2f}<extra></extra>',
        customdata=worst_performers.index
    ),
    row=1, col=2
)

# Update layout with more space for text
fig.update_layout(
    height=700, 
    width=1600,
    title_text="Industry Performance Benchmarking", 
    title_x=0.5, 
    showlegend=False,
    font=dict(size=12)
)

# Add more margin for the text
fig.update_xaxes(range=[0, max(best_performers['trir_mean'].max(), worst_performers['trir_mean'].max()) * 1.3], row=1, col=1)
fig.update_xaxes(range=[0, max(best_performers['trir_mean'].max(), worst_performers['trir_mean'].max()) * 1.3], row=1, col=2)

fig.write_html('../outputs/kpi_dashboards/performance_benchmarking.html')
fig.write_image('../outputs/kpi_dashboards/performance_benchmarking.png', width=1600, height=700, scale=2)

# 5.3 SIMPLE FIX: Trend Analysis - add text labels
trending_data = sector_trends.merge(industry_kpis[['trir_mean']], left_on='naics_sector_name', right_index=True)
trending_data = trending_data.sort_values('trir_mean', ascending=False).head(10)

fig = px.scatter(
    trending_data,
    x='trir_mean',
    y='trend_slope',
    hover_name='naics_sector_name',
    color='trend_direction',
    text='naics_sector_name',
    title='Risk Trend Analysis: Current Performance vs Trend Direction',
    labels={'trir_mean': 'Current TRIR Level', 'trend_slope': 'Trend Direction (Slope)'},
    color_discrete_map={'Improving': 'green', 'Stable': 'blue', 'Worsening': 'red'}
)
fig.update_traces(textposition='top center', textfont_size=8)
fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No Change Line")
fig.update_layout(title_x=0.5, height=600)
fig.write_html('../outputs/kpi_dashboards/trend_analysis.html')
fig.write_image('../outputs/kpi_dashboards/trend_analysis.png', width=1200, height=600, scale=2)
fig.show()

# 5.4 KPI Comparison by Company Size (this one was fine, just minor improvements)
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('TRIR by Company Size', 'LTIFR by Company Size', 'DART Rate by Company Size', 'Severity Rate by Company Size')
)

kpi_metrics = ['trir_mean', 'ltifr_mean', 'dart_rate_mean', 'severity_rate_mean']
titles = ['TRIR', 'LTIFR', 'DART Rate', 'Severity Rate']
positions = [(1,1), (1,2), (2,1), (2,2)]

for i, (metric, title, pos) in enumerate(zip(kpi_metrics, titles, positions)):
    fig.add_trace(
        go.Bar(
            x=size_kpis['size_label'],
            y=size_kpis[metric],
            name=title,
            marker_color=colors[i],
            text=[f'{val:.2f}' for val in size_kpis[metric]],
            textposition='outside',
            textfont=dict(size=11, color='black')
        ),
        row=pos[0], col=pos[1]
    )

fig.update_layout(height=650, title_text="Safety KPIs by Company Size", title_x=0.5, showlegend=False, font=dict(size=12))
fig.write_html('../outputs/kpi_dashboards/kpi_by_company_size.html')
fig.write_image('../outputs/kpi_dashboards/kpi_by_company_size.png', width=1200, height=650, scale=2)

print("✓ All PNG-friendly visualizations created with visible labels and proper spacing")

# ==================== SECTION 6: SUMMARY REPORTS ====================
print("\n6. GENERATING SUMMARY REPORTS...")

# Executive Summary Report
print(f"\nEXECUTIVE KPI SUMMARY:")
print(f"="*40)
print(f"Overall Industry TRIR: {df['trir'].mean():.2f}")
print(f"Overall Industry LTIFR: {df['ltifr'].mean():.2f}")
print(f"Overall DART Rate: {df['dart_rate'].mean():.2f}")
print(f"Total Workplace Fatalities: {df['total_deaths'].sum():,}")

print(f"\nHIGHEST RISK INDUSTRIES:")
for industry in worst_performers.head(5).index:
    trir = worst_performers.loc[industry, 'trir_mean']
    risk = industry_kpis.loc[industry, 'risk_level']
    print(f"  {industry}: TRIR {trir} ({risk})")

print(f"\nBEST SAFETY PERFORMERS:")
for industry in best_performers.head(5).index:
    trir = best_performers.loc[industry, 'trir_mean']
    risk = industry_kpis.loc[industry, 'risk_level']
    print(f"  {industry}: TRIR {trir} ({risk})")

print(f"\nCOMPANY SIZE INSIGHTS:")
for size in size_kpis.index:
    label = size_kpis.loc[size, 'size_label']
    trir = size_kpis.loc[size, 'trir_mean']
    print(f"  {label}: TRIR {trir}")

# Save comprehensive datasets
industry_kpis.to_csv('../data/processed/industry_kpi_benchmarks.csv')
size_kpis.to_csv('../data/processed/size_kpi_analysis.csv')
sector_trends.to_csv('../data/processed/industry_risk_trends.csv')

# Save enhanced dataset with all KPIs
df.to_csv('../data/processed/ita_osha_with_kpis.csv', index=False)

print(f"\nOutput files:")
print(f"✓ FIXED KPI Dashboards: ../outputs/kpi_dashboards/")
print(f"✓ Benchmark data: ../data/processed/industry_kpi_benchmarks.csv")
print(f"✓ Enhanced dataset: ../data/processed/ita_osha_with_kpis.csv")

print("\n" + "="*60)
print("FIXED KPI & KRI DEVELOPMENT COMPLETED")
print("="*60)
print("✅ Performance benchmarking: Fixed text overlap with wider layout and better spacing")
print("✅ Industry risk assessment: Added visible labels instead of hover-only")
print("✅ Trend analysis: Added visible industry labels and clearer trend indicators")
print("✅ All charts now PNG-friendly with readable labels")
