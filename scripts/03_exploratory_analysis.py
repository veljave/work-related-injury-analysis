import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

print("Loading enhanced dataset...")
df = pd.read_csv('../data/processed/ita_osha_enhanced.csv')

print("="*60)
print("EXPLORATORY DATA ANALYSIS REPORT")
print("="*60)

# Basic dataset overview
print(f"Dataset Overview:")
print(f"Total records: {len(df):,}")
print(f"Time period: {df['year_filing_for'].min()}-{df['year_filing_for'].max()}")
print(f"Unique companies: {df['company_name'].nunique():,}")
print(f"Total employees covered: {df['annual_average_employees'].sum():,}")
print(f"Total work hours: {df['total_hours_worked'].sum():,.0f}")

# Create output directory for plots
os.makedirs('../outputs/figures', exist_ok=True)

# Set professional color palette
colors = px.colors.qualitative.Set2

# 1. Year-over-Year Trends
yearly_stats = df.groupby('year_filing_for').agg({
    'company_name': 'count',
    'total_injuries': 'sum',
    'annual_average_employees': 'sum',
    'injury_rate_per_100_employees': 'mean'
}).round(2)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Number of Records', 'Total Injuries', 'Total Employees', 'Average Injury Rate per 100 Employees'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

fig.add_trace(go.Scatter(x=yearly_stats.index, y=yearly_stats['company_name'], 
                        mode='lines+markers', name='Records', line=dict(color=colors[0], width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=yearly_stats.index, y=yearly_stats['total_injuries'], 
                        mode='lines+markers', name='Injuries', line=dict(color=colors[1], width=3)), row=1, col=2)
fig.add_trace(go.Scatter(x=yearly_stats.index, y=yearly_stats['annual_average_employees'], 
                        mode='lines+markers', name='Employees', line=dict(color=colors[2], width=3)), row=2, col=1)
fig.add_trace(go.Scatter(x=yearly_stats.index, y=yearly_stats['injury_rate_per_100_employees'], 
                        mode='lines+markers', name='Injury Rate', line=dict(color=colors[3], width=3)), row=2, col=2)

fig.update_layout(height=600, title_text="Year-over-Year Trends (2016-2021)", title_x=0.5, showlegend=False)
fig.write_html('../outputs/figures/yearly_trends.html')
fig.write_image('../outputs/figures/yearly_trends.png', width=1200, height=600, scale=2)
fig.show()

# 2. Industry Sector Analysis
sector_analysis = df.groupby('naics_sector_name').size().sort_values(ascending=False).head(10)

fig = px.bar(x=sector_analysis.values, y=sector_analysis.index, orientation='h',
             title='Top 10 Industry Sectors by Record Count',
             labels={'x': 'Number of Records', 'y': 'Industry Sector'},
             color=sector_analysis.values, color_continuous_scale='Blues')

fig.update_layout(height=600, title_x=0.5, showlegend=False)
fig.update_traces(texttemplate='%{x:,}', textposition='outside')
fig.write_html('../outputs/figures/industry_sectors.html')
fig.write_image('../outputs/figures/industry_sectors.png', width=1200, height=600, scale=2)
fig.show()

# 3. Company Size Distribution
size_labels = {1: 'Small (<20)', 2: 'Medium (20-249)', 3: 'Large (250+)'}
size_dist = df['size'].value_counts().sort_index()
labels = [size_labels[size] for size in size_dist.index]

fig = px.pie(values=size_dist.values, names=labels, title='Company Size Distribution',
             color_discrete_sequence=colors)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title_x=0.5)
fig.write_html('../outputs/figures/company_size_distribution.html')
fig.write_image('../outputs/figures/company_size_distribution.png', width=800, height=600, scale=2)
fig.show()

# 4. Injury Rate Analysis
zero_injuries = (df['total_injuries'] == 0).sum()
has_injuries = (df['total_injuries'] > 0).sum()
size_labels_clean = {1: 'Small (<20)', 2: 'Medium (20-249)', 3: 'Large (250+)'}
avg_injury_rates = df.groupby('size')['injury_rate_per_100_employees'].mean()
companies_with_injuries = df[df['total_injuries'] > 0]

# Injury severity data
severity_totals = [
    companies_with_injuries['total_deaths'].sum(),
    companies_with_injuries['total_dafw_cases'].sum(),
    companies_with_injuries['total_djtr_cases'].sum(),
    companies_with_injuries['total_other_cases'].sum()
]
severity_categories = ['Deaths', 'Days Away from Work', 'Job Transfer/Restriction', 'Other Cases']

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Companies with vs without Injuries', 'Average Injury Rate by Company Size', 'Injury Severity Distribution'),
    specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
)

# Panel 1: Zero vs Has Injuries
fig.add_trace(go.Bar(x=['Zero Injuries', 'Has Injuries'], y=[zero_injuries, has_injuries],
                    text=[f'{zero_injuries:,}', f'{has_injuries:,}'], textposition='outside',
                    marker_color=[colors[0], colors[1]], name=''), row=1, col=1)

# Panel 2: Injury rates by size
size_names = [size_labels_clean[size] for size in avg_injury_rates.index]
fig.add_trace(go.Bar(x=size_names, y=avg_injury_rates.values,
                    text=[f'{v:.2f}%' for v in avg_injury_rates.values], textposition='outside',
                    marker_color=[colors[2], colors[3], colors[4]], name=''), row=1, col=2)

# Panel 3: Severity distribution
fig.add_trace(go.Bar(x=severity_categories, y=severity_totals,
                    text=[f'{v:,}' for v in severity_totals], textposition='outside',
                    marker_color=['darkred', 'red', 'orange', 'gold'], name=''), row=1, col=3)

fig.update_layout(height=500, title_text="Injury Analysis", title_x=0.5, showlegend=False)
fig.update_xaxes(tickangle=45, row=1, col=3)
fig.write_html('../outputs/figures/injury_rate_distributions.html')
fig.write_image('../outputs/figures/injury_rate_distributions.png', width=1400, height=500, scale=2)
fig.show()

# 5. High-Risk Sector Analysis
high_injury_rate = df[df['injury_rate_per_100_employees'] > 10]
if len(high_injury_rate) > 0:
    high_risk_sectors = high_injury_rate['naics_sector_name'].value_counts().head(8)
    
    fig = px.bar(x=high_risk_sectors.values, y=high_risk_sectors.index, orientation='h',
                 title='High-Risk Sectors (Companies with >10% Injury Rate)',
                 labels={'x': 'Number of High-Risk Companies', 'y': 'Industry Sector'},
                 color=high_risk_sectors.values, color_continuous_scale='Reds')
    
    fig.update_layout(height=500, title_x=0.5, showlegend=False)
    fig.update_traces(texttemplate='%{x}', textposition='outside')
    fig.write_html('../outputs/figures/high_risk_sectors.html')
    fig.write_image('../outputs/figures/high_risk_sectors.png', width=1000, height=500, scale=2)
    fig.show()

# Print summary statistics
print(f"\nIndustry Analysis:")
for sector, count in sector_analysis.items():
    print(f"  {sector}: {count:,} records")

print(f"\nCompany Size Distribution:")
for size, count in size_dist.items():
    pct = (count / len(df)) * 100
    label = size_labels[size]
    print(f"  {label}: {count:,} ({pct:.1f}%)")

print(f"\nInjury Statistics:")
print(f"Total workplace injuries: {df['total_injuries'].sum():,}")
print(f"Total fatalities: {df['total_deaths'].sum():,}")
print(f"Average injury rate per 100 employees: {df['injury_rate_per_100_employees'].mean():.2f}")
print(f"Companies with zero injuries: {zero_injuries:,} ({zero_injuries/len(df)*100:.1f}%)")
print(f"Companies with injuries: {has_injuries:,} ({has_injuries/len(df)*100:.1f}%)")

print(f"\nInjury Rate Analysis:")
print(f"\nAverage injury rates by company size:")
for size, rate in avg_injury_rates.items():
    print(f"  {size_labels_clean[size]}: {rate:.2f}%")

# Save analysis results
yearly_stats.to_csv('../data/processed/yearly_trends.csv')
sector_analysis.to_csv('../data/processed/sector_analysis.csv')

print(f"\nOutput files:")
print(f"✓ Interactive plots: ../outputs/figures/*.html")
print(f"✓ High-res images: ../outputs/figures/*.png")
print(f"✓ Analysis CSVs: ../data/processed/")

print("\n" + "="*60)
print("EXPLORATORY ANALYSIS COMPLETED - Ready for Phase 3: KPI Development")
print("="*60)
