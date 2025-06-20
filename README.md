# Analiza Povreda na Radu

Kvantitativna analiza podataka o povredama na radu u SAD (2016-2021) za kurs "Savremeni menadžment i mrežna organizacija preduzeća".

## Pregled

Analiza ITA_OSHA_Combined.csv dataseta (1.635M zapisa) kroz razvoj KPI/KRI sistema za bezbednost na radu i benchmark poređenje između industrija i veličina kompanija.

## Faze Analize

1. **Čišćenje podataka** (`scripts/02_data_cleaning.py`)
   - Uklanjanje outlier-a i nevalidnih podataka
   - Statistička validacija (IQR metoda, winsorization)
   - Finalni dataset: 1.31M validnih zapisa

2. **Eksplorativna analiza** (`scripts/03_exploratory_analysis.py`)
   - Trendovi po godinama (2016-2021)
   - Distribucija po industrijama i veličini kompanija
   - Identifikacija visokorizičnih sektora

3. **KPI razvoj** (`scripts/04_kpi_development.py`)
   - TRIR, LTIFR, DART Rate, Severity Rate, Fatality Rate
   - Benchmark analiza po industrijama
   - Interactive dashboard kreiranje

## Ključni Nalazi

- **82%** kompanija nema evidentirane povrede
- **TRIR prosek**: 3.84 povreda na 100 FTE
- **Najrizičnije**: Real Estate (8.30), Arts/Entertainment (7.49), Transportation (7.28)
- **Najsigurnije**: Management (2.18), Mining (2.92), Professional Services (3.51)
- **Veličina uticaj**: Male kompanije 20% veći rizik od velikih

## Tehnologije

- **Python**: pandas, numpy, plotly
- **Vizualizacija**: Interactive HTML + high-res PNG
- **Statistika**: IQR outlier detection, 99th percentile winsorization

## Pokretanje

```bash
# Čišćenje podataka
python scripts/02_data_cleaning.py

# Eksplorativna analiza
python scripts/03_exploratory_analysis.py

# KPI dashboard
python scripts/04_kpi_development.py
```

## Outputi

- `data/processed/` - Očišćeni podaci
- `outputs/figures/` - Statičke vizualizacije
- `outputs/kpi_dashboards/` - KPI dashboard-i

## Dataset

ITA OSHA Combined (2016-2021) - workplace injury data across US industries