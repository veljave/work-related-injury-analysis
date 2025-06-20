# Analiza Povreda na Radu

Kvantitativna analiza podataka o povredama na radu u SAD (2016-2021) za master rad.

## Pokretanje Projekta

### Priprema

1. Kloniraj projekat i pozicioniraj se u direktorijum:
```bash
cd work-related-injury-analysis
```

2. Instaliraj zavisnosti:
```bash
pip install -r requirements.txt
```

3. Postavi podatke:
   - Stavi `ITA_OSHA_Combined.csv` u `data/raw/` folder

### Pokretanje Analize

Pokreni skriptove redom:

```bash
# 1. Čišćenje podataka
python scripts/02_data_cleaning.py

# 2. Eksplorativna analiza
python scripts/03_exploratory_analysis.py

# 3. KPI razvoj i dashboard
python scripts/04_kpi_development.py
```

### Rezultati

Rezultati analize se čuvaju u:
- `data/processed/` - Obrađeni podaci
- `outputs/figures/` - Grafici (.png i .html)
- `outputs/kpi_dashboards/` - KPI dashboard-i
