# Parking Tickets Follow the Renters: LA Citations by Council District, 2022 to 2025

How do LA parking ticket patterns relate to housing and infrastructure? This project joins parking citations, Census ACS data, Zillow rent and council district boundaries into a district-by-year panel (15 districts, 4 years), then tests which district characteristics go with ticket rates.

**Main finding:** renter-heavy council districts receive roughly 50% more tickets per household for every 10 points of renter share (Spearman rho = 0.78, Bonferroni-corrected p = 0.012; negative binomial rate ratio 1.56, 95% CI 1.37 to 1.77). Income, household size and rent burden do not show the same pattern, and nothing tested explains which districts saw tickets grow. These are associations across 15 districts, not causal effects.

## Run the dashboard

```
pip install -r requirements.txt
streamlit run sl_dashboard.py
```

Run it from the project folder, since the app reads files with relative paths.

## Rebuild the data and analysis

```
pip install -r requirements-pipeline.txt
```

The pipeline needs a Census API key. Create a file named `api_keys.py` in the project root containing:

```python
census_api = "YOUR_KEY_HERE"
```

This file is listed in `.gitignore` and should never be committed.

## Dashboard tabs

Overview (headline metrics, scatter plot, district map), Results tables, Regression, Methods and limitations, Write-up.

## Files the dashboard reads

In `data/results/`:
- district_summary.csv, headline_numbers.csv
- results_primary.csv, results_secondary.csv, results_tertiary.csv
- nb_poisson_labeled.csv, nb_leave_one_district_out.csv
- council_districts.geojson (district shapes and label points)

In `data/`: writeup.md. Figures used by the write-up are `trend_plots.png` and `trend_plots_housing.png` in `data/results/`.

## Method summary

- Tickets are assigned to council districts with a spatial join, and Census block groups are allocated to districts by area-weighted interpolation.
- Spearman rank correlations with permutation p-values (10,000 shuffles), Bonferroni correction within the primary and housing families, and leave-one-district-out stability checks.
- Negative binomial regression of district-year ticket counts with log households as the offset and district-clustered standard errors.

See the Write-up tab for the full method, results and limitations.

## Data sources

- City of Los Angeles open data portal: parking citations and council district boundaries
- U.S. Census Bureau: American Community Survey 5-year estimates, 2024
- Zillow: observed rent index by ZIP code

Raw data files are not included in this repository because of their size. Download them from the sources above.

## Limitations

Only 15 districts, so the tests have low power and clustered standard errors are optimistic. Rates use resident denominators, so districts that draw many visitors and commuters (notably District 14) look higher. Predictors are constant across years, and the housing and transit tests were chosen after earlier results, so they are exploratory.