# RideShare Demand Forecasting

Hourly bike rental demand forecasting for Washington D.C.'s Capital Bikeshare system using ensemble machine learning models with SHAP interpretability.

## Problem

Bike-sharing systems need to predict hourly demand accurately to prevent empty or overstocked stations. Underprediction leads to frustrated users who find no bikes; overprediction wastes redistribution resources. This project builds a forecasting model that predicts hourly ride volume based on temporal patterns, weather conditions, and commuter behavior.

## Dataset

**Capital Bikeshare (2011–2012)** - 17,379 hourly records with 17 features including season, hour of day, weather condition, temperature, humidity, windspeed, holiday/working day status, and rental counts (casual + registered users).

Source: [Capital Bikeshare System Data](https://www.capitalbikeshare.com/system-data)

## Approach

**Exploratory Analysis** revealed strong temporal and weather-driven patterns: demand peaks at 8 AM and 5-6 PM (commuter hours), summer and autumn see highest usage, and clear weather drives significantly more rentals than rain or snow. A custom binary feature `is_peak_hour` was engineered to capture rush hour commuting windows (7-9 AM, 4-7 PM). The target variable was log-transformed (`log1p`) to reduce right skewness.

**Modeling** followed an incremental approach, starting with a Linear Regression baseline, then progressing to ensemble methods. All models used a time-aware 80/20 train-test split (chronological, not random) to prevent temporal leakage, with 5-fold cross-validation for generalizability. Hyperparameters for XGBoost and Gradient Boosting were tuned using RandomizedSearchCV.

**Interpretability** was addressed using SHAP values, which confirmed hour of day, temperature, and working day status as the dominant predictors across both XGBoost and Random Forest models.

## Results

| Model | R² | RMSE |
|-------|-----|------|
| Linear Regression | 0.345 | 144.02 |
| Random Forest | 0.841 | 70.97 |
| XGBoost (Tuned) | 0.856 | 67.61 |
| Gradient Boosting (Tuned) | 0.858 | 67.05 |

The tuned Gradient Boosting model achieved the best performance, explaining 85.8% of demand variability with an RMSE of 67 rides per hour.

**Key findings from SHAP analysis:**
- Hour of day is by far the most influential feature (commuter pattern)
- Temperature has a non-linear effect, demand peaks at moderate temperatures (0.6–0.8 normalized) and drops at extremes
- Working day status creates distinct demand shapes (commuter peaks on weekdays vs. midday peaks on weekends)

## Tech Stack

- **Language:** Python
- **ML Models:** Scikit-learn (Linear Regression, Random Forest, Gradient Boosting), XGBoost
- **Interpretability:** SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Tuning:** RandomizedSearchCV, 5-fold Cross-Validation

## Project Structure

```
├── notebooks/
│   └── rideshare_forecasting.ipynb   # Full analysis and modeling pipeline
├── requirements.txt
└── README.md
```

## How to Run

```bash
# Clone the repository
git clone https://github.com/sabburi6/rideshare-demand-forecasting.git
cd rideshare-demand-forecasting

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebooks/rideshare_forecasting.ipynb
```

## Future Improvements

- Integrate station-level spatial data for location-specific predictions
- Add real-time weather forecast feeds for adaptive demand estimation
- Explore LSTM and other sequence models for capturing higher-order temporal dependencies
- Incorporate external context like city events (parades, sports games) as demand drivers

## Acknowledgments

Built as the DSE 501 Applied Data Science project at Arizona State University (2025).
