# Realized Volatility Change Forecasting

This repository contains a quantitative research framework for forecasting changes in realized volatility using machine learning and econometric models. The methodology focuses on predicting the log-difference of future mean realized volatility from observable past data.

## Overview

The framework implements a comprehensive volatility forecasting system that:
- Constructs daily realized volatility from high-frequency (15-minute) price data
- Engineers features from the volatility time series using technical indicators and time-series analysis
- Evaluates multiple forecasting models including linear, tree-based, neural network, and econometric benchmarks
- Performs regime-based analysis to assess model performance across different volatility environments

**Note**: This is a research implementation for academic and methodological purposes. It does not include trading logic, execution assumptions, or profitability claims.

## Forecasting Target

The target variable is the **log-difference of future mean realized volatility**:

```
rv_logdiff_{t} = log(mean(RV_{t+1:t+H})) - log(RV_t)
```

where:
- `RV_t` is the daily realized volatility at time `t`
- `mean(RV_{t+1:t+H})` is the mean of realized volatility over the next `H` days (default: 7 days)
- The log-difference transformation provides a more stable target for regression

The framework also computes the simple difference `rv_diff = mean(RV_{t+1:t+H}) - RV_t` for evaluation purposes, but the primary training target is the log-difference.

## Data Construction

### Realized Volatility Computation

1. **High-Frequency Returns**: Log returns are computed from 15-minute price data:
   ```
   ret_{15m} = log(close_t) - log(close_{t-1})
   ```

2. **Daily Aggregation**: Realized variance is computed by summing squared 15-minute returns within each calendar day:
   ```
   RV^2_{daily} = Σ(ret_{15m}^2) over all 15-min bars in day
   ```

3. **Realized Volatility**: Daily RV is the square root of realized variance, annualized:
   ```
   RV_{daily} = sqrt(RV^2_{daily}) × sqrt(252)
   ```

4. **Data Quality Filters**:
   - Days with fewer than 20 fifteen-minute bars are excluded (partial trading days)
   - Weekends are filtered out
   - Days with zero realized volatility are excluded (invalid data)

The data spans approximately 10 years of SPY (S&P 500 ETF) trading data from 2015 to 2025, with timezone conversion to America/New_York for proper trading day alignment.

## Feature Engineering

Features are constructed exclusively from past data available at time `t` to ensure no lookahead bias. The feature families include:

### Lagged Realized Volatility Features
- Lagged RV levels at multiple horizons (t-1, t-2, t-3, t-5, t-10)
- RV changes (differences between current and lagged values)
- Rolling means and standard deviations of past RV

### Volatility Ratios
- Ratio of current RV to RV from 20 days ago
- Ratio of RV from 2 days ago to RV from 14 days ago
- Ratio of current RV to rolling 21-day mean

### Technical Indicators on Volatility Path
- **RSI (Relative Strength Index)**: Computed on the volatility series with a 14-day period
- **Autocorrelation**: Rolling autocorrelation at lags 1, 2, 3, 5, and 10 computed on the volatility path

### Time-Based Features
- Day of week (0-6)
- Week of month (1-5)
- Month of year (1-12)

### Historical Ranks
- Historical percentile ranks of RSI, RV ratios, and RV-to-mean ratios (computed using expanding windows)

### Correlation Features
- Rolling correlation between RV ratios and current RV
- Rolling correlation between volatility and ratio-based features

### Frequency-Domain Features
- **Fourier Analysis**: Dominant frequencies and amplitudes from rolling FFT analysis (63-day window)
- Total spectral energy

### Time-Series Model Features
- **SARIMA Forecasts**: One-step-ahead forecasts from rolling SARIMA(1,1,1)×(1,1,1,5) models fitted on 252-day windows
- SARIMA residual standard deviation

All features are computed using rolling windows that end at time `t`, ensuring strict temporal ordering and no information leakage.

## Models

The framework evaluates six forecasting approaches:

1. **Baseline Linear Regression**: Ordinary least squares regression as a simple linear benchmark
2. **Random Forest**: Ensemble of regression trees with hyperparameter tuning (max depth, min samples, number of estimators, feature sampling)
3. **HAR-RV (Heterogeneous Autoregressive - Realized Volatility)**: Econometric benchmark using lagged daily, weekly, and monthly RV aggregates
4. **HAR-X**: Extension of HAR-RV incorporating lagged returns and jump measures
5. **Neural Network**: Shallow multi-layer perceptron (MLP) with architecture: Input → [128, 64, 32, 16] → Output, used as a nonlinear benchmark
6. **Ensemble**: Simple average of all model predictions

Hyperparameter tuning for Random Forest is performed using random search on the training set, with final model selection based on validation set performance. The Neural Network uses early stopping based on validation loss with a patience of 50 epochs.

## Train/Validation/Test Split

The data is split chronologically into:
- **Training**: 60% (approximately 1,352 days, 2016-11-18 to 2022-04-04)
- **Validation**: 20% (approximately 450 days, 2022-04-05 to 2024-01-19)
- **Test**: 20% (approximately 452 days, 2024-01-22 to 2025-11-06)

All preprocessing, feature scaling, and model fitting are performed exclusively on the training set. Validation set is used only for hyperparameter selection and early stopping. The test set is held out completely until final evaluation.

## Evaluation Metrics

Models are evaluated using multiple metrics computed on the test set:

### Primary Metrics (on rv_logdiff)
- **RMSE_logdiff**: Root mean squared error on log-difference predictions
- **MAE_logdiff**: Mean absolute error on log-difference predictions
- **R²**: Coefficient of determination on log-difference predictions
- **Correlation**: Pearson correlation coefficient between predicted and actual log-differences

### Secondary Metrics (on rv_diff)
- **RMSE_diff**: Root mean squared error on simple difference predictions
- **MAE_diff**: Mean absolute error on simple difference predictions

### Directional Accuracy
- **Directional_Acc**: Percentage of correct sign predictions on the simple difference (rv_diff), excluding zero changes

## Results Summary

Test set performance (as reported in notebook outputs):

| Model | R² | RMSE_logdiff | MAE_logdiff | Correlation | Directional_Acc |
|-------|----|--------------|-------------|-------------|-----------------|
| RandomForest | 0.3890 | 0.336087 | 0.260667 | 0.632851 | 0.730679 |
| Baseline Linear | 0.3295 | 0.349065 | 0.267968 | 0.581479 | 0.758782 |
| Neural Network | 0.3184 | 0.351937 | 0.272594 | 0.566421 | 0.733021 |
| Ensemble | 0.3173 | 0.352209 | 0.271840 | 0.621662 | 0.749415 |
| HAR-X | 0.0672 | 0.411698 | 0.320587 | 0.298917 | 0.601874 |
| HAR-RV | 0.0509 | 0.415287 | 0.324204 | 0.244184 | 0.580796 |

**Key Observations**:
- Random Forest achieves the highest R² (0.3890) and lowest RMSE on the test set
- Linear regression performs competitively, suggesting the relationship is largely linear
- HAR-RV and HAR-X benchmarks show weak performance (R² < 0.07), indicating limited predictive power from simple lagged RV aggregates alone
- Ensemble performance is similar to individual models, suggesting limited complementarity
- All models show moderate correlation (0.24-0.63) between predictions and actuals

## Regime-Based Analysis

The framework includes two types of regime analysis:

### 1. Future RV Quantile Analysis
Models are evaluated separately in low, medium, and high volatility regimes based on the **realized future RV level**. Quantile thresholds (33rd and 67th percentiles) are computed on the training set and applied to the test set to avoid lookahead bias.

### 2. Current Regime Conditioning Analysis
Model performance is conditioned on the **current observable volatility regime** at time `t`, defined using a rolling 5-day mean of daily RV. Regimes (low, medium, high) are determined using 33rd and 67th percentile thresholds computed exclusively on training data. For each model and regime, the analysis reports:
- Precision and recall for predicting high volatility increases
- RMSE within each regime
- Frequency of high-volatility predictions

This analysis addresses the question of whether models perform differently in different observable market conditions, which is relevant for practical deployment.

## Limitations and Disclaimers

1. **Single Asset**: The analysis is performed on SPY (S&P 500 ETF) only. Results may not generalize to other assets or markets.

2. **No Trading Logic**: This framework is purely predictive and does not include:
   - Position sizing
   - Entry/exit rules
   - Transaction costs
   - Slippage assumptions
   - Risk management

3. **No Profitability Claims**: The evaluation metrics (R², RMSE, correlation) measure forecast accuracy, not trading profitability. No claims are made about the economic value or tradability of these forecasts.

4. **Lookback Period**: The analysis uses approximately 10 years of data. Model performance may vary in different market regimes not represented in this sample.

5. **Feature Engineering**: Features are designed based on domain knowledge and may not be exhaustive. Alternative feature sets or transformations may yield different results.

6. **Model Assumptions**: All models assume stationarity and may not capture structural breaks or regime changes not explicitly modeled.

7. **Computational Considerations**: SARIMA and Fourier features are computationally intensive. The framework includes toggles to disable these features for faster iteration.

## File Structure

- `alpha2_3_paper.ipynb`: Main research notebook containing all analysis
- `saved_models/`: Directory containing trained model artifacts and metadata
- `*.png`: Visualization outputs (comparison plots, feature importance, residual diagnostics)

## Usage

The notebook is self-contained and can be executed cell-by-cell. Key parameters can be adjusted at the top of the notebook:
- `ROLLING_WINDOW`: Rolling window size for predictor features (default: 30 days)
- `FORECAST_TARGET`: Forecast horizon in days (default: 7 days)
- `ENABLE_SARIMA` / `ENABLE_FOURIER`: Toggles for computationally intensive features

## Citation

If using this code for research, please cite appropriately and note that this is a methodological framework for volatility forecasting research, not a trading system.

