# EUR/USD-LSTM-ARIMA-MSc-Dissertation

# MSc Dissertation: A Deep Learning Approach to 
# Intraday EUR/USD Forecasting: Evaluating LSTM 
# Performance and Interpretability via SHAP

## Overview
This repository contains all Python scripts used 
in the empirical analysis for my MSc dissertation. 
The study evaluates whether an LSTM network can 
outperform a traditional ARIMA benchmark for 
one-step-ahead EUR/USD price forecasting at 
15-minute frequency, and applies SHAP-based 
interpretability analysis to explain the LSTM 
model's predictions.


---

## Repository Structure

| File | Description |
|---|---|
| AddFeatures.py | Resamples 1-minute data to 15-minute frequency and computes SMA-20, Bollinger Bands, and RSI-14 |
| TrainModel_LSTM.py | Builds and evaluates the LSTM model using five-fold walk-forward validation |
| ARIMA_Baseline_1.py | Implements rolling one-step-ahead ARIMA forecasting with checkpointing |
| Diebold_Mariano.py | Runs the Diebold-Mariano test to formally compare LSTM and ARIMA forecast accuracy |
| SHAP_Analysis.py | Applies KernelExplainer SHAP analysis to the trained LSTM model |
| Sanity_Check.py | Validates the data pipeline and model outputs through a series of assertions and checks |

---
