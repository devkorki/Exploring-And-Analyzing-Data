# Exploring and Analyzing Data – Final Project (ITC6002A1)

This repository contains the final project for the course **ITC6002A1 - Exploring and Analyzing Data** (Fall Term 2024) under the supervision of **Prof. Dimitrios Milioris**.

## 👨‍💻 Team Members
- Evangelos Aspiotis
- Panagiotis Korkizoglou
- Christos Liakopoulos

## 📌 Project Overview

We analyzed and forecasted **daily temperature data** from 2022–2023 using various statistical and time series forecasting methods.

The data is sourced from the **National Observatory of Athens / meteo.gr**, covering cities such as Athens, Ioannina, Thessaloniki, and more.

---

## 📂 Project Structure

- `ITC6002 Project Team 1.py`: Main Python script containing all data processing, forecasting methods, and evaluation.
- `data/`: Folder containing raw `.csv` datasets for each city.

---

## 📈 Techniques Used

### ➤ Data Cleaning
- Handling missing values (`---`) using moving averages
- Outlier detection using interquartile range
- Forward/backward filling of gaps

### ➤ Forecasting Methods
- **Moving Average (MA)**
- **Weighted Moving Average (WMA)**
- **Exponential Smoothing (ES)**
- **Trend-adjusted Forecasting using Linear Regression**
- **Trend + Seasonality using Adaptive Multiplicative Model**

Each method includes **parameter optimization** based on **MAPE (Mean Absolute Percentage Error)**.

---

## 📊 Questions Addressed

1. **Data Cleaning**: Cleaned raw temperature datasets from multiple cities.
2. **Year 1 Forecasting**: Predicted February–December based on January using multiple methods.
3. **Trend Analysis**: Estimated linear trend and adjusted forecasts.
4. **Seasonality Detection**: Applied seasonality-adjusted forecasting models.
5. **Year 2 Validation**: Repeated analysis using the second year to validate models.
6. **Cross-city Comparison**: Compared performance across 10+ cities using monthly snapshots.

---

## ⚙️ How to Run

1. Install dependencies:
   ```bash
   pip install requests pandas numpy matplotlib
