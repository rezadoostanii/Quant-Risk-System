# Quantitative Risk Management System
Advanced Portfolio Risk Engine using GARCH, EVT, Copula & Factor Models

---

## 📌 Project Objective

This project implements a full-scale quantitative risk management pipeline including:

- Portfolio performance analysis
- Volatility modeling (GARCH(1,1))
- Tail risk estimation (EVT - Generalized Pareto)
- Monte Carlo VaR (Normal & Student-t)
- Copula-based dependency modeling
- Markowitz & Black-Litterman optimization
- Fama-French factor regression
- VaR backtesting (Kupiec test)

---

## 🏗 System Architecture

Data (SQLite)
   ↓
Return Engine
   ↓
Risk Models:
   - GARCH
   - EVT
   - Monte Carlo
   - Copula
   ↓
Portfolio Construction:
   - Markowitz Optimization
   - Black-Litterman Model
   ↓
Validation Layer:
   - VaR Backtesting (Kupiec Test)
   ↓
Outputs:
   - 10 Visualization Plots
   - Excel Risk Report

---

## ⚙ Features

- Volatility modeling using GARCH(1,1)
- Tail risk estimation using Extreme Value Theory (EVT)
- Monte Carlo VaR simulation (Normal & Student-t)
- Copula-based tail dependence analysis
- Mean-Variance Optimization (Markowitz)
- Black-Litterman portfolio optimization
- Fama-French 3-factor regression model
- Kupiec VaR backtesting framework
- Rolling risk analysis and dynamic correlation
- Automated Excel reporting system
- 10 high-quality visualization plots

---

## 🧠 Why This Project Matters

This system is designed to replicate institutional-grade risk frameworks used in hedge funds and asset management firms for:

- Tail risk estimation beyond normal distribution assumptions
- Volatility clustering modeling via GARCH
- Dependency modeling via Copulas
- Stress testing portfolios under extreme market conditions
- Robust portfolio optimization under uncertainty

---

## 🔑 Key Insight

Traditional VaR models often underestimate tail risk due to normality assumptions.  
This system combines:

- EVT (Extreme Value Theory)
- GARCH volatility clustering
- Monte Carlo simulation

to better capture fat-tail behavior in financial returns.

---

## 📦 Installation

pip install numpy pandas matplotlib seaborn scipy scikit-learn arch openpyxl

---

## 🚀 Run Project

python main.py

---

## 📁 Project Structure

Quant-Risk-System/
│
├── main.py
├── README.md
├── requirements.txt
├── Plots/
│   ├── 1_var_comparison.png
│   ├── 2_risk_return_profile.png
│   ├── ...
│   └── 10_copula_tail_dependence.png
│
└── portfolio.db

---

## 📊 Outputs

- 10 risk visualization plots
- Excel report with:
  - Basic metrics
  - GARCH estimates
  - Monte Carlo VaR
  - EVT analysis
  - Portfolio optimization
  - Copula dependency structure

---

## 📈 Visualization Summary

All plots are saved automatically in `/Plots/`:

1. VaR Comparison (Historical vs MC vs GARCH-MC)
2. Risk-Return Profile
3. Correlation Heatmap
4. Fama-French Factor Exposure
5. Portfolio Weights
6. VaR Backtesting
7. Dynamic Correlation
8. EVT Tail Risk
9. GARCH Persistence
10. Copula Tail Dependence
