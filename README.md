# 📊 Quantitative Risk Management System

## Overview
This project is a full quantitative finance risk engine for portfolio analysis, combining statistical modeling, risk measurement, and portfolio optimization.

It includes:
- Volatility modeling (GARCH)
- Risk estimation (VaR / CVaR)
- Extreme Value Theory (EVT)
- Monte Carlo simulation
- Portfolio optimization (Markowitz & Black-Litterman)
- Factor modeling (Fama-French)
- Dependency modeling (Correlation & Copula)
- Risk backtesting

---

## ⚙️ Features

### 📉 Risk Models
- Historical VaR & CVaR
- Monte Carlo VaR (Normal distribution)
- GARCH(1,1) volatility model
- GARCH Monte Carlo (Student-t distribution)
- Extreme Value Theory (GPD tail risk)

### 📊 Portfolio Optimization
- Markowitz Mean-Variance Optimization
- Black-Litterman model with investor views

### 📈 Performance Metrics
- Sharpe Ratio
- Sortino Ratio
- Treynor Ratio
- Information Ratio
- Beta / Alpha

### 🔗 Dependency Analysis
- Correlation matrix
- Rolling correlation
- Copula tail dependence

### 🧪 Validation
- Kupiec VaR backtesting

---

## 📁 Data Source

Data is loaded from a local SQLite database.

