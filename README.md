# Quantitative Risk Management System  
Institutional-Grade Portfolio Risk Engine using GARCH, EVT, Copula & Factor Models  

---

## 📌 Overview  

This project implements a full-scale quantitative risk management system designed to replicate institutional portfolio risk frameworks used in hedge funds and asset management firms.  

The system models and integrates:  
- Volatility clustering (GARCH)  
- Tail risk behavior (Extreme Value Theory)  
- Non-linear dependency structures (Copula models)  
- Portfolio optimization (Markowitz & Black-Litterman)  
- Multi-factor asset pricing (Fama-French model)  
- Risk model validation (VaR backtesting)  

It provides a complete end-to-end pipeline from raw market data → risk estimation → portfolio construction → validation.

---

## 🎯 Objective  

The goal of this system is to:  

- Improve risk estimation beyond Gaussian assumptions  
- Capture fat-tail behavior in financial returns  
- Model dynamic volatility and correlation structures  
- Construct optimized portfolios under realistic constraints  
- Validate risk models using statistical backtesting  

---

## 🏗 System Architecture  

Market Data (SQLite)  
        ↓  
Return Computation Engine  
        ↓  
Risk Modeling Layer:  
    - GARCH(1,1) Volatility Model  
    - Extreme Value Theory (EVT)  
    - Monte Carlo Simulation (Normal & Student-t)  
    - Copula Dependency Modeling  
        ↓  
Portfolio Construction:  
    - Markowitz Optimization  
    - Black-Litterman Model  
        ↓  
Risk Validation:  
    - VaR Backtesting (Kupiec Test)  
        ↓  
Outputs:  
    - 10 Risk Visualization Plots  
    - Excel Risk Report  

---

## ⚙️ Key Features  

### 📊 Risk Modeling  
- GARCH(1,1) volatility estimation  
- Monte Carlo VaR simulation (Normal & Student-t)  
- Extreme Value Theory (Generalized Pareto Distribution)  
- Copula-based tail dependence modeling  

### 📈 Portfolio Optimization  
- Mean-Variance Optimization (Markowitz)  
- Black-Litterman model with investor views  
- Constrained portfolio optimization  

### 🧠 Factor Modeling  
- Fama-French 3-Factor regression  
- Market, size (SMB), and value (HML) exposures  

### 📉 Risk Validation  
- Historical VaR  
- Monte Carlo VaR  
- EVT-based VaR  
- Kupiec backtesting framework  

### 🔄 Risk Dynamics  
- Rolling volatility analysis  
- Dynamic correlation tracking  
- GARCH persistence estimation  

---

## 🔑 Key Insight  

Traditional risk models assume normality and underestimate extreme market movements.  

This system improves risk estimation by combining:  
- Volatility clustering (GARCH)  
- Extreme tail events (EVT)  
- Dependency modeling (Copulas)  

This results in a more realistic representation of portfolio risk under stress conditions.

---

## 📦 Installation  

pip install numpy pandas matplotlib seaborn scipy scikit-learn arch openpyxl  

---

## 🚀 Execution  

python main.py  

---

## 📁 Project Structure  

Quant-Risk-System/  
│  
├── main.py  
├── README.md  
├── requirements.txt  
├── portfolio.db  
│  
├── Plots/  
│   ├── 1_var_comparison.png  
│   ├── 2_risk_return_profile.png  
│   ├── 3_correlation_heatmap.png  
│   ├── 4_fama_french_factors.png  
│   ├── 5_portfolio_weights.png  
│   ├── 6_var_backtesting.png  
│   ├── 7_dynamic_correlation.png  
│   ├── 8_evt_tail_shapes.png  
│   ├── 9_garch_persistence.png  
│   └── 10_copula_tail_dependence.png  
│  
└── Portfolio_Report.xlsx  

---

## 📊 Outputs  

### 📈 Risk Metrics  
- Sharpe Ratio  
- Alpha / Beta  
- VaR (95% / 99%)  
- CVaR (Expected Shortfall)  
- Max Drawdown  

### 📉 Model Outputs  
- GARCH volatility estimates  
- EVT tail risk parameters  
- Copula dependency matrix  
- Fama-French factor loadings  

### 📁 Reports  
- Excel multi-sheet risk report  
- 10 visualization plots  

---

## 📉 Visualization Suite  

1. VaR Comparison (Historical vs Monte Carlo vs GARCH-MC)  
2. Risk-Return Profile  
3. Correlation Heatmap  
4. Fama-French Factor Exposure  
5. Portfolio Weights Comparison  
6. VaR Backtesting Results  
7. Dynamic Correlation Over Time  
8. EVT Tail Risk Analysis  
9. GARCH Persistence Analysis  
10. Copula Tail Dependence  

---

## 🧠 Use Case  

This system can be used for:  

- Portfolio risk management  
- Stress testing under extreme market conditions  
- Quantitative research and modeling  
- Academic finance research projects  
- Institutional risk framework simulation  

---

## 📌 Note  

This project is built as a quantitative research simulation of institutional risk management systems used in hedge funds and asset management firms. It demonstrates advanced financial modeling techniques including volatility modeling, tail risk estimation, and portfolio optimization under uncertainty.
