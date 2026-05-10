"""
================================================================================
QUANTITATIVE RISK MANAGEMENT SYSTEM
Complete Portfolio Analysis with GARCH, EVT, Copula
================================================================================
"""

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy.optimize import minimize
from pathlib import Path
from sklearn.covariance import LedoitWolf
from scipy.stats import norm, chi2, genpareto
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 50)

# =============================================================================
# CONFIGURATION - MODIFY FOR YOUR USE CASE
# =============================================================================
TICKERS = [
    'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN',
    'META', 'TSLA', 'JPM', 'JNJ', 'V',
    'PG', 'XOM', 'UNH', 'HD', 'MA'
]
BENCHMARK = '^GSPC'
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0425
CONFIDENCE_LEVEL = 0.95

# Paths - update based on your system
DB_PATH = Path.home() / "Downloads" / "portfolio.db"
DESKTOP_PATH = Path.home() / "Desktop"
EXCEL_PATH = DESKTOP_PATH / "Portfolio_Report.xlsx"

def display_df(title, df, max_rows=20):
    """Print dataframe to console"""
    print(f"\n{'='*80}")
    print(f"📊 {title}")
    print(f"{'='*80}")
    if len(df) > max_rows:
        print(df.head(max_rows))
        print(f"... and {len(df) - max_rows} more rows")
    else:
        print(df)
    print(f"{'='*80}\n")


# =============================================================================
# DATA LOADING
# =============================================================================
def get_data():
    """Load price data from SQLite and/or yfinance"""
    print("\n" + "="*80)
    print("📥 LOADING DATA FROM SQLITE")
    print("="*80)
    
    try:
        import yfinance as yf
    except:
        import subprocess
        subprocess.check_call(['pip', 'install', 'yfinance'])
        import yfinance as yf
    
    conn = sqlite3.connect(DB_PATH)
    prices_df = None
    
    for ticker in TICKERS:
        table = f"{ticker.lower()}_prices"
        try:
            df = pd.read_sql(f"SELECT Date, Close FROM {table}", conn)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.rename(columns={'Close': ticker}).set_index('Date')
            if prices_df is None:
                prices_df = df
            else:
                prices_df = prices_df.join(df, how='inner')
            print(f"  ✅ {ticker}")
        except:
            print(f"  ❌ {ticker}")
    
    # Load benchmark
    benchmark = None
    bench_tables = ["gspc_prices", "gspc_price", "GSPC_prices"]
    for bench_table in bench_tables:
        try:
            bench_df = pd.read_sql(f'SELECT Date, Close FROM {bench_table}', conn)
            bench_df['Date'] = pd.to_datetime(bench_df['Date'])
            bench_df = bench_df.rename(columns={'Close': BENCHMARK}).set_index('Date')
            benchmark = bench_df[BENCHMARK]
            print(f"  ✅ Benchmark found")
            break
        except:
            continue
    
    conn.close()
    
    if benchmark is None:
        print(f"  ⚠️ Downloading benchmark...")
        benchmark = yf.download(BENCHMARK, start=START_DATE, end=END_DATE, progress=False)['Close']
    
    common_dates = prices_df.index.intersection(benchmark.index)
    prices_df = prices_df.loc[common_dates]
    benchmark = benchmark.loc[common_dates]
    
    print(f"\n✅ Final: {len(prices_df.columns)} assets, {len(prices_df)} days")
    return prices_df, benchmark


# =============================================================================
# COMPLETE BASIC METRICS
# =============================================================================
def compute_metrics(prices, benchmark):
    """Calculate all basic risk and return metrics"""
    print("\n" + "="*80)
    print("📊 CALCULATING COMPLETE BASIC METRICS")
    print("="*80)
    
    returns = prices.pct_change().dropna()
    bench_returns = benchmark.pct_change().dropna()
    
    common = returns.index.intersection(bench_returns.index)
    returns = returns.loc[common]
    bench_returns = bench_returns.loc[common]
    
    metrics = pd.DataFrame(index=prices.columns)
    
    # RETURNS
    metrics['Total_Return_%'] = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    metrics['Annual_Return_%'] = (1 + metrics['Total_Return_%']/100) ** (252/len(prices)) * 100 - 100
    
    # RISK
    metrics['Annual_Volatility_%'] = returns.std() * np.sqrt(252) * 100
    
    # SHARPE RATIO
    excess_return = metrics['Annual_Return_%']/100 - RISK_FREE_RATE
    metrics['Sharpe_Ratio'] = excess_return / (metrics['Annual_Volatility_%']/100)
    
    # MAX DRAWDOWN
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max
    metrics['Max_Drawdown_%'] = drawdown.min() * 100
    
    # BETA & ALPHA
    betas, alphas = [], []
    bench_array = bench_returns.values.flatten()
    market_return = bench_returns.mean() * 252
    
    for ticker in returns.columns:
        stock_returns = returns[ticker].values
        cov = np.cov(stock_returns, bench_array)[0, 1]
        var_bench = np.var(bench_array)
        beta = cov / var_bench if var_bench > 0 else 1
        
        exp_return = RISK_FREE_RATE + beta * (market_return - RISK_FREE_RATE)
        alpha = (metrics.loc[ticker, 'Annual_Return_%']/100) - exp_return
        
        betas.append(beta)
        alphas.append(alpha * 100)
    
    metrics['Beta'] = betas
    metrics['Alpha_%'] = alphas
    
    # HISTORICAL VAR & CVAR
    metrics['Historical_VaR_95_%'] = returns.quantile(1 - CONFIDENCE_LEVEL) * 100
    
    cvar = []
    for col in returns.columns:
        var_val = metrics.loc[col, 'Historical_VaR_95_%'] / 100
        cvar.append(returns[returns[col] <= var_val][col].mean() * 100)
    metrics['Historical_CVaR_95_%'] = cvar
    
    # SORTINO RATIO
    downside = returns[returns < 0]
    metrics['Sortino_Ratio'] = (metrics['Annual_Return_%']/100 - RISK_FREE_RATE) / (downside.std() * np.sqrt(252))
    
    # M-SQUARED
    bench_vol = bench_returns.std() * np.sqrt(252)
    metrics['M_Squared_%'] = ((metrics['Sharpe_Ratio'] * bench_vol) + RISK_FREE_RATE) * 100
    
    # TREYNOR RATIO
    metrics['Treynor_Ratio'] = (metrics['Annual_Return_%']/100 - RISK_FREE_RATE) / metrics['Beta']
    
    # INFORMATION RATIO
    tracking_error = (returns.sub(bench_returns, axis=0)).std() * np.sqrt(252)
    metrics['Information_Ratio'] = (metrics['Annual_Return_%']/100 - market_return) / tracking_error
    
    print(f"✅ Calculated {len(metrics.columns)} metrics for {len(metrics)} assets")
    return metrics, returns, bench_returns


# =============================================================================
# GARCH(1,1) MODEL
# =============================================================================
def fit_garch(returns):
    """Fit GARCH(1,1) model for each asset"""
    print("\n" + "="*80)
    print("📐 GARCH(1,1) VOLATILITY MODELING")
    print("="*80)
    
    results = []
    for ticker in returns.columns:
        try:
            r = returns[ticker].dropna() * 100
            if len(r) > 100:
                model = arch_model(r, vol='GARCH', p=1, q=1, dist='normal')
                fitted = model.fit(disp='off')
                
                omega = fitted.params.get('omega', 0)
                alpha = fitted.params.get('alpha[1]', 0)
                beta = fitted.params.get('beta[1]', 0)
                persistence = alpha + beta
                
                unconditional_var = omega / (1 - alpha - beta) if (alpha + beta) < 1 else omega
                annual_vol = np.sqrt(unconditional_var * 252)
                
                results.append({
                    'Ticker': ticker,
                    'GARCH_Omega': omega,
                    'GARCH_Alpha': alpha,
                    'GARCH_Beta': beta,
                    'GARCH_Persistence': persistence,
                    'GARCH_Annual_Vol_%': annual_vol
                })
                print(f"  ✅ {ticker}: Persistence={persistence:.3f}")
        except:
            continue
    
    df = pd.DataFrame(results).set_index('Ticker')
    print(f"\n✅ GARCH fitted for {len(df)} assets")
    return df


# =============================================================================
# MONTE CARLO VAR
# =============================================================================
def mc_var_simulation(returns, n_sims=10000):
    """Monte Carlo VaR assuming normal distribution"""
    print("\n" + "="*80)
    print("🎲 MONTE CARLO VAR SIMULATION (Normal Distribution)")
    print(f"   Simulations: {n_sims:,}")
    print("="*80)
    
    results = []
    for ticker in returns.columns:
        hist_returns = returns[ticker].dropna().values
        mu = np.mean(hist_returns)
        sigma = np.std(hist_returns)
        simulated = np.random.normal(mu, sigma, n_sims)
        
        var_95 = np.percentile(simulated, 5) * 100
        var_99 = np.percentile(simulated, 1) * 100
        cvar_95 = simulated[simulated <= np.percentile(simulated, 5)].mean() * 100
        
        results.append({
            'Ticker': ticker,
            'MC_VaR_95_%': var_95,
            'MC_VaR_99_%': var_99,
            'MC_CVaR_95_%': cvar_95
        })
        print(f"  {ticker}: VaR95={var_95:.2f}%")
    
    return pd.DataFrame(results).set_index('Ticker')


# =============================================================================
# GARCH-MONTE CARLO WITH STUDENT-T
# =============================================================================
def garch_mc_studentt(returns, n_sims=20000):
    """GARCH-based Monte Carlo with Student-t distribution"""
    print("\n" + "="*80)
    print("🎲 GARCH-MONTE CARLO VAR (Student-t Distribution)")
    print(f"   Simulations: {n_sims:,}")
    print("="*80)
    
    results = []
    for ticker in returns.columns:
        try:
            r = returns[ticker].dropna().values * 100
            if len(r) < 100:
                continue
            
            model = arch_model(r, vol='GARCH', p=1, q=1, dist='t')
            fitted = model.fit(disp='off')
            
            mu = fitted.params.get('mu', np.mean(r))
            omega = fitted.params.get('omega', 0.01)
            alpha = fitted.params.get('alpha[1]', 0.05)
            beta = fitted.params.get('beta[1]', 0.90)
            nu = fitted.params.get('nu', 5)
            
            simulated = np.zeros(n_sims)
            sigma2 = omega / (1 - alpha - beta)
            
            for i in range(n_sims):
                z = np.random.standard_t(df=nu, size=1)[0]
                simulated[i] = mu + np.sqrt(sigma2) * z
                sigma2 = omega + alpha * (z**2) + beta * sigma2
            
            var_95 = np.percentile(simulated, 5)
            var_99 = np.percentile(simulated, 1)
            cvar_95 = simulated[simulated <= var_95].mean()
            
            results.append({
                'Ticker': ticker,
                'GARCH_MC_VaR_95_%': var_95,
                'GARCH_MC_VaR_99_%': var_99,
                'GARCH_MC_CVaR_95_%': cvar_95,
                'GARCH_DoF': nu
            })
            print(f"  {ticker}: VaR95={var_95:.2f}%, DoF={nu:.1f}")
        except:
            continue
    
    return pd.DataFrame(results).set_index('Ticker')


# =============================================================================
# EXTREME VALUE THEORY (EVT)
# =============================================================================
def extreme_value_analysis(returns):
    """Extreme Value Theory using GPD for tail risk"""
    print("\n" + "="*80)
    print("⚠️ EXTREME VALUE THEORY (EVT) - GPD")
    print("="*80)
    
    results = []
    for ticker in returns.columns:
        returns_series = returns[ticker].dropna().values
        losses = -returns_series[returns_series < 0]
        
        if len(losses) < 50:
            continue
        
        threshold_val = np.percentile(losses, 95)
        exceedances = losses[losses >= threshold_val] - threshold_val
        
        if len(exceedances) < 10:
            continue
        
        try:
            params = genpareto.fit(exceedances, floc=0)
            shape, loc, scale = params
        except:
            continue
        
        n = len(losses)
        nu = len(exceedances)
        
        row = {'Ticker': ticker, 'EVT_Threshold_%': threshold_val * 100, 'EVT_Shape_(Xi)': shape}
        
        for conf in [0.95, 0.99]:
            p = conf
            if shape != 0:
                var_evt = threshold_val + (scale / shape) * (((n / nu) * (1 - p)) ** (-shape) - 1)
            else:
                var_evt = threshold_val + scale * np.log((n / nu) * (1 - p))
            row[f'EVT_VaR_{int(conf*100)}_%'] = var_evt * 100
        
        results.append(row)
        print(f"  {ticker}: EVT VaR99={row['EVT_VaR_99_%']:.2f}%")
    
    if len(results) == 0:
        print("  ⚠️ No EVT results computed")
    return pd.DataFrame(results).set_index('Ticker')


# =============================================================================
# MARKOWITZ MEAN-VARIANCE OPTIMIZATION
# =============================================================================
def markowitz_optimization(returns):
    """Mean-variance optimization with weight constraints"""
    print("\n" + "="*80)
    print("📈 MEAN-VARIANCE OPTIMIZATION (Markowitz)")
    print("="*80)
    
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    n_assets = len(mu)
    
    def port_return(w): return np.sum(w * mu)
    def port_vol(w): return np.sqrt(np.dot(w.T, np.dot(cov, w)))
    def neg_sharpe(w): return -(port_return(w) - RISK_FREE_RATE) / port_vol(w)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 0.3) for _ in range(n_assets))
    result = minimize(neg_sharpe, np.ones(n_assets)/n_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    
    weights = result.x
    opt_results = pd.DataFrame({
        'Asset': mu.index,
        'Markowitz_Weight_%': weights * 100,
        'Annual_Return_%': mu.values * 100,
        'Annual_Volatility_%': np.sqrt(np.diag(cov)) * 100
    }).sort_values('Markowitz_Weight_%', ascending=False)
    opt_results = opt_results[opt_results['Markowitz_Weight_%'] > 0.01]
    
    portfolio_metrics = {
        'Return_%': port_return(weights) * 100,
        'Volatility_%': port_vol(weights) * 100,
        'Sharpe': (port_return(weights) - RISK_FREE_RATE) / port_vol(weights),
        'Assets': len(opt_results)
    }
    
    print(f"  ✅ Return: {portfolio_metrics['Return_%']:.2f}%")
    print(f"  ✅ Sharpe: {portfolio_metrics['Sharpe']:.3f}")
    print(f"  ✅ Assets: {portfolio_metrics['Assets']}")
    
    return opt_results, portfolio_metrics


# =============================================================================
# BLACK-LITTERMAN OPTIMIZATION
# =============================================================================
def black_litterman(returns):
    """Black-Litterman model with subjective views"""
    print("\n" + "="*80)
    print("🏦 BLACK-LITTERMAN OPTIMIZATION")
    print("="*80)
    
    mu_hist = returns.mean() * 252
    lw = LedoitWolf()
    lw.fit(returns)
    cov_shrunk = lw.covariance_ * 252
    
    n_assets = len(mu_hist)
    market_weights = np.ones(n_assets) / n_assets
    delta = 2.5
    pi = delta * cov_shrunk @ market_weights
    
    # View: NVDA expected to outperform AAPL
    nvda_idx = list(mu_hist.index).index('NVDA') if 'NVDA' in mu_hist.index else 0
    aapl_idx = list(mu_hist.index).index('AAPL') if 'AAPL' in mu_hist.index else 1
    P = np.zeros((1, n_assets))
    P[0, nvda_idx] = 1
    P[0, aapl_idx] = -1
    Q = np.array([0.10])  # View: NVDA 10% excess return
    
    tau = 0.05
    Omega = tau * (P @ cov_shrunk @ P.T)
    
    inv_cov = np.linalg.inv(cov_shrunk)
    inv_omega = np.linalg.inv(Omega)
    mu_bl = np.linalg.inv(inv_cov + P.T @ inv_omega @ P) @ (inv_cov @ pi + P.T @ inv_omega @ Q)
    cov_bl = np.linalg.inv(inv_cov + P.T @ inv_omega @ P)
    
    def neg_sharpe_bl(w): return -((np.sum(w * mu_bl) - RISK_FREE_RATE) / np.sqrt(w.T @ cov_bl @ w))
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 0.2) for _ in range(n_assets))
    result = minimize(neg_sharpe_bl, market_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    weights = result.x
    opt_results = pd.DataFrame({
        'Asset': mu_hist.index,
        'BL_Weight_%': weights * 100,
        'BL_Implied_Return_%': mu_bl * 100
    }).sort_values('BL_Weight_%', ascending=False)
    opt_results = opt_results[opt_results['BL_Weight_%'] > 0.01]
    
    portfolio_metrics = {
        'Return_%': np.sum(weights * mu_bl) * 100,
        'Volatility_%': np.sqrt(weights.T @ cov_bl @ weights) * 100,
        'Sharpe': (np.sum(weights * mu_bl) - RISK_FREE_RATE) / np.sqrt(weights.T @ cov_bl @ weights),
        'Assets': len(opt_results)
    }
    
    print(f"  ✅ Return: {portfolio_metrics['Return_%']:.2f}%")
    print(f"  ✅ Sharpe: {portfolio_metrics['Sharpe']:.3f}")
    
    return opt_results, portfolio_metrics


# =============================================================================
# FAMA-FRENCH 3-FACTOR
# =============================================================================
def fama_french_3factor(returns, bench_returns):
    """Fama-French 3-factor model (SMB and HML simulated)"""
    print("\n" + "="*80)
    print("🏛️ FAMA-FRENCH 3-FACTOR MODEL")
    print("="*80)
    
    results = []
    for ticker in returns.columns:
        stock_returns = returns[ticker].values
        market_returns = bench_returns.values
        
        # Construct SMB and HML factors (simulated)
        np.random.seed(42)
        smb = np.random.normal(0, 0.01, len(stock_returns)) + 0.3 * market_returns
        hml = np.random.normal(0, 0.01, len(stock_returns)) - 0.2 * market_returns
        
        X = np.column_stack([np.ones(len(stock_returns)), market_returns, smb, hml])
        beta, _, _, _ = np.linalg.lstsq(X, stock_returns, rcond=None)
        
        results.append({
            'Ticker': ticker,
            'FF_Alpha_%': beta[0] * 252 * 100,
            'FF_Beta_Market': beta[1],
            'FF_Beta_SMB': beta[2],
            'FF_Beta_HML': beta[3]
        })
    
    df = pd.DataFrame(results).set_index('Ticker')
    print(f"  ✅ Fitted for {len(df)} assets")
    return df


# =============================================================================
# VAR BACKTESTING (KUPIEC TEST)
# =============================================================================
def backtest_var(returns, var_forecasts):
    """Kupiec test for VaR model validation"""
    print("\n" + "="*80)
    print("📊 VAR BACKTESTING (Kupiec Test)")
    print("="*80)
    
    results = []
    for ticker in returns.columns:
        if ticker not in var_forecasts.index:
            continue
        
        actual = returns[ticker].dropna().values
        var_95 = var_forecasts.loc[ticker, 'GARCH_MC_VaR_95_%'] / 100
        exceptions = (actual < var_95).sum()
        n_obs = len(actual)
        exp_rate = exceptions / n_obs
        expected = (1 - CONFIDENCE_LEVEL) * n_obs
        
        p_hat = exp_rate
        p_star = 1 - CONFIDENCE_LEVEL
        
        if p_hat > 0 and p_hat < 1 and p_star > 0 and p_star < 1:
            lr = -2 * (exceptions * np.log(p_star) + (n_obs - exceptions) * np.log(1 - p_star) -
                       (exceptions * np.log(p_hat) + (n_obs - exceptions) * np.log(1 - p_hat)))
        else:
            lr = 0
        
        p_value = 1 - chi2.cdf(lr, df=1) if lr > 0 else 0.5
        valid = p_value > 0.05
        
        results.append({
            'Ticker': ticker,
            'Exceptions': exceptions,
            'Expected': expected,
            'Rate_%': exp_rate * 100,
            'P_Value': p_value,
            'Valid': valid
        })
        print(f"  {ticker}: {'✅' if valid else '❌'} Exceptions={exceptions}/{int(expected)}")
    
    return pd.DataFrame(results).set_index('Ticker')


# =============================================================================
# ROLLING ANALYSIS
# =============================================================================
def rolling_risk_analysis(returns, window=60):
    """Rolling volatility and correlation analysis"""
    print("\n" + "="*80)
    print(f"🔄 ROLLING ANALYSIS (Window: {window})")
    print("="*80)
    
    results = {}
    for ticker in returns.columns:
        vols, sharps = [], []
        for i in range(window, len(returns)):
            w = returns.iloc[i-window:i]
            vol = w[ticker].std() * np.sqrt(252) * 100
            ret = w[ticker].mean() * 252
            sharpe = (ret - RISK_FREE_RATE) / (vol/100) if vol > 0 else 0
            vols.append(vol)
            sharps.append(sharpe)
        
        if vols:
            results[ticker] = {
                'Current_Vol_%': vols[-1],
                'Current_Sharpe': sharps[-1] if sharps else 0,
                'Trend': 'Increasing' if len(vols) > 20 and vols[-1] > np.mean(vols[-20:]) else 'Decreasing'
            }
            print(f"  {ticker}: Vol={results[ticker]['Current_Vol_%']:.1f}%, {results[ticker]['Trend']}")
        else:
            results[ticker] = {'Current_Vol_%': 0, 'Current_Sharpe': 0, 'Trend': 'Unknown'}

    # Rolling correlation
    dyn_corr = []
    for i in range(window, len(returns)):
        w = returns.iloc[i-window:i]
        corr = w.corr()
        avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
        dyn_corr.append(avg_corr)
    
    return results, dyn_corr


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================
def correlation_analysis(returns):
    """Full correlation matrix and average correlation"""
    print("\n" + "="*80)
    print("🔄 CORRELATION ANALYSIS")
    print("="*80)
    
    corr = returns.corr()
    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    print(f"  Average Correlation: {avg_corr:.3f}")
    return corr


# =============================================================================
# COPULA TAIL DEPENDENCE
# =============================================================================
def copula_tail_dependence(returns):
    """Measure tail dependence using copula approach"""
    print("\n" + "="*80)
    print("🎲 COPULA DEPENDENCY MODELING")
    print("="*80)
    
    n_assets = len(returns.columns)
    tail_dep = pd.DataFrame(index=returns.columns, columns=returns.columns, dtype=float)
    
    for i, a1 in enumerate(returns.columns):
        for j, a2 in enumerate(returns.columns):
            if i == j:
                tail_dep.loc[a1, a2] = 1.0
                continue
            
            # Transform to uniform via normal CDF
            u1 = norm.cdf((returns[a1] - returns[a1].mean()) / returns[a1].std())
            u2 = norm.cdf((returns[a2] - returns[a2].mean()) / returns[a2].std())
            
            threshold = 0.95
            upper_tail = np.mean((u1 > threshold) & (u2 > threshold)) / (1 - threshold)
            lower_tail = np.mean((u1 < 1-threshold) & (u2 < 1-threshold)) / (1 - threshold)
            
            tail_dep.loc[a1, a2] = float(max(upper_tail, lower_tail))
    
    values = tail_dep.values[np.triu_indices_from(tail_dep.values, k=1)]
    values = values[~np.isnan(values)]
    print(f"  ✅ Average Tail Dependence: {np.mean(values):.4f}")
    
    return tail_dep.astype(float)


# =============================================================================
# ALL PLOTS (10 PLOTS)
# =============================================================================
def generate_plots(metrics, returns, mc_var, garch_mc_var, evt_var, mw_weights, 
                   bl_weights, ff_model, corr_matrix, backtest_df, dyn_corr, 
                   copula_tail, garch_df):
    """Generate all 10 analysis plots"""
    print("\n" + "="*80)
    print("📈 CREATING ALL 10 PLOTS")
    print("="*80)
    
    tickers = metrics.index.tolist()
    
    # PLOT 1: VaR Comparison
    fig1, ax = plt.subplots(figsize=(14, 7))
    hist_var = [metrics.loc[t, 'Historical_VaR_95_%'] for t in tickers]
    mc_dict = dict(zip(mc_var.index, mc_var['MC_VaR_95_%']))
    mc_vals = [mc_dict.get(t, np.nan) for t in tickers]
    garch_dict = dict(zip(garch_mc_var.index, garch_mc_var['GARCH_MC_VaR_95_%']))
    garch_vals = [garch_dict.get(t, np.nan) for t in tickers]
    
    x = np.arange(len(tickers))
    width = 0.25
    ax.bar(x - width, hist_var, width, label='Historical VaR 95%', color='steelblue', alpha=0.8, edgecolor='black')
    ax.bar(x, mc_vals, width, label='MC VaR 95% (Normal)', color='crimson', alpha=0.8, edgecolor='black')
    ax.bar(x + width, garch_vals, width, label='GARCH-MC VaR 95% (Student-t)', color='forestgreen', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Assets', fontsize=12)
    ax.set_ylabel('VaR (%)', fontsize=12)
    ax.set_title('VaR Comparison: Historical vs MC Normal vs GARCH-MC Student-t', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(DESKTOP_PATH / "1_var_comparison.png", dpi=300)
    plt.close(fig1)
    print("  ✅ 1_var_comparison.png")
    
    # PLOT 2: Risk-Return Profile
    fig2, ax = plt.subplots(figsize=(12, 8))
    vols = [metrics.loc[t, 'Annual_Volatility_%'] for t in tickers]
    rets = [metrics.loc[t, 'Annual_Return_%'] for t in tickers]
    sharpes = [metrics.loc[t, 'Sharpe_Ratio'] for t in tickers]
    
    scatter = ax.scatter(vols, rets, c=sharpes, cmap='RdYlGn', s=250, alpha=0.7, edgecolors='black')
    for i, t in enumerate(tickers):
        ax.annotate(t, (vols[i], rets[i]), fontsize=9, fontweight='bold', ha='center')
    
    ax.set_xlabel('Annual Volatility (%)')
    ax.set_ylabel('Annual Return (%)')
    ax.set_title('Risk-Return Profile (Color = Sharpe Ratio)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Sharpe Ratio')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DESKTOP_PATH / "2_risk_return_profile.png", dpi=300)
    plt.close(fig2)
    print("  ✅ 2_risk_return_profile.png")
    
    # PLOT 3: Correlation Heatmap
    fig3, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(DESKTOP_PATH / "3_correlation_heatmap.png", dpi=300)
    plt.close(fig3)
    print("  ✅ 3_correlation_heatmap.png")
    
    # PLOT 4: Fama-French Factors
    fig4, ax = plt.subplots(figsize=(14, 7))
    ff_tickers = ff_model.index.tolist()[:10]
    beta_mkt = [ff_model.loc[t, 'FF_Beta_Market'] for t in ff_tickers]
    beta_smb = [ff_model.loc[t, 'FF_Beta_SMB'] for t in ff_tickers]
    beta_hml = [ff_model.loc[t, 'FF_Beta_HML'] for t in ff_tickers]
    
    x = np.arange(len(ff_tickers))
    width = 0.25
    ax.bar(x - width, beta_mkt, width, label='Market Beta', color='navy', alpha=0.8, edgecolor='black')
    ax.bar(x, beta_smb, width, label='SMB Beta', color='crimson', alpha=0.8, edgecolor='black')
    ax.bar(x + width, beta_hml, width, label='HML Beta', color='forestgreen', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Assets')
    ax.set_ylabel('Factor Loading')
    ax.set_title('Fama-French 3-Factor Model Exposures', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ff_tickers, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(DESKTOP_PATH / "4_fama_french_factors.png", dpi=300)
    plt.close(fig4)
    print("  ✅ 4_fama_french_factors.png")
    
    # PLOT 5: Portfolio Weights
    fig5, ax = plt.subplots(figsize=(12, 6))
    mw_dict = dict(zip(mw_weights['Asset'], mw_weights['Markowitz_Weight_%']))
    bl_dict = dict(zip(bl_weights['Asset'], bl_weights['BL_Weight_%']))
    
    all_assets = sorted(set(mw_dict.keys()) | set(bl_dict.keys()))
    mw_vals = [mw_dict.get(a, 0) for a in all_assets]
    bl_vals = [bl_dict.get(a, 0) for a in all_assets]
    
    x = np.arange(len(all_assets))
    width = 0.35
    ax.bar(x - width/2, mw_vals, width, label='Markowitz', color='steelblue', alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, bl_vals, width, label='Black-Litterman', color='coral', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Assets')
    ax.set_ylabel('Weight (%)')
    ax.set_title('Portfolio Weights: Markowitz vs Black-Litterman', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_assets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(DESKTOP_PATH / "5_portfolio_weights.png", dpi=300)
    plt.close(fig5)
    print("  ✅ 5_portfolio_weights.png")
    
    # PLOT 6: VaR Backtesting
    if len(backtest_df) > 0:
        fig6, ax = plt.subplots(figsize=(12, 6))
        colors = ['green' if v else 'red' for v in backtest_df['Valid']]
        ax.bar(backtest_df.index, backtest_df['Rate_%'], color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=5, color='blue', linestyle='--', linewidth=2, label='Expected (5%)')
        ax.set_xlabel('Assets')
        ax.set_ylabel('Exception Rate (%)')
        ax.set_title('VaR Backtesting Results (Kupiec Test)', fontsize=14, fontweight='bold')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(DESKTOP_PATH / "6_var_backtesting.png", dpi=300)
        plt.close(fig6)
        print("  ✅ 6_var_backtesting.png")
    
    # PLOT 7: Dynamic Correlation
    fig7, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dyn_corr, color='purple', linewidth=2)
    ax.axhline(y=np.mean(dyn_corr), color='red', linestyle='--', label=f'Mean: {np.mean(dyn_corr):.3f}')
    ax.set_xlabel('Time (Rolling Windows)')
    ax.set_ylabel('Average Correlation')
    ax.set_title('Dynamic Correlation Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DESKTOP_PATH / "7_dynamic_correlation.png", dpi=300)
    plt.close(fig7)
    print("  ✅ 7_dynamic_correlation.png")
    
    # PLOT 8: EVT Tail Shapes
    if len(evt_var) > 0:
        fig8, ax = plt.subplots(figsize=(12, 6))
        shapes = evt_var['EVT_Shape_(Xi)']
        colors = ['red' if s > 0 else 'blue' for s in shapes]
        ax.bar(evt_var.index, shapes, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linewidth=1)
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Heavy Tail Threshold')
        ax.set_xlabel('Assets')
        ax.set_ylabel('EVT Shape Parameter (ξ)')
        ax.set_title('Extreme Value Theory - Tail Shape Parameters', fontsize=14, fontweight='bold')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(DESKTOP_PATH / "8_evt_tail_shapes.png", dpi=300)
        plt.close(fig8)
        print("  ✅ 8_evt_tail_shapes.png")
    
    # PLOT 9: GARCH Persistence
    fig9, ax = plt.subplots(figsize=(12, 6))
    if 'GARCH_Persistence' in garch_df.columns:
        persistence = garch_df['GARCH_Persistence']
        tickers_garch = persistence.index.tolist()
        
        colors = ['red' if p > 0.98 else 'orange' if p > 0.95 else 'green' for p in persistence]
        
        ax.bar(tickers_garch, persistence, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='High Persistence (0.95)')
        ax.axhline(y=0.98, color='red', linestyle='--', linewidth=2, label='Very High (0.98)')
        
        ax.set_xlabel('Assets', fontsize=12)
        ax.set_ylabel('GARCH Persistence (α+β)', fontsize=12)
        ax.set_title('GARCH(1,1) Volatility Persistence', fontsize=14, fontweight='bold')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, p in enumerate(persistence):
            ax.text(i, p + 0.005, f'{p:.3f}', ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(DESKTOP_PATH / "9_garch_persistence.png", dpi=300)
        plt.close(fig9)
        print("  ✅ 9_garch_persistence.png")
    else:
        print("  ⚠️ GARCH_Persistence not found")
        plt.close(fig9)
    
    # PLOT 10: Copula Tail Dependence
    fig10, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(copula_tail, annot=True, fmt='.2f', cmap='Reds', 
                center=0.3, square=True, linewidths=0.5, 
                cbar_kws={'label': 'Tail Dependence Coefficient'}, ax=ax)
    ax.set_title('Copula Tail Dependence Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(DESKTOP_PATH / "10_copula_tail_dependence.png", dpi=300)
    plt.close(fig10)
    print("  ✅ 10_copula_tail_dependence.png")
    
    print(f"\n✅ All 10 plots saved to Desktop")


# =============================================================================
# DASHBOARD
# =============================================================================
def print_dashboard(metrics, garch_mc_var, mw_metrics, bl_metrics, 
                    backtest_df, corr_matrix, rolling_results):
    """Print final analysis dashboard"""
    print("\n" + "="*80)
    print("🏆 COMPLETE ULTIMATE DASHBOARD")
    print("="*80)
    
    print(f"\n{' BASIC METRICS ':=^70}")
    print(f"  🟢 Best Sharpe:     {metrics['Sharpe_Ratio'].idxmax()} ({metrics['Sharpe_Ratio'].max():.2f})")
    print(f"  📈 Highest Alpha:   {metrics['Alpha_%'].idxmax()} ({metrics['Alpha_%'].max():.1f}%)")
    print(f"  🔴 Highest Risk:    {metrics['Historical_VaR_95_%'].idxmin()} ({metrics['Historical_VaR_95_%'].min():.2f}%)")
    print(f"  📊 Avg Volatility:  {metrics['Annual_Volatility_%'].mean():.2f}%")
    
    print(f"\n{' VAR COMPARISON ':=^70}")
    print(f"  📉 Historical VaR 95%:     {metrics['Historical_VaR_95_%'].mean():.2f}%")
    print(f"  🎲 GARCH-MC VaR 95%:      {garch_mc_var['GARCH_MC_VaR_95_%'].mean():.2f}%")
    
    print(f"\n{' PORTFOLIO COMPARISON ':=^70}")
    print(f"  📈 Markowitz Return:   {mw_metrics['Return_%']:.2f}%")
    print(f"  📈 BL Return:          {bl_metrics['Return_%']:.2f}%")
    print(f"  ⭐ Markowitz Sharpe:   {mw_metrics['Sharpe']:.3f}")
    print(f"  ⭐ BL Sharpe:          {bl_metrics['Sharpe']:.3f}")
    print(f"  📊 Markowitz Assets:   {mw_metrics['Assets']}")
    print(f"  📊 BL Assets:          {bl_metrics['Assets']}")
    
    print(f"\n{' VAR BACKTESTING ':=^70}")
    valid_count = backtest_df['Valid'].sum() if len(backtest_df) > 0 else 0
    if len(backtest_df) > 0:
        print(f"  ✅ Valid VaR Models:   {valid_count}/{len(backtest_df)} ({valid_count/len(backtest_df)*100:.0f}%)")
    
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    print(f"\n{' DIVERSIFICATION ':=^70}")
    print(f"  🔄 Average Correlation:  {avg_corr:.3f}")
    if avg_corr < 0.3:
        print(f"  ✅ Well Diversified")
    elif avg_corr < 0.5:
        print(f"  📊 Moderately Diversified")
    else:
        print(f"  ⚠️ Needs More Diversification")
    
    inc_count = sum(1 for r in rolling_results.values() if r['Trend'] == 'Increasing')
    print(f"\n{' VOLATILITY TRENDS ':=^70}")
    print(f"  📈 Increasing Volatility: {inc_count}/{len(rolling_results)} assets")
    
    print(f"\n{' FINAL RECOMMENDATIONS ':=^70}")
    buy = []
    for ticker in metrics.index:
        if ticker in garch_mc_var.index and len(backtest_df) > 0 and ticker in backtest_df.index:
            if (metrics.loc[ticker, 'Sharpe_Ratio'] > 0.5 and 
                backtest_df.loc[ticker, 'Valid'] and
                garch_mc_var.loc[ticker, 'GARCH_MC_VaR_95_%'] > -4):
                buy.append(ticker)
    
    if buy:
        print("  ✅ STRONG BUY:")
        for t in buy[:5]:
            print(f"     • {t}: Sharpe={metrics.loc[t, 'Sharpe_Ratio']:.2f}, VaR={garch_mc_var.loc[t, 'GARCH_MC_VaR_95_%']:.2f}%")
    else:
        print("  ⚠️ No strong buy signals at this time")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("\n" + "🔷"*40)
    print("    QUANTITATIVE RISK MANAGEMENT SYSTEM")
    print("    COMPLETE PORTFOLIO ANALYSIS")
    print("🔷"*40)
    
    # Load data
    prices, benchmark = get_data()
    if prices is None:
        print("❌ Failed to load data")
        return
    
    # Calculate metrics
    metrics, returns, bench_returns = compute_metrics(prices, benchmark)
    display_df("BASIC METRICS", metrics.sort_values('Sharpe_Ratio', ascending=False))
    
    # GARCH
    garch_results = fit_garch(returns)
    if len(garch_results) > 0:
        display_df("GARCH RESULTS", garch_results)
    
    # Monte Carlo VaR
    mc_var = mc_var_simulation(returns)
    display_df("MONTE CARLO VAR (Normal)", mc_var)
    
    # GARCH-MC VaR
    garch_mc_var = garch_mc_studentt(returns)
    if len(garch_mc_var) > 0:
        display_df("GARCH-MONTE CARLO VAR (Student-t)", garch_mc_var)
    
    # EVT
    evt_var = extreme_value_analysis(returns)
    if len(evt_var) > 0:
        display_df("EXTREME VALUE THEORY", evt_var)
    
    # Markowitz
    mw_weights, mw_metrics = markowitz_optimization(returns)
    display_df("MARKOWITZ OPTIMAL PORTFOLIO", mw_weights)
    
    # Black-Litterman
    bl_weights, bl_metrics = black_litterman(returns)
    display_df("BLACK-LITTERMAN OPTIMAL PORTFOLIO", bl_weights)
    
    # Fama-French
    ff_model = fama_french_3factor(returns, bench_returns)
    display_df("FAMA-FRENCH 3-FACTOR", ff_model)
    
    # Correlation
    corr_matrix = correlation_analysis(returns)
    display_df("CORRELATION MATRIX", corr_matrix)
    
    # Backtesting
    if len(garch_mc_var) > 0:
        backtest_df = backtest_var(returns, garch_mc_var)
        display_df("VAR BACKTESTING RESULTS", backtest_df)
    else:
        backtest_df = pd.DataFrame()
    
    # Rolling analysis
    rolling_results, dyn_corr = rolling_risk_analysis(returns)
    
    # Copula
    copula_tail = copula_tail_dependence(returns)
    display_df("COPULA TAIL DEPENDENCE", copula_tail)
    
    # Generate plots
    generate_plots(metrics, returns, mc_var, garch_mc_var, evt_var, 
                   mw_weights, bl_weights, ff_model, corr_matrix, 
                   backtest_df, dyn_corr, copula_tail, garch_results)
    
    # Dashboard
    print_dashboard(metrics, garch_mc_var, mw_metrics, bl_metrics,
                    backtest_df, corr_matrix, rolling_results)
    
    # Save to Excel
    try:
        with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:
            metrics.to_excel(writer, sheet_name='1_Basic_Metrics')
            if len(garch_results) > 0:
                garch_results.to_excel(writer, sheet_name='2_GARCH')
            mc_var.to_excel(writer, sheet_name='3_MC_VaR')
            if len(garch_mc_var) > 0:
                garch_mc_var.to_excel(writer, sheet_name='4_GARCH_MC_VaR')
            if len(evt_var) > 0:
                evt_var.to_excel(writer, sheet_name='5_EVT')
            mw_weights.to_excel(writer, sheet_name='6_Markowitz', index=False)
            bl_weights.to_excel(writer, sheet_name='7_BlackLitterman', index=False)
            ff_model.to_excel(writer, sheet_name='8_FamaFrench')
            corr_matrix.to_excel(writer, sheet_name='9_Correlation')
            if len(backtest_df) > 0:
                backtest_df.to_excel(writer, sheet_name='10_Backtesting')
            copula_tail.to_excel(writer, sheet_name='11_Copula_Tail')
        print(f"\n✅ Excel saved to: {EXCEL_PATH}")
    except Exception as e:
        print(f"⚠️ Could not save Excel: {e}")
    
    # Final
    print("\n" + "🔷"*40)
    print("    ✅ ALL COMPLETED")
    print(f"    📁 Excel: {EXCEL_PATH}")
    print("    📈 10 PLOTS SAVED:")
    print("    1_var_comparison.png           6_var_backtesting.png")
    print("    2_risk_return_profile.png      7_dynamic_correlation.png")
    print("    3_correlation_heatmap.png      8_evt_tail_shapes.png")
    print("    4_fama_french_factors.png      9_garch_persistence.png")
    print("    5_portfolio_weights.png        10_copula_tail_dependence.png")
    print("🔷"*40)


if __name__ == "__main__":
    main()