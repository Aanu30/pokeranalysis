"""
Analysing 35 Poker Sessions: What the Data Says About My Edge
=============================================================
A statistical breakdown of live poker results examining win rates, variance, 
session length patterns, and what the numbers reveal about decision quality versus luck.

Statistical Methods Used (ST202 - Probability and Statistical Inference):
- Hypothesis Testing (Chapter 8.4)
- Confidence Intervals (Chapter 8.3)
- Maximum Likelihood Estimation (Chapter 9.3)
- Likelihood Ratio Tests (Chapter 9.4)
- Bootstrap Resampling
- Power Analysis
- Bayesian Updating

Author: Aarin Bhatt
Course: BSc Data Science, LSE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PLOT STYLING - Dark theme matching website
# =============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_card': '#16213e',
    'accent': '#e94560',
    'cyan': '#00d9ff',
    'green': '#00ff88',
    'red': '#ff4466',
    'gold': '#ffd700',
    'text': '#eaeaea',
    'grid': '#0f3460'
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg_dark'],
    'axes.facecolor': COLORS['bg_card'],
    'axes.edgecolor': COLORS['accent'],
    'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'grid.color': COLORS['grid'],
    'font.family': 'sans-serif',
    'font.size': 11
})

# =============================================================================
# DATA
# =============================================================================
data = [
    ("02/10/25", "Poker Society", "10p/20p", 7, -18),
    ("09/10/25", "Poker Society", "10p/25p", 8, 10.25),
    ("24/10/25", "Random Home Game", "1/1", 8, 69),
    ("30/10/25", "HP DG MB JF RP", "10p/20p", 6, 38),
    ("03/11/25", "HP DG MB RP EB AB", "10p/20p", 7, -17.6),
    ("04/11/25", "HP DG MB RP", "10p/20p", 5, -7.7),
    ("07/11/25", "HP DG MB JF (KD)", "10p/20p", 5.5, -19.5),
    ("07/11/25", "HP DG MB", "10p/20p", 5, -16.6),
    ("08/11/25", "Random Home Game (MB)", "20p/20p", 6, 138),
    ("11/11/25", "HP DG MB Zain Avi AJ RS", "10p/20p", 6.5, 37),
    ("14/11/25", "HP AS DG MB ZH EB", "10p/20p", 5.5, -0.6),
    ("18/11/25", "HP AS DG MA...", "10p/20p", 6, 4.8),
    ("19/11/25", "HP DG AS MB", "10p/20p", 5.5, 10.3),
    ("20/11/25", "HP VC EB", "10p/20p", 4, 5.85),
    ("28/11/25", "HP DG AS ZH RS EB", "10p/20p", 5, -46.4),
    ("29/11/25", "Random Home Game", "20p/20p", 6, -86.4),
    ("02/12/25", "HP DG AS MB", "10p/20p", 5.5, -17.2),
    ("03/12/25", "HP DG AS MB ZH", "10p/20p", 6, -26.4),
    ("04/12/25", "HP DG AS MB ZH EB", "10p/20p", 6, 46.5),
    ("04/12/25", "Poker Society", "10p/20p", 6, 21.5),
    ("08/12/25", "HP AS DG MB", "10p/20p", 5, 27.2),
    ("09/12/25", "HP AS DG MB DH AJ", "10p/20p", 7, -17.4),
    ("09/12/25", "Hindu Society Poker (MB)", "10p/20p", 5.5, 25.5),
    ("09/12/25", "MB JF AK (Carr)", "10p/20p", 6, -10),
    ("10/12/25", "MB", "10p/20p", 2, 3.4),
    ("10/12/25", "HP DG MB AS JF", "10p/20p", 6, -20),
    ("11/12/25", "Poker Society (MB)", "10p/20p", 6, 3.51),
    ("16/01/26", "HP MB DG AS AJ ZH JF", "10p/20p", 8, 70.6),
    ("19/01/26", "HP MB DG AS JF", "10p/20p", 6, -20.2),
    ("21/01/26", "HP MB AS", "10p/20p", 4, -12),
    ("22/01/26", "MB JF AK (Carr)", "10p/20p", 7, 14),
    ("23/01/26", "Random Home Game", "10p/20p", 6.5, -13),
    ("27/01/26", "HP MB AS", "10p/20p", 4, 23),
    ("29/01/26", "Poker Society", "5p/10p", 7, 28.65),
    ("29/01/26", "Poker Society", "20p/20p", 8, 18.8),
    ("02/02/26", "Random Home Game", "50p/50p", 7, 28),
]

# Create DataFrame
df = pd.DataFrame(data, columns=['Date', 'Group', 'Blinds', 'Players', 'Profit'])
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
df = df.sort_values('Date').reset_index(drop=True)

# Categorise games
def categorise_game(group):
    if 'Poker Society' in group or 'Hindu Society' in group:
        return 'Poker Soc'
    elif 'Random Home Game' in group:
        return 'Random Home Games'
    else:
        return 'Friend Games'

df['Game_Type'] = df['Group'].apply(categorise_game)
df['Cumulative_Profit'] = df['Profit'].cumsum()
df['Session_Number'] = range(1, len(df) + 1)

# Extract profit array for analysis
profits = df['Profit'].values
n = len(profits)

# =============================================================================
# SECTION 1: DESCRIPTIVE STATISTICS
# =============================================================================
print("=" * 80)
print("ANALYSING 35 POKER SESSIONS: WHAT THE DATA SAYS ABOUT MY EDGE")
print("=" * 80)
print("\n" + "─" * 80)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("─" * 80)

total_profit = profits.sum()
win_sessions = (profits > 0).sum()
lose_sessions = (profits < 0).sum()
win_rate = win_sessions / n * 100

sample_mean = profits.mean()
sample_median = np.median(profits)
sample_std = profits.std(ddof=1)  # Sample std with Bessel's correction
sample_var = profits.var(ddof=1)

avg_win = profits[profits > 0].mean()
avg_loss = profits[profits < 0].mean()

# Skewness and Kurtosis
skewness = scipy_stats.skew(profits)
kurtosis = scipy_stats.kurtosis(profits)  # Excess kurtosis

print(f"""
Total Sessions:           {n}
Total Profit:             £{total_profit:.2f}

Winning Sessions:         {win_sessions} ({win_rate:.1f}%)
Losing Sessions:          {lose_sessions} ({100-win_rate:.1f}%)

Sample Mean (x̄):          £{sample_mean:.2f}
Sample Median:            £{sample_median:.2f}
Sample Std Dev (s):       £{sample_std:.2f}
Sample Variance (s²):     £{sample_var:.2f}

Average Win:              £{avg_win:.2f}
Average Loss:             £{avg_loss:.2f}
Win/Loss Ratio:           {abs(avg_win/avg_loss):.2f}

Skewness:                 {skewness:.3f} {'(right-skewed)' if skewness > 0 else '(left-skewed)'}
Excess Kurtosis:          {kurtosis:.3f} {'(heavy tails)' if kurtosis > 0 else '(light tails)'}

Biggest Win:              £{profits.max():.2f}
Biggest Loss:             £{profits.min():.2f}
""")

# =============================================================================
# SECTION 2: HYPOTHESIS TESTING (ST202 Chapter 8.4)
# =============================================================================
print("─" * 80)
print("SECTION 2: HYPOTHESIS TESTING (ST202 Ch. 8.4)")
print("─" * 80)

print("""
Testing whether there is a statistically significant edge:

    H₀: μ = 0    (No edge - expected profit is zero)
    H₁: μ ≠ 0    (Edge exists - expected profit is non-zero)

Using a one-sample t-test since population variance is unknown.
""")

# One-sample t-test
t_statistic = sample_mean / (sample_std / np.sqrt(n))
df_ttest = n - 1
p_value_ttest = 2 * (1 - scipy_stats.t.cdf(abs(t_statistic), df_ttest))

print(f"Test Statistic:  t = x̄ / (s/√n) = {sample_mean:.2f} / ({sample_std:.2f}/√{n})")
print(f"                 t = {t_statistic:.4f}")
print(f"Degrees of Freedom: {df_ttest}")
print(f"P-value (two-tailed): {p_value_ttest:.4f}")

# Critical values
alpha_levels = [0.10, 0.05, 0.01]
print(f"\nCritical Values:")
for alpha in alpha_levels:
    t_crit = scipy_stats.t.ppf(1 - alpha/2, df_ttest)
    reject = abs(t_statistic) > t_crit
    print(f"  α = {alpha:.2f}: t_crit = ±{t_crit:.3f} → {'REJECT H₀' if reject else 'Fail to reject H₀'}")

# One-sided test (more relevant: is edge positive?)
print(f"""
One-Sided Test (H₁: μ > 0):
    P-value: {p_value_ttest/2:.4f}
    {'Significant at 10% level' if p_value_ttest/2 < 0.10 else 'Not significant at 10% level'}
""")

# =============================================================================
# SECTION 3: CONFIDENCE INTERVALS (ST202 Chapter 8.3)
# =============================================================================
print("─" * 80)
print("SECTION 3: CONFIDENCE INTERVALS (ST202 Ch. 8.3)")
print("─" * 80)

print("""
Constructing confidence intervals for the true mean profit μ.
Using t-distribution since σ is unknown.

    CI = x̄ ± t_{α/2, n-1} × (s/√n)
""")

confidence_levels = [0.90, 0.95, 0.99]
print("Confidence Intervals for Mean Profit (μ):")
for conf in confidence_levels:
    alpha = 1 - conf
    t_crit = scipy_stats.t.ppf(1 - alpha/2, df_ttest)
    margin = t_crit * (sample_std / np.sqrt(n))
    ci_lower = sample_mean - margin
    ci_upper = sample_mean + margin
    contains_zero = ci_lower <= 0 <= ci_upper
    print(f"  {int(conf*100)}% CI: [£{ci_lower:.2f}, £{ci_upper:.2f}] {'(contains 0)' if contains_zero else '(excludes 0)'}")

# CI for win rate (proportion)
print(f"\nConfidence Intervals for Win Rate (p):")
p_hat = win_rate / 100
for conf in confidence_levels:
    alpha = 1 - conf
    z_crit = scipy_stats.norm.ppf(1 - alpha/2)
    # Wilson score interval (better for proportions)
    denominator = 1 + z_crit**2 / n
    centre = (p_hat + z_crit**2 / (2*n)) / denominator
    margin = z_crit * np.sqrt((p_hat*(1-p_hat) + z_crit**2/(4*n)) / n) / denominator
    ci_lower = (centre - margin) * 100
    ci_upper = (centre + margin) * 100
    print(f"  {int(conf*100)}% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")

# =============================================================================
# SECTION 4: MAXIMUM LIKELIHOOD ESTIMATION (ST202 Chapter 9.3)
# =============================================================================
print("\n" + "─" * 80)
print("SECTION 4: MAXIMUM LIKELIHOOD ESTIMATION (ST202 Ch. 9.3)")
print("─" * 80)

print("""
Fitting distributions to the profit data using MLE.

Model 1: Normal Distribution X ~ N(μ, σ²)
    Log-likelihood: ℓ(μ,σ²) = -n/2 log(2π) - n/2 log(σ²) - 1/(2σ²) Σ(xᵢ-μ)²
    
    MLEs: μ̂ = x̄,  σ̂² = (1/n) Σ(xᵢ-x̄)²
""")

# MLE for Normal distribution
mu_mle = profits.mean()
sigma2_mle = profits.var(ddof=0)  # MLE uses n, not n-1
sigma_mle = np.sqrt(sigma2_mle)

# Log-likelihood at MLE
def normal_loglik(params, data):
    mu, sigma = params
    if sigma <= 0:
        return -np.inf
    n = len(data)
    return -n/2 * np.log(2*np.pi) - n/2 * np.log(sigma**2) - 1/(2*sigma**2) * np.sum((data - mu)**2)

loglik_normal = normal_loglik([mu_mle, sigma_mle], profits)

print(f"Normal Distribution MLE:")
print(f"  μ̂ = £{mu_mle:.2f}")
print(f"  σ̂ = £{sigma_mle:.2f}")
print(f"  Log-likelihood: ℓ = {loglik_normal:.2f}")

# AIC and BIC for model comparison
k_normal = 2  # number of parameters
aic_normal = 2*k_normal - 2*loglik_normal
bic_normal = k_normal*np.log(n) - 2*loglik_normal
print(f"  AIC = {aic_normal:.2f}")
print(f"  BIC = {bic_normal:.2f}")

# Model 2: Student's t-distribution (heavier tails)
print(f"""
Model 2: Student's t-Distribution (accounts for heavy tails)
    Fitting location (μ), scale (σ), and degrees of freedom (ν)
""")

# Fit t-distribution
t_params = scipy_stats.t.fit(profits)
t_df, t_loc, t_scale = t_params

def t_loglik(data, df, loc, scale):
    return np.sum(scipy_stats.t.logpdf(data, df, loc, scale))

loglik_t = t_loglik(profits, t_df, t_loc, t_scale)
k_t = 3
aic_t = 2*k_t - 2*loglik_t
bic_t = k_t*np.log(n) - 2*loglik_t

print(f"Student's t MLE:")
print(f"  ν̂ (df) = {t_df:.2f}")
print(f"  μ̂ (loc) = £{t_loc:.2f}")
print(f"  σ̂ (scale) = £{t_scale:.2f}")
print(f"  Log-likelihood: ℓ = {loglik_t:.2f}")
print(f"  AIC = {aic_t:.2f}")
print(f"  BIC = {bic_t:.2f}")

# Model comparison
print(f"\nModel Comparison:")
print(f"  {'Student-t preferred' if aic_t < aic_normal else 'Normal preferred'} by AIC (Δ = {abs(aic_t - aic_normal):.2f})")
print(f"  {'Student-t preferred' if bic_t < bic_normal else 'Normal preferred'} by BIC (Δ = {abs(bic_t - bic_normal):.2f})")

# =============================================================================
# SECTION 5: LIKELIHOOD RATIO TEST (ST202 Chapter 9.4)
# =============================================================================
print("\n" + "─" * 80)
print("SECTION 5: LIKELIHOOD RATIO TEST (ST202 Ch. 9.4)")
print("─" * 80)

print("""
Testing H₀: μ = 0 vs H₁: μ ≠ 0 using the likelihood ratio test.

    Λ = L(μ=0, σ̂₀) / L(μ̂, σ̂)
    
    -2 log(Λ) ~ χ²(1) under H₀ (by Wilks' theorem)
""")

# MLE under H0 (μ = 0)
sigma2_mle_h0 = np.mean(profits**2)  # MLE of σ² when μ=0
sigma_mle_h0 = np.sqrt(sigma2_mle_h0)

loglik_h0 = normal_loglik([0, sigma_mle_h0], profits)
loglik_h1 = normal_loglik([mu_mle, sigma_mle], profits)

# Likelihood ratio statistic
lr_statistic = -2 * (loglik_h0 - loglik_h1)
p_value_lr = 1 - scipy_stats.chi2.cdf(lr_statistic, df=1)

print(f"Under H₀ (μ=0):  σ̂₀ = £{sigma_mle_h0:.2f}, ℓ₀ = {loglik_h0:.2f}")
print(f"Under H₁:        μ̂ = £{mu_mle:.2f}, σ̂ = £{sigma_mle:.2f}, ℓ₁ = {loglik_h1:.2f}")
print(f"\nLikelihood Ratio Statistic: -2log(Λ) = {lr_statistic:.4f}")
print(f"P-value (χ²₁): {p_value_lr:.4f}")
print(f"\nConclusion: {'Reject H₀ at 5% level' if p_value_lr < 0.05 else 'Fail to reject H₀ at 5% level'}")

# =============================================================================
# SECTION 6: POWER ANALYSIS
# =============================================================================
print("\n" + "─" * 80)
print("SECTION 6: POWER ANALYSIS")
print("─" * 80)

print("""
Power = P(Reject H₀ | H₁ is true)

Calculating:
1. Power of current test (n=36) to detect the observed effect
2. Sample size needed for 80% power
3. Detectable effect size at 80% power with current n
""")

# Effect size (Cohen's d)
effect_size = sample_mean / sample_std
print(f"\nObserved Effect Size (Cohen's d): d = μ/σ = {effect_size:.3f}")
print(f"  Interpretation: {'Small' if abs(effect_size) < 0.5 else 'Medium' if abs(effect_size) < 0.8 else 'Large'} effect")

# Power of current test
alpha = 0.05
t_crit_power = scipy_stats.t.ppf(1 - alpha/2, n-1)
ncp = effect_size * np.sqrt(n)  # Non-centrality parameter
power_current = 1 - scipy_stats.nct.cdf(t_crit_power, n-1, ncp) + scipy_stats.nct.cdf(-t_crit_power, n-1, ncp)

print(f"\nPower of Current Test (n={n}, α=0.05):")
print(f"  Power = {power_current:.1%}")
print(f"  {'Adequately powered (≥80%)' if power_current >= 0.80 else 'Underpowered (<80%)'}")

# Sample size for 80% power
def calculate_sample_size(effect_size, alpha=0.05, power=0.80):
    """Calculate required n for given effect size and power"""
    for n_test in range(10, 5000):
        t_crit = scipy_stats.t.ppf(1 - alpha/2, n_test-1)
        ncp = effect_size * np.sqrt(n_test)
        achieved_power = 1 - scipy_stats.nct.cdf(t_crit, n_test-1, ncp) + scipy_stats.nct.cdf(-t_crit, n_test-1, ncp)
        if achieved_power >= power:
            return n_test
    return 5000

n_required_80 = calculate_sample_size(effect_size)
n_required_90 = calculate_sample_size(effect_size, power=0.90)

print(f"\nSample Size Required:")
print(f"  For 80% power: n = {n_required_80} sessions")
print(f"  For 90% power: n = {n_required_90} sessions")
print(f"  Sessions remaining: {n_required_80 - n} more for 80% power")

# Minimum detectable effect
def min_detectable_effect(n, alpha=0.05, power=0.80):
    """Find minimum effect size detectable with given n and power"""
    for d in np.arange(0.01, 2.0, 0.01):
        t_crit = scipy_stats.t.ppf(1 - alpha/2, n-1)
        ncp = d * np.sqrt(n)
        achieved_power = 1 - scipy_stats.nct.cdf(t_crit, n-1, ncp) + scipy_stats.nct.cdf(-t_crit, n-1, ncp)
        if achieved_power >= power:
            return d
    return 2.0

mde = min_detectable_effect(n)
mde_pounds = mde * sample_std

print(f"\nMinimum Detectable Effect (80% power, α=0.05):")
print(f"  Cohen's d = {mde:.2f}")
print(f"  In pounds: £{mde_pounds:.2f} per session")
print(f"  Your observed effect (£{sample_mean:.2f}) is {'detectable' if abs(sample_mean) >= mde_pounds else 'below detection threshold'}")

# =============================================================================
# SECTION 7: BOOTSTRAP INFERENCE
# =============================================================================
print("\n" + "─" * 80)
print("SECTION 7: BOOTSTRAP INFERENCE")
print("─" * 80)

print("""
Non-parametric bootstrap for robust inference.
Resampling with replacement B=10,000 times.
""")

np.random.seed(42)
B = 10000
bootstrap_means = np.array([np.mean(np.random.choice(profits, n, replace=True)) for _ in range(B)])
bootstrap_medians = np.array([np.median(np.random.choice(profits, n, replace=True)) for _ in range(B)])

# Bootstrap confidence intervals
def bootstrap_ci(bootstrap_stats, confidence=0.95):
    """Percentile method CI"""
    alpha = 1 - confidence
    return np.percentile(bootstrap_stats, [100*alpha/2, 100*(1-alpha/2)])

ci_mean_boot = bootstrap_ci(bootstrap_means)
ci_median_boot = bootstrap_ci(bootstrap_medians)

print(f"Bootstrap Results (B = {B:,}):")
print(f"\n  Mean Profit:")
print(f"    Bootstrap SE: £{bootstrap_means.std():.2f}")
print(f"    95% CI (percentile): [£{ci_mean_boot[0]:.2f}, £{ci_mean_boot[1]:.2f}]")

print(f"\n  Median Profit:")
print(f"    Bootstrap SE: £{bootstrap_medians.std():.2f}")
print(f"    95% CI (percentile): [£{ci_median_boot[0]:.2f}, £{ci_median_boot[1]:.2f}]")

# Bootstrap p-value (test H0: μ = 0)
# Shift data to have mean 0, then see how often bootstrap mean exceeds observed
profits_centered = profits - profits.mean()
bootstrap_means_h0 = np.array([np.mean(np.random.choice(profits_centered, n, replace=True)) for _ in range(B)])
p_value_bootstrap = np.mean(np.abs(bootstrap_means_h0) >= np.abs(sample_mean))

print(f"\n  Bootstrap Hypothesis Test (H₀: μ = 0):")
print(f"    P-value: {p_value_bootstrap:.4f}")

# Probability of being a winning player
prob_positive = np.mean(bootstrap_means > 0)
print(f"\n  P(μ > 0) from bootstrap: {prob_positive:.1%}")

# =============================================================================
# SECTION 8: BAYESIAN ANALYSIS
# =============================================================================
print("\n" + "─" * 80)
print("SECTION 8: BAYESIAN INFERENCE")
print("─" * 80)

print("""
Bayesian updating with conjugate priors.

Prior: μ ~ N(μ₀, σ₀²)  [Prior belief about edge]
Likelihood: X|μ ~ N(μ, σ²)
Posterior: μ|X ~ N(μₙ, σₙ²)

Using three different priors to test sensitivity:
""")

# Known variance approximation (using sample variance)
sigma_known = sample_std

priors = [
    ("Skeptical", 0, 20),      # Prior centred at 0 (no edge), moderate uncertainty
    ("Neutral", 0, 50),        # Wider prior, more uncertainty
    ("Optimistic", 10, 30),    # Prior centred at positive edge
]

print(f"Data: n = {n}, x̄ = £{sample_mean:.2f}, s = £{sample_std:.2f}\n")

for prior_name, mu_0, sigma_0 in priors:
    # Posterior parameters (conjugate normal-normal)
    precision_0 = 1 / sigma_0**2
    precision_data = n / sigma_known**2
    precision_n = precision_0 + precision_data
    sigma_n = np.sqrt(1 / precision_n)
    mu_n = (precision_0 * mu_0 + precision_data * sample_mean) / precision_n
    
    # Posterior probability of positive edge
    prob_positive_bayes = 1 - scipy_stats.norm.cdf(0, mu_n, sigma_n)
    
    # 95% Credible interval
    ci_lower_bayes = scipy_stats.norm.ppf(0.025, mu_n, sigma_n)
    ci_upper_bayes = scipy_stats.norm.ppf(0.975, mu_n, sigma_n)
    
    print(f"{prior_name} Prior: N({mu_0}, {sigma_0}²)")
    print(f"  Posterior: μ|X ~ N({mu_n:.2f}, {sigma_n:.2f}²)")
    print(f"  95% Credible Interval: [£{ci_lower_bayes:.2f}, £{ci_upper_bayes:.2f}]")
    print(f"  P(μ > 0 | data) = {prob_positive_bayes:.1%}")
    print()

# =============================================================================
# SECTION 9: VARIANCE & RISK ANALYSIS
# =============================================================================
print("─" * 80)
print("SECTION 9: VARIANCE & RISK ANALYSIS")
print("─" * 80)

# Drawdown analysis
cumulative = df['Cumulative_Profit'].values
peak = np.maximum.accumulate(cumulative)
drawdown = cumulative - peak
max_drawdown = drawdown.min()
max_drawdown_idx = drawdown.argmin()

# Risk metrics
sharpe_like = sample_mean / sample_std  # Per-session Sharpe ratio
sortino_denom = np.sqrt(np.mean(np.minimum(profits, 0)**2))  # Downside deviation
sortino_like = sample_mean / sortino_denom if sortino_denom > 0 else np.inf

# Value at Risk
var_95 = np.percentile(profits, 5)
var_99 = np.percentile(profits, 1)
cvar_95 = profits[profits <= var_95].mean()

print(f"""
Risk Metrics:
  Maximum Drawdown:        £{max_drawdown:.2f} (at session {max_drawdown_idx + 1})
  
  Sharpe-like Ratio:       {sharpe_like:.3f} (mean/std per session)
  Sortino-like Ratio:      {sortino_like:.3f} (mean/downside dev)
  
  Value at Risk (95%):     £{var_95:.2f} (5% chance of losing more)
  Value at Risk (99%):     £{var_99:.2f} (1% chance of losing more)
  Conditional VaR (95%):   £{cvar_95:.2f} (expected loss in worst 5%)
  
Streak Analysis:
  Longest Win Streak:      {max([len(list(g)) for k, g in __import__('itertools').groupby(profits > 0) if k])}
  Longest Loss Streak:     {max([len(list(g)) for k, g in __import__('itertools').groupby(profits > 0) if not k])}
""")

# =============================================================================
# SECTION 10: GAME TYPE ANALYSIS
# =============================================================================
print("─" * 80)
print("SECTION 10: PERFORMANCE BY GAME TYPE")
print("─" * 80)

for game_type in df['Game_Type'].unique():
    subset = df[df['Game_Type'] == game_type]['Profit']
    n_sub = len(subset)
    mean_sub = subset.mean()
    std_sub = subset.std(ddof=1) if n_sub > 1 else 0
    win_rate_sub = (subset > 0).mean() * 100
    
    # t-test for this game type
    if n_sub > 1 and std_sub > 0:
        t_stat_sub = mean_sub / (std_sub / np.sqrt(n_sub))
        p_val_sub = 2 * (1 - scipy_stats.t.cdf(abs(t_stat_sub), n_sub - 1))
    else:
        t_stat_sub, p_val_sub = np.nan, np.nan
    
    print(f"\n{game_type}:")
    print(f"  Sessions: {n_sub}")
    print(f"  Total: £{subset.sum():.2f}")
    print(f"  Mean: £{mean_sub:.2f} ± £{std_sub:.2f}")
    print(f"  Win Rate: {win_rate_sub:.1f}%")
    if not np.isnan(p_val_sub):
        print(f"  t-test p-value: {p_val_sub:.3f} {'*' if p_val_sub < 0.1 else ''}")

# =============================================================================
# CREATE VISUALISATIONS
# =============================================================================
print("\n" + "─" * 80)
print("GENERATING VISUALISATIONS...")
print("─" * 80)

# Figure 1: Cumulative Profit
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.fill_between(df['Session_Number'], 0, df['Cumulative_Profit'], 
                  where=(df['Cumulative_Profit'] >= 0), alpha=0.3, color=COLORS['cyan'])
ax1.fill_between(df['Session_Number'], 0, df['Cumulative_Profit'], 
                  where=(df['Cumulative_Profit'] < 0), alpha=0.3, color=COLORS['red'])
ax1.plot(df['Session_Number'], df['Cumulative_Profit'], color=COLORS['cyan'], linewidth=2.5)
ax1.axhline(y=0, color=COLORS['accent'], linestyle='--', alpha=0.5)
colors_scatter = [COLORS['green'] if p > 0 else COLORS['red'] for p in profits]
ax1.scatter(df['Session_Number'], df['Cumulative_Profit'], c=colors_scatter, s=50, zorder=5, 
            edgecolors='white', linewidth=0.5)
ax1.set_xlabel('Session Number', fontsize=12)
ax1.set_ylabel('Cumulative Profit (£)', fontsize=12)
ax1.set_title('Cumulative Profit Over Time', fontsize=16, fontweight='bold', pad=20)
ax1.annotate(f'Total: £{total_profit:.2f}', 
             xy=(n, df['Cumulative_Profit'].iloc[-1]),
             xytext=(10, 10), textcoords='offset points',
             fontsize=11, color=COLORS['cyan'], fontweight='bold')
plt.tight_layout()
fig1.savefig('/home/claude/01_cumulative_profit.png', dpi=150, bbox_inches='tight', 
             facecolor=COLORS['bg_dark'], edgecolor='none')
plt.close()

# Figure 2: Session Results
fig2, ax2 = plt.subplots(figsize=(12, 6))
colors_bar = [COLORS['green'] if p > 0 else COLORS['red'] for p in profits]
ax2.bar(df['Session_Number'], profits, color=colors_bar, edgecolor='white', linewidth=0.5)
ax2.axhline(y=0, color='white', linestyle='-', alpha=0.3)
ax2.axhline(y=sample_mean, color=COLORS['cyan'], linestyle='--', linewidth=2, label=f'Mean: £{sample_mean:.2f}')
ax2.axhline(y=sample_mean + 1.96*sample_std/np.sqrt(n), color=COLORS['gold'], linestyle=':', alpha=0.7)
ax2.axhline(y=sample_mean - 1.96*sample_std/np.sqrt(n), color=COLORS['gold'], linestyle=':', alpha=0.7, 
            label='95% CI for Mean')
ax2.set_xlabel('Session Number', fontsize=12)
ax2.set_ylabel('Profit/Loss (£)', fontsize=12)
ax2.set_title('Individual Session Results', fontsize=16, fontweight='bold', pad=20)
ax2.legend(loc='upper left', facecolor=COLORS['bg_card'], edgecolor=COLORS['accent'])
plt.tight_layout()
fig2.savefig('/home/claude/02_session_profits.png', dpi=150, bbox_inches='tight',
             facecolor=COLORS['bg_dark'], edgecolor='none')
plt.close()

# Figure 3: Distribution with MLE Fit
fig3, ax3 = plt.subplots(figsize=(10, 6))
n_bins, bins, patches = ax3.hist(profits, bins=15, density=True, alpha=0.7, edgecolor='white', linewidth=0.5)
for i, patch in enumerate(patches):
    patch.set_facecolor(COLORS['green'] if bins[i] >= 0 else COLORS['red'])

# Overlay normal MLE fit
x_range = np.linspace(profits.min() - 20, profits.max() + 20, 200)
normal_pdf = scipy_stats.norm.pdf(x_range, mu_mle, sigma_mle)
ax3.plot(x_range, normal_pdf, color=COLORS['cyan'], linewidth=2, label=f'Normal MLE: N({mu_mle:.1f}, {sigma_mle:.1f}²)')

# Overlay t-distribution fit
t_pdf = scipy_stats.t.pdf(x_range, t_df, t_loc, t_scale)
ax3.plot(x_range, t_pdf, color=COLORS['gold'], linewidth=2, linestyle='--', 
         label=f't-dist MLE: t({t_df:.1f}, {t_loc:.1f}, {t_scale:.1f})')

ax3.axvline(x=0, color='white', linestyle='-', alpha=0.5)
ax3.axvline(x=sample_mean, color=COLORS['accent'], linestyle='-', linewidth=2, label=f'Sample Mean: £{sample_mean:.2f}')
ax3.set_xlabel('Profit/Loss (£)', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title('Profit Distribution with MLE Fits', fontsize=16, fontweight='bold', pad=20)
ax3.legend(loc='upper right', facecolor=COLORS['bg_card'], edgecolor=COLORS['accent'])
plt.tight_layout()
fig3.savefig('/home/claude/03_distribution_mle.png', dpi=150, bbox_inches='tight',
             facecolor=COLORS['bg_dark'], edgecolor='none')
plt.close()

# Figure 4: Bootstrap Distribution
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.hist(bootstrap_means, bins=50, density=True, color=COLORS['cyan'], alpha=0.7, edgecolor='white', linewidth=0.3)
ax4.axvline(x=0, color=COLORS['red'], linestyle='--', linewidth=2, label='H₀: μ = 0')
ax4.axvline(x=sample_mean, color=COLORS['green'], linestyle='-', linewidth=2, label=f'Observed: £{sample_mean:.2f}')
ax4.axvline(x=ci_mean_boot[0], color=COLORS['gold'], linestyle=':', linewidth=2)
ax4.axvline(x=ci_mean_boot[1], color=COLORS['gold'], linestyle=':', linewidth=2)
ax4.fill_betweenx([0, ax4.get_ylim()[1] if ax4.get_ylim()[1] > 0 else 0.1], 
                   ci_mean_boot[0], ci_mean_boot[1], alpha=0.2, color=COLORS['gold'],
                   label=f'95% CI: [£{ci_mean_boot[0]:.2f}, £{ci_mean_boot[1]:.2f}]')
ax4.set_xlabel('Mean Profit per Session (£)', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.set_title(f'Bootstrap Distribution of Mean (B={B:,})', fontsize=16, fontweight='bold', pad=20)
ax4.legend(loc='upper right', facecolor=COLORS['bg_card'], edgecolor=COLORS['accent'])
plt.tight_layout()
fig4.savefig('/home/claude/04_bootstrap.png', dpi=150, bbox_inches='tight',
             facecolor=COLORS['bg_dark'], edgecolor='none')
plt.close()

# Figure 5: Power Curve
fig5, ax5 = plt.subplots(figsize=(10, 6))
sample_sizes = np.arange(10, 300, 5)
powers = []
for n_test in sample_sizes:
    t_crit = scipy_stats.t.ppf(1 - 0.05/2, n_test-1)
    ncp = effect_size * np.sqrt(n_test)
    pwr = 1 - scipy_stats.nct.cdf(t_crit, n_test-1, ncp) + scipy_stats.nct.cdf(-t_crit, n_test-1, ncp)
    powers.append(pwr)

ax5.plot(sample_sizes, powers, color=COLORS['cyan'], linewidth=2.5)
ax5.axhline(y=0.80, color=COLORS['gold'], linestyle='--', linewidth=2, label='80% Power')
ax5.axhline(y=0.90, color=COLORS['accent'], linestyle='--', linewidth=2, label='90% Power')
ax5.axvline(x=n, color=COLORS['green'], linestyle='-', linewidth=2, label=f'Current n={n}')
ax5.axvline(x=n_required_80, color=COLORS['gold'], linestyle=':', linewidth=2, alpha=0.7)
ax5.scatter([n], [power_current], color=COLORS['green'], s=100, zorder=5, edgecolors='white')
ax5.annotate(f'Current: {power_current:.0%}', xy=(n, power_current), xytext=(10, 10), 
             textcoords='offset points', color=COLORS['green'], fontweight='bold')
ax5.set_xlabel('Sample Size (n)', fontsize=12)
ax5.set_ylabel('Power', fontsize=12)
ax5.set_title(f'Power Curve (Effect Size d = {effect_size:.3f})', fontsize=16, fontweight='bold', pad=20)
ax5.legend(loc='lower right', facecolor=COLORS['bg_card'], edgecolor=COLORS['accent'])
ax5.set_ylim(0, 1)
plt.tight_layout()
fig5.savefig('/home/claude/05_power_curve.png', dpi=150, bbox_inches='tight',
             facecolor=COLORS['bg_dark'], edgecolor='none')
plt.close()

# Figure 6: Game Type Comparison
fig6, axes = plt.subplots(1, 2, figsize=(14, 6))

game_stats = df.groupby('Game_Type').agg({
    'Profit': ['sum', 'mean', 'count']
}).round(2)
game_stats.columns = ['Total', 'Mean', 'Count']
game_stats = game_stats.sort_values('Total')

colors_game = [COLORS['red'] if x < 0 else COLORS['green'] for x in game_stats['Total']]
axes[0].barh(game_stats.index, game_stats['Total'], color=colors_game, edgecolor='white', linewidth=0.5)
axes[0].axvline(x=0, color='white', linestyle='-', alpha=0.3)
axes[0].set_xlabel('Total Profit (£)', fontsize=12)
axes[0].set_title('Total Profit by Game Type', fontsize=14, fontweight='bold')
for i, (idx, val) in enumerate(game_stats['Total'].items()):
    axes[0].text(val + 3 if val >= 0 else val - 3, i, f'£{val:.0f}', 
                 va='center', ha='left' if val >= 0 else 'right', fontsize=10, color='white')

win_rates = df.groupby('Game_Type').apply(lambda x: (x['Profit'] > 0).mean() * 100)
win_rates = win_rates.reindex(game_stats.index)
colors_wr = [COLORS['green'] if x > 50 else COLORS['gold'] if x == 50 else COLORS['red'] for x in win_rates]
axes[1].barh(win_rates.index, win_rates.values, color=colors_wr, edgecolor='white', linewidth=0.5)
axes[1].axvline(x=50, color='white', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Win Rate (%)', fontsize=12)
axes[1].set_title('Win Rate by Game Type', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 100)
for i, (idx, val) in enumerate(win_rates.items()):
    axes[1].text(val + 2, i, f'{val:.0f}%', va='center', fontsize=10, color='white')

plt.tight_layout()
fig6.savefig('/home/claude/06_game_type.png', dpi=150, bbox_inches='tight',
             facecolor=COLORS['bg_dark'], edgecolor='none')
plt.close()

# Figure 7: Drawdown
fig7, ax7 = plt.subplots(figsize=(12, 6))
ax7.fill_between(df['Session_Number'], drawdown, 0, alpha=0.5, color=COLORS['red'])
ax7.plot(df['Session_Number'], drawdown, color=COLORS['red'], linewidth=2)
ax7.scatter([max_drawdown_idx + 1], [max_drawdown], color=COLORS['gold'], s=150, zorder=5, marker='v')
ax7.annotate(f'Max DD: £{max_drawdown:.2f}', xy=(max_drawdown_idx + 1, max_drawdown),
             xytext=(10, -20), textcoords='offset points', fontsize=11, color=COLORS['gold'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLORS['gold']))
ax7.set_xlabel('Session Number', fontsize=12)
ax7.set_ylabel('Drawdown (£)', fontsize=12)
ax7.set_title('Drawdown from Peak', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
fig7.savefig('/home/claude/07_drawdown.png', dpi=150, bbox_inches='tight',
             facecolor=COLORS['bg_dark'], edgecolor='none')
plt.close()

# Figure 8: Bayesian Posterior
fig8, ax8 = plt.subplots(figsize=(10, 6))
x_range = np.linspace(-30, 40, 500)

for prior_name, mu_0, sigma_0 in priors:
    precision_0 = 1 / sigma_0**2
    precision_data = n / sigma_known**2
    precision_n = precision_0 + precision_data
    sigma_n = np.sqrt(1 / precision_n)
    mu_n = (precision_0 * mu_0 + precision_data * sample_mean) / precision_n
    
    posterior_pdf = scipy_stats.norm.pdf(x_range, mu_n, sigma_n)
    ax8.plot(x_range, posterior_pdf, linewidth=2, label=f'{prior_name}: N({mu_n:.1f}, {sigma_n:.1f}²)')

ax8.axvline(x=0, color=COLORS['red'], linestyle='--', linewidth=2, alpha=0.7, label='μ = 0 (no edge)')
ax8.axvline(x=sample_mean, color=COLORS['green'], linestyle='-', linewidth=2, alpha=0.7, label=f'MLE: £{sample_mean:.2f}')
ax8.fill_betweenx([0, ax8.get_ylim()[1] if ax8.get_ylim()[1] > 0 else 0.1], 
                   -100, 0, alpha=0.1, color=COLORS['red'])
ax8.set_xlim(-30, 40)
ax8.set_xlabel('Mean Profit μ (£)', fontsize=12)
ax8.set_ylabel('Posterior Density', fontsize=12)
ax8.set_title('Bayesian Posterior Distributions', fontsize=16, fontweight='bold', pad=20)
ax8.legend(loc='upper right', facecolor=COLORS['bg_card'], edgecolor=COLORS['accent'])
plt.tight_layout()
fig8.savefig('/home/claude/08_bayesian.png', dpi=150, bbox_inches='tight',
             facecolor=COLORS['bg_dark'], edgecolor='none')
plt.close()

print("✅ All visualisations saved!")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSIONS")
print("=" * 80)

print(f"""
Over {n} sessions, total profit of £{total_profit:.2f} with {win_rate:.1f}% win rate.

FREQUENTIST ANALYSIS:
  • Sample mean: £{sample_mean:.2f} per session
  • t-test p-value: {p_value_ttest:.3f} (not significant at α=0.05)
  • LR test p-value: {p_value_lr:.3f}
  • 95% CI for μ: [£{sample_mean - scipy_stats.t.ppf(0.975, n-1) * sample_std/np.sqrt(n):.2f}, £{sample_mean + scipy_stats.t.ppf(0.975, n-1) * sample_std/np.sqrt(n):.2f}]
  • Current power: {power_current:.1%} (need {n_required_80} sessions for 80%)

BOOTSTRAP ANALYSIS:
  • 95% CI: [£{ci_mean_boot[0]:.2f}, £{ci_mean_boot[1]:.2f}]
  • P(profitable) = {prob_positive:.1%}

BAYESIAN ANALYSIS (with skeptical prior):
  • Posterior mean: ~£{mu_n:.2f}
  • P(edge > 0 | data) ≈ {1 - scipy_stats.norm.cdf(0, mu_n, sigma_n):.1%}

VERDICT: 
  The data suggests a positive edge ({prob_positive:.0%} probability), but {n_required_80 - n} 
  more sessions are needed to achieve statistical significance with 80% power.
  Keep playing and tracking — the signal is promising but not yet conclusive.
""")

# Export data
df.to_csv('/home/claude/poker_sessions.csv', index=False)

# Export summary statistics as JSON for website
import json
summary = {
    'total_sessions': int(n),
    'total_profit': round(total_profit, 2),
    'win_rate': round(win_rate, 1),
    'avg_profit': round(sample_mean, 2),
    'std_dev': round(sample_std, 2),
    'p_value_ttest': round(p_value_ttest, 4),
    'p_value_lr': round(p_value_lr, 4),
    'ci_95_lower': round(ci_mean_boot[0], 2),
    'ci_95_upper': round(ci_mean_boot[1], 2),
    'prob_winning_player': round(prob_positive * 100, 1),
    'sessions_for_80_power': int(n_required_80),
    'power_current': round(power_current * 100, 1),
    'max_drawdown': round(max_drawdown, 2),
    'effect_size': round(effect_size, 3),
    'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d')
}

with open('/home/claude/summary_stats.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n📁 Exported: poker_sessions.csv, summary_stats.json")
