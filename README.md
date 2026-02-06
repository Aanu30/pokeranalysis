# Analysing 35 Poker Sessions: What the Data Says About My Edge

A statistical breakdown of my live poker results using hypothesis testing, maximum likelihood estimation, Bayesian inference, and bootstrap resampling. Built as a practical application of concepts from **ST202 - Probability and Statistical Inference** at LSE.

![Cumulative Profit](01_cumulative_profit.png)

## Key Findings

| Metric | Value |
|--------|-------|
| Total Sessions | 36 |
| Total Profit | £274.86 |
| Win Rate | 55.6% |
| Average Profit/Session | £7.64 |
| P(Winning Player) | 88.6% |
| Sessions for 80% Power | 195 |

**Verdict:** The data suggests an 89% probability of a positive edge, but 159 more sessions are needed to achieve statistical significance at 80% power.

## Statistical Methods

### Frequentist Analysis (ST202 Ch. 8)
- **Hypothesis Testing:** One-sample t-test for H₀: μ = 0 vs H₁: μ ≠ 0
- **Confidence Intervals:** t-distribution based CIs for mean profit
- **Result:** p = 0.23 (not significant at α = 0.05)

### Maximum Likelihood Estimation (ST202 Ch. 9.3)
- Fitted Normal and Student's t distributions
- Model comparison via AIC/BIC
- **Result:** Student-t preferred (accounts for heavy tails)

### Likelihood Ratio Test (ST202 Ch. 9.4)
- Testing μ = 0 using Wilks' theorem
- **Result:** -2log(Λ) = 1.48, p = 0.22

### Power Analysis
- Current power: 21.8%
- Effect size (Cohen's d): 0.20 (small)
- Required n for 80% power: 195 sessions

### Bootstrap Inference
- Non-parametric bootstrap (B = 10,000)
- 95% CI: [£-4.42, £20.28]
- P(μ > 0): 88.6%

### Bayesian Analysis
- Conjugate Normal-Normal model
- Prior sensitivity analysis (skeptical, neutral, optimistic)
- Posterior P(edge > 0 | data) ≈ 88-90%

## Visualisations

| Chart | Description |
|-------|-------------|
| `01_cumulative_profit.png` | Profit over time |
| `02_session_profits.png` | Individual session results |
| `03_distribution_mle.png` | Profit distribution with MLE fits |
| `04_bootstrap.png` | Bootstrap distribution of mean |
| `05_power_curve.png` | Power vs sample size |
| `06_game_type.png` | Performance by game type |
| `07_drawdown.png` | Drawdown from peak |
| `08_bayesian.png` | Bayesian posterior distributions |

## Performance by Game Type

| Game Type | Sessions | Total | Win Rate | p-value |
|-----------|----------|-------|----------|---------|
| Poker Soc | 7 | £90.21 | 85.7% | 0.079* |
| Random Home Games | 5 | £135.60 | 60.0% | 0.513 |
| Friend Games | 24 | £49.05 | 45.8% | 0.716 |

*Marginally significant at 10% level

## Project Structure

```
pokeranalysis/
├── README.md
├── poker_analysis.py          # Main analysis script
├── requirements.txt           # Python dependencies
├── data/
│   └── sessions.csv           # Raw session data
├── outputs/
│   ├── *.png                  # Generated charts
│   └── summary_stats.json     # Summary for web integration
└── docs/
    └── methodology.md         # Detailed methodology notes
```

## Usage

```bash
# Clone the repo
git clone https://github.com/Aanu30/pokeranalysis.git
cd pokeranalysis

# Install dependencies
pip install -r requirements.txt

# Run analysis
python poker_analysis.py
```

## Adding New Sessions

1. Add new rows to `data/sessions.csv`
2. Run `python poker_analysis.py`
3. New charts and stats will be generated in `outputs/`

## Dependencies

- Python 3.8+
- pandas
- numpy
- scipy
- matplotlib

## Course Context

This project applies concepts from **ST202 - Probability and Statistical Inference** at the London School of Economics:

- Chapter 8: Estimation, Testing, and Prediction
- Chapter 9: Likelihood-based Inference
- Textbook: Mavrakakis & Penzer, *Probability and Statistical Inference*

## Author

**Aarin Bhatt**  
BSc Data Science, LSE (2027)  
[aarinbhatt.com](https://aarinbhatt.com)

## License

MIT
