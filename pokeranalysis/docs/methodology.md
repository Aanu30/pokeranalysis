# Methodology

## Overview

This analysis applies classical statistical inference methods to determine whether observed poker results indicate genuine skill (positive expected value) or are consistent with random chance.

## Data Collection

- **Period:** October 2025 - February 2026
- **Sessions:** 36 live cash game sessions
- **Stakes:** Primarily 10p/20p blinds with £10 buy-ins
- **Venues:** University poker society events, home games with friends, random home games

## Statistical Framework

### 1. Hypothesis Testing

We test the null hypothesis that the true mean profit per session is zero:

$$H_0: \mu = 0 \quad \text{vs} \quad H_1: \mu \neq 0$$

Using the t-statistic:

$$t = \frac{\bar{x} - 0}{s / \sqrt{n}}$$

where $\bar{x}$ is the sample mean, $s$ is the sample standard deviation, and $n$ is the number of sessions.

### 2. Confidence Intervals

The $(1-\alpha)$ confidence interval for the true mean is:

$$\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

### 3. Maximum Likelihood Estimation

We fit two distributions to the profit data:

**Normal Distribution:**
$$f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

MLEs: $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$

**Student's t-Distribution:**
Accounts for heavier tails observed in the data (occasional large wins/losses).

Model comparison uses AIC and BIC.

### 4. Likelihood Ratio Test

Testing $H_0: \mu = 0$ against $H_1: \mu \neq 0$:

$$\Lambda = \frac{L(\mu=0, \hat{\sigma}_0)}{L(\hat{\mu}, \hat{\sigma})}$$

Under $H_0$, $-2\log(\Lambda) \sim \chi^2(1)$ by Wilks' theorem.

### 5. Power Analysis

Power is the probability of correctly rejecting $H_0$ when $H_1$ is true:

$$\text{Power} = P(\text{Reject } H_0 | H_1 \text{ true})$$

We calculate:
- Power of current test given observed effect size
- Sample size required for 80% and 90% power
- Minimum detectable effect size

### 6. Bootstrap Inference

Non-parametric bootstrap with B = 10,000 resamples provides:
- Distribution-free confidence intervals
- Robust standard error estimates
- P-values without distributional assumptions

### 7. Bayesian Analysis

Using conjugate Normal-Normal model:

**Prior:** $\mu \sim N(\mu_0, \sigma_0^2)$

**Posterior:** $\mu | X \sim N(\mu_n, \sigma_n^2)$

where:
$$\mu_n = \frac{\sigma^{-2}\sum x_i + \sigma_0^{-2}\mu_0}{\sigma^{-2}n + \sigma_0^{-2}}$$

We test sensitivity to three priors:
- **Skeptical:** $N(0, 20^2)$ — centred at no edge
- **Neutral:** $N(0, 50^2)$ — wide uncertainty
- **Optimistic:** $N(10, 30^2)$ — slight positive prior

## Limitations

1. **Sample size:** 36 sessions provides limited statistical power for detecting small effects
2. **Selection bias:** Only sessions where I played are recorded
3. **Stake variation:** Most sessions at 10p/20p, but some at higher stakes
4. **Player pool variation:** Different opponents across game types
5. **Non-stationarity:** Skill may improve over time

## References

- Mavrakakis, M. C., & Penzer, J. (2021). *Probability and Statistical Inference: From Basic Principles to Advanced Models*. CRC Press.
- Casella, G., & Berger, R. L. (2002). *Statistical Inference*. Duxbury.
