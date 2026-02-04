"""
Analysing 35 Poker Sessions: What the Data Says About My Edge
A statistical breakdown of live poker results examining win rates, variance, 
session length patterns, and what the numbers reveal about decision quality versus luck.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#e94560'
plt.rcParams['axes.labelcolor'] = '#eaeaea'
plt.rcParams['text.color'] = '#eaeaea'
plt.rcParams['xtick.color'] = '#eaeaea'
plt.rcParams['ytick.color'] = '#eaeaea'
plt.rcParams['grid.color'] = '#0f3460'
plt.rcParams['font.family'] = 'sans-serif'

# Raw data from sessions
data = [
    ("02/10/25", "Poker Society", "10p/20p", 7, -18),
    ("09/10/25", "Poker Society", "10p/25p", 8, 10.25),
    ("24/10/25", "Random Home Game", "1/1", 8, 69),
    ("30/10/25", "HP DG MB JF RP", "10p/20p", 6, 38),
    ("03/11/25", "HP DG MB RP EB AB", "10p/20p", 7, -17.6),
    ("04/11/25", "HP DG MB RP", "10p/20p", 5, -7.7),
    ("07/11/25", "HP DG MB JF (KD)", "10p/20p", 5.5, -19.5),  # 5-6 averaged
    ("07/11/25", "HP DG MB", "10p/20p", 5, -16.6),
    ("08/11/25", "Random Home Game (MB)", "20p/20p", 6, 138),  # 4-8 averaged
    ("11/11/25", "HP DG MB Zain Avi AJ RS", "10p/20p", 6.5, 37),  # 5-8 averaged
    ("14/11/25", "HP AS DG MB ZH EB", "10p/20p", 5.5, -0.6),
    ("18/11/25", "HP AS DG MA...", "10p/20p", 6, 4.8),
    ("19/11/25", "HP DG AS MB", "10p/20p", 5.5, 10.3),
    ("20/11/25", "HP VC EB", "10p/20p", 4, 5.85),
    ("28/11/25", "HP DG AS ZH RS EB", "10p/20p", 5, -46.4),  # 4-6 averaged
    ("29/11/25", "Random Home Game", "20p/20p", 6, -86.4),  # 5-7 averaged
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
    ("22/01/26", "MB JF AK (Carr)", "10p/20p", 7, 14),  # 6-8 averaged
    ("23/01/26", "Random Home Game", "10p/20p", 6.5, -13),  # 5-8 averaged
    ("27/01/26", "HP MB AS", "10p/20p", 4, 23),
    ("29/01/26", "Poker Society", "5p/10p", 7, 28.65),  # 6-8 averaged
    ("29/01/26", "Poker Society", "20p/20p", 8, 18.8),
    ("02/02/26", "Random Home Game", "50p/50p", 7, 28),
]

# Create DataFrame
df = pd.DataFrame(data, columns=['Date', 'Group', 'Blinds', 'Players', 'Profit'])
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
df = df.sort_values('Date').reset_index(drop=True)

# Categorise game types
def categorise_game(group):
    if 'Poker Society' in group or 'Hindu Society' in group:
        return 'Poker Soc'
    elif 'Random Home Game' in group:
        return 'Random Home Games'
    else:
        return 'Friend Games'

df['Game_Type'] = df['Group'].apply(categorise_game)

# Calculate cumulative profit
df['Cumulative_Profit'] = df['Profit'].cumsum()
df['Session_Number'] = range(1, len(df) + 1)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("=" * 70)
print("ANALYSING 35 POKER SESSIONS: WHAT THE DATA SAYS ABOUT MY EDGE")
print("=" * 70)
print()

total_sessions = len(df)
total_profit = df['Profit'].sum()
win_sessions = (df['Profit'] > 0).sum()
lose_sessions = (df['Profit'] < 0).sum()
break_even = (df['Profit'] == 0).sum()
win_rate = win_sessions / total_sessions * 100

avg_profit = df['Profit'].mean()
median_profit = df['Profit'].median()
std_dev = df['Profit'].std()

avg_win = df[df['Profit'] > 0]['Profit'].mean()
avg_loss = df[df['Profit'] < 0]['Profit'].mean()

biggest_win = df['Profit'].max()
biggest_loss = df['Profit'].min()

print("📊 CORE STATISTICS")
print("-" * 40)
print(f"Total Sessions:          {total_sessions}")
print(f"Total Profit:            £{total_profit:.2f}")
print(f"Winning Sessions:        {win_sessions} ({win_rate:.1f}%)")
print(f"Losing Sessions:         {lose_sessions} ({100-win_rate:.1f}%)")
print()
print(f"Average Profit/Session:  £{avg_profit:.2f}")
print(f"Median Profit/Session:   £{median_profit:.2f}")
print(f"Standard Deviation:      £{std_dev:.2f}")
print()
print(f"Average Win:             £{avg_win:.2f}")
print(f"Average Loss:            £{avg_loss:.2f}")
print(f"Win/Loss Ratio:          {abs(avg_win/avg_loss):.2f}")
print()
print(f"Biggest Win:             £{biggest_win:.2f}")
print(f"Biggest Loss:            £{biggest_loss:.2f}")
print()

# ============================================================================
# VARIANCE & RISK ANALYSIS
# ============================================================================
print("📈 VARIANCE & RISK ANALYSIS")
print("-" * 40)

# Calculate downswings
cumulative = df['Cumulative_Profit'].values
peak = np.maximum.accumulate(cumulative)
drawdown = cumulative - peak
max_drawdown = drawdown.min()
max_drawdown_idx = drawdown.argmin()

# Find the peak before max drawdown
peak_before_dd = peak[max_drawdown_idx]

print(f"Maximum Drawdown:        £{max_drawdown:.2f}")
print(f"Peak Before Drawdown:    £{peak_before_dd:.2f}")

# Coefficient of Variation
cv = (std_dev / abs(avg_profit)) * 100 if avg_profit != 0 else float('inf')
print(f"Coefficient of Variation: {cv:.1f}%")

# Sharpe-like ratio (profit per unit of risk)
sharpe_like = avg_profit / std_dev if std_dev != 0 else 0
print(f"Risk-Adjusted Return:    {sharpe_like:.3f}")

# Consecutive wins/losses
df['Win'] = df['Profit'] > 0
streaks = []
current_streak = 1
for i in range(1, len(df)):
    if df.iloc[i]['Win'] == df.iloc[i-1]['Win']:
        current_streak += 1
    else:
        streaks.append((df.iloc[i-1]['Win'], current_streak))
        current_streak = 1
streaks.append((df.iloc[-1]['Win'], current_streak))

win_streaks = [s[1] for s in streaks if s[0]]
loss_streaks = [s[1] for s in streaks if not s[0]]

print(f"Longest Win Streak:      {max(win_streaks) if win_streaks else 0}")
print(f"Longest Loss Streak:     {max(loss_streaks) if loss_streaks else 0}")
print()

# ============================================================================
# GAME TYPE ANALYSIS
# ============================================================================
print("🎯 PERFORMANCE BY GAME TYPE")
print("-" * 40)

game_type_stats = df.groupby('Game_Type').agg({
    'Profit': ['sum', 'mean', 'std', 'count']
}).round(2)

game_type_stats.columns = ['Total', 'Avg', 'Std', 'Sessions']
game_type_stats['Win_Rate'] = df.groupby('Game_Type').apply(
    lambda x: (x['Profit'] > 0).sum() / len(x) * 100
).round(1)

for game_type in game_type_stats.index:
    stats = game_type_stats.loc[game_type]
    print(f"\n{game_type}:")
    print(f"  Sessions: {int(stats['Sessions'])}")
    print(f"  Total: £{stats['Total']:.2f}")
    print(f"  Average: £{stats['Avg']:.2f}")
    print(f"  Win Rate: {stats['Win_Rate']:.1f}%")

print()

# ============================================================================
# TABLE SIZE ANALYSIS
# ============================================================================
print("👥 PERFORMANCE BY TABLE SIZE")
print("-" * 40)

df['Players_Binned'] = pd.cut(df['Players'], bins=[0, 4, 6, 10], labels=['Small (2-4)', 'Medium (5-6)', 'Large (7+)'])
table_stats = df.groupby('Players_Binned').agg({
    'Profit': ['sum', 'mean', 'count']
}).round(2)
table_stats.columns = ['Total', 'Avg', 'Sessions']

for size in table_stats.index:
    stats = table_stats.loc[size]
    print(f"{size}: {int(stats['Sessions'])} sessions, £{stats['Total']:.2f} total, £{stats['Avg']:.2f} avg")
print()

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================
print("📅 TEMPORAL PATTERNS")
print("-" * 40)

df['Day_of_Week'] = df['Date'].dt.day_name()
df['Month'] = df['Date'].dt.to_period('M')

day_stats = df.groupby('Day_of_Week')['Profit'].agg(['sum', 'mean', 'count']).round(2)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_stats = day_stats.reindex([d for d in day_order if d in day_stats.index])

print("\nBy Day of Week:")
for day in day_stats.index:
    stats = day_stats.loc[day]
    print(f"  {day}: {int(stats['count'])} sessions, £{stats['sum']:.2f} total")

monthly_profit = df.groupby('Month')['Profit'].sum()
print("\nBy Month:")
for month, profit in monthly_profit.items():
    print(f"  {month}: £{profit:.2f}")
print()

# ============================================================================
# STATISTICAL SIGNIFICANCE
# ============================================================================
print("🔬 IS THE EDGE REAL? STATISTICAL ANALYSIS")
print("-" * 40)

# One-sample t-test (is mean significantly different from 0?)
from scipy import stats as scipy_stats

t_stat, p_value = scipy_stats.ttest_1samp(df['Profit'], 0)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Statistically significant at 5% level ✓")
elif p_value < 0.1:
    print("Result: Marginally significant at 10% level")
else:
    print("Result: Not statistically significant")

# Confidence interval for true win rate
confidence_level = 0.95
n = total_sessions
p_hat = win_rate / 100
se = np.sqrt(p_hat * (1 - p_hat) / n)
z = scipy_stats.norm.ppf((1 + confidence_level) / 2)
ci_lower = (p_hat - z * se) * 100
ci_upper = (p_hat + z * se) * 100

print(f"\n95% Confidence Interval for Win Rate:")
print(f"  {ci_lower:.1f}% - {ci_upper:.1f}%")

# Bootstrap confidence interval for average profit
np.random.seed(42)
n_bootstrap = 10000
bootstrap_means = []
for _ in range(n_bootstrap):
    sample = np.random.choice(df['Profit'].values, size=len(df), replace=True)
    bootstrap_means.append(np.mean(sample))

ci_profit_lower = np.percentile(bootstrap_means, 2.5)
ci_profit_upper = np.percentile(bootstrap_means, 97.5)

print(f"\n95% Confidence Interval for Average Profit:")
print(f"  £{ci_profit_lower:.2f} - £{ci_profit_upper:.2f}")

# Probability of being a winning player
prob_winning = (np.array(bootstrap_means) > 0).mean() * 100
print(f"\nProbability of being a winning player: {prob_winning:.1f}%")
print()

# ============================================================================
# LUCK VS SKILL DECOMPOSITION
# ============================================================================
print("🎲 LUCK VS SKILL: WHAT PORTION IS VARIANCE?")
print("-" * 40)

# Expected sessions to confirm edge
# Using formula: n = (z * σ / E)² where E is the margin of error we'd accept
margin_of_error = 5  # £5
z_score = 1.96  # 95% confidence
sessions_needed = (z_score * std_dev / margin_of_error) ** 2
print(f"Sessions needed to confirm edge (±£5): {sessions_needed:.0f}")

# What if results were pure luck? (binomial simulation)
np.random.seed(42)
simulations = 10000
sim_results = []
for _ in range(simulations):
    # Simulate 35 sessions with 50% win rate, using actual win/loss amounts
    wins = np.random.binomial(35, 0.5)
    losses = 35 - wins
    sim_profit = wins * avg_win + losses * avg_loss
    sim_results.append(sim_profit)

percentile = (np.array(sim_results) < total_profit).mean() * 100
print(f"\nIf 50/50 luck, your results beat {percentile:.1f}% of simulations")
print(f"This suggests {percentile:.1f}% of your profit is skill-attributed")
print()

# ============================================================================
# CREATE VISUALISATIONS
# ============================================================================
print("Creating visualisations...")

# Figure 1: Cumulative Profit Over Time
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.fill_between(df['Session_Number'], 0, df['Cumulative_Profit'], 
                  where=(df['Cumulative_Profit'] >= 0), alpha=0.3, color='#00d9ff')
ax1.fill_between(df['Session_Number'], 0, df['Cumulative_Profit'], 
                  where=(df['Cumulative_Profit'] < 0), alpha=0.3, color='#e94560')
ax1.plot(df['Session_Number'], df['Cumulative_Profit'], color='#00d9ff', linewidth=2.5)
ax1.axhline(y=0, color='#e94560', linestyle='--', alpha=0.5)
ax1.scatter(df['Session_Number'], df['Cumulative_Profit'], 
            c=np.where(df['Profit'] > 0, '#00ff88', '#ff4466'), s=50, zorder=5, edgecolors='white', linewidth=0.5)
ax1.set_xlabel('Session Number', fontsize=12)
ax1.set_ylabel('Cumulative Profit (£)', fontsize=12)
ax1.set_title('Cumulative Profit Over 35 Sessions', fontsize=16, fontweight='bold', pad=20)

# Add annotations
ax1.annotate(f'Total: £{total_profit:.2f}', 
             xy=(df['Session_Number'].iloc[-1], df['Cumulative_Profit'].iloc[-1]),
             xytext=(10, 10), textcoords='offset points',
             fontsize=11, color='#00d9ff', fontweight='bold')

plt.tight_layout()
fig1.savefig('/home/claude/01_cumulative_profit.png', dpi=150, bbox_inches='tight', 
             facecolor='#1a1a2e', edgecolor='none')
plt.close()

# Figure 2: Session Profit Distribution
fig2, ax2 = plt.subplots(figsize=(10, 6))
colors = ['#00ff88' if x > 0 else '#ff4466' for x in df['Profit']]
bars = ax2.bar(df['Session_Number'], df['Profit'], color=colors, edgecolor='white', linewidth=0.5)
ax2.axhline(y=0, color='white', linestyle='-', alpha=0.3)
ax2.axhline(y=avg_profit, color='#00d9ff', linestyle='--', linewidth=2, label=f'Mean: £{avg_profit:.2f}')
ax2.set_xlabel('Session Number', fontsize=12)
ax2.set_ylabel('Profit/Loss (£)', fontsize=12)
ax2.set_title('Individual Session Results', fontsize=16, fontweight='bold', pad=20)
ax2.legend(loc='upper left', facecolor='#16213e', edgecolor='#e94560')
plt.tight_layout()
fig2.savefig('/home/claude/02_session_profits.png', dpi=150, bbox_inches='tight',
             facecolor='#1a1a2e', edgecolor='none')
plt.close()

# Figure 3: Profit Distribution Histogram
fig3, ax3 = plt.subplots(figsize=(10, 6))
n, bins, patches = ax3.hist(df['Profit'], bins=15, edgecolor='white', linewidth=0.5, alpha=0.7)
for i, patch in enumerate(patches):
    if bins[i] >= 0:
        patch.set_facecolor('#00ff88')
    else:
        patch.set_facecolor('#ff4466')
ax3.axvline(x=0, color='white', linestyle='-', alpha=0.5)
ax3.axvline(x=avg_profit, color='#00d9ff', linestyle='--', linewidth=2, label=f'Mean: £{avg_profit:.2f}')
ax3.axvline(x=median_profit, color='#ffd700', linestyle='--', linewidth=2, label=f'Median: £{median_profit:.2f}')
ax3.set_xlabel('Profit/Loss (£)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Distribution of Session Results', fontsize=16, fontweight='bold', pad=20)
ax3.legend(loc='upper right', facecolor='#16213e', edgecolor='#e94560')
plt.tight_layout()
fig3.savefig('/home/claude/03_profit_distribution.png', dpi=150, bbox_inches='tight',
             facecolor='#1a1a2e', edgecolor='none')
plt.close()

# Figure 4: Performance by Game Type
fig4, axes = plt.subplots(1, 2, figsize=(14, 6))

# Total profit by game type
game_totals = df.groupby('Game_Type')['Profit'].sum().sort_values()
colors_bar = ['#ff4466' if x < 0 else '#00ff88' for x in game_totals.values]
axes[0].barh(game_totals.index, game_totals.values, color=colors_bar, edgecolor='white', linewidth=0.5)
axes[0].axvline(x=0, color='white', linestyle='-', alpha=0.3)
axes[0].set_xlabel('Total Profit (£)', fontsize=12)
axes[0].set_title('Total Profit by Game Type', fontsize=14, fontweight='bold')
for i, (idx, val) in enumerate(game_totals.items()):
    axes[0].text(val + 2 if val >= 0 else val - 2, i, f'£{val:.0f}', 
                 va='center', ha='left' if val >= 0 else 'right', fontsize=10, color='white')

# Win rate by game type
win_rates = df.groupby('Game_Type').apply(lambda x: (x['Profit'] > 0).sum() / len(x) * 100)
colors_wr = ['#00ff88' if x > 50 else '#ffd700' if x == 50 else '#ff4466' for x in win_rates.values]
axes[1].barh(win_rates.index, win_rates.values, color=colors_wr, edgecolor='white', linewidth=0.5)
axes[1].axvline(x=50, color='white', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Win Rate (%)', fontsize=12)
axes[1].set_title('Win Rate by Game Type', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 100)
for i, (idx, val) in enumerate(win_rates.items()):
    axes[1].text(val + 2, i, f'{val:.0f}%', va='center', fontsize=10, color='white')

plt.tight_layout()
fig4.savefig('/home/claude/04_game_type_analysis.png', dpi=150, bbox_inches='tight',
             facecolor='#1a1a2e', edgecolor='none')
plt.close()

# Figure 5: Table Size vs Performance
fig5, ax5 = plt.subplots(figsize=(10, 6))
scatter_colors = np.where(df['Profit'] > 0, '#00ff88', '#ff4466')
sizes = np.abs(df['Profit']) * 3 + 30
ax5.scatter(df['Players'], df['Profit'], c=scatter_colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=0.5)
ax5.axhline(y=0, color='white', linestyle='--', alpha=0.3)

# Add trend line
z = np.polyfit(df['Players'], df['Profit'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['Players'].min(), df['Players'].max(), 100)
ax5.plot(x_line, p(x_line), color='#ffd700', linestyle='--', linewidth=2, label=f'Trend')

ax5.set_xlabel('Number of Players', fontsize=12)
ax5.set_ylabel('Profit/Loss (£)', fontsize=12)
ax5.set_title('Profit vs Table Size', fontsize=16, fontweight='bold', pad=20)
ax5.legend(loc='upper right', facecolor='#16213e', edgecolor='#e94560')
plt.tight_layout()
fig5.savefig('/home/claude/05_table_size_performance.png', dpi=150, bbox_inches='tight',
             facecolor='#1a1a2e', edgecolor='none')
plt.close()

# Figure 6: Drawdown Analysis
fig6, ax6 = plt.subplots(figsize=(12, 6))
ax6.fill_between(df['Session_Number'], drawdown, 0, alpha=0.5, color='#e94560')
ax6.plot(df['Session_Number'], drawdown, color='#e94560', linewidth=2)
ax6.set_xlabel('Session Number', fontsize=12)
ax6.set_ylabel('Drawdown (£)', fontsize=12)
ax6.set_title('Drawdown from Peak Profit', fontsize=16, fontweight='bold', pad=20)

# Mark maximum drawdown
ax6.scatter([max_drawdown_idx + 1], [max_drawdown], color='#ffd700', s=150, zorder=5, marker='v')
ax6.annotate(f'Max Drawdown: £{max_drawdown:.2f}', 
             xy=(max_drawdown_idx + 1, max_drawdown),
             xytext=(10, -20), textcoords='offset points',
             fontsize=11, color='#ffd700', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#ffd700'))

plt.tight_layout()
fig6.savefig('/home/claude/06_drawdown_analysis.png', dpi=150, bbox_inches='tight',
             facecolor='#1a1a2e', edgecolor='none')
plt.close()

# Figure 7: Bootstrap Distribution
fig7, ax7 = plt.subplots(figsize=(10, 6))
ax7.hist(bootstrap_means, bins=50, color='#00d9ff', alpha=0.7, edgecolor='white', linewidth=0.3)
ax7.axvline(x=0, color='#e94560', linestyle='--', linewidth=2, label='Break-even')
ax7.axvline(x=avg_profit, color='#00ff88', linestyle='-', linewidth=2, label=f'Observed Mean: £{avg_profit:.2f}')
ax7.axvline(x=ci_profit_lower, color='#ffd700', linestyle=':', linewidth=2)
ax7.axvline(x=ci_profit_upper, color='#ffd700', linestyle=':', linewidth=2, label=f'95% CI: £{ci_profit_lower:.2f} - £{ci_profit_upper:.2f}')
ax7.fill_betweenx([0, ax7.get_ylim()[1]], ci_profit_lower, ci_profit_upper, alpha=0.2, color='#ffd700')
ax7.set_xlabel('Average Profit per Session (£)', fontsize=12)
ax7.set_ylabel('Frequency', fontsize=12)
ax7.set_title('Bootstrap Distribution of Expected Profit', fontsize=16, fontweight='bold', pad=20)
ax7.legend(loc='upper right', facecolor='#16213e', edgecolor='#e94560')
plt.tight_layout()
fig7.savefig('/home/claude/07_bootstrap_distribution.png', dpi=150, bbox_inches='tight',
             facecolor='#1a1a2e', edgecolor='none')
plt.close()

# Figure 8: Monthly Trend
fig8, ax8 = plt.subplots(figsize=(12, 6))
monthly_data = df.groupby('Month').agg({
    'Profit': ['sum', 'count', 'mean']
}).reset_index()
monthly_data.columns = ['Month', 'Total', 'Sessions', 'Average']
monthly_data['Month_str'] = monthly_data['Month'].astype(str)

colors_monthly = ['#00ff88' if x > 0 else '#ff4466' for x in monthly_data['Total']]
bars = ax8.bar(monthly_data['Month_str'], monthly_data['Total'], color=colors_monthly, 
               edgecolor='white', linewidth=0.5)
ax8.axhline(y=0, color='white', linestyle='-', alpha=0.3)

# Add session count labels
for i, (bar, sessions) in enumerate(zip(bars, monthly_data['Sessions'])):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + 5 if height >= 0 else height - 15,
             f'n={int(sessions)}', ha='center', va='bottom' if height >= 0 else 'top',
             fontsize=9, color='#eaeaea')

ax8.set_xlabel('Month', fontsize=12)
ax8.set_ylabel('Total Profit (£)', fontsize=12)
ax8.set_title('Monthly Profit Trend', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45)
plt.tight_layout()
fig8.savefig('/home/claude/08_monthly_trend.png', dpi=150, bbox_inches='tight',
             facecolor='#1a1a2e', edgecolor='none')
plt.close()

print("✅ All visualisations saved!")
print()

# ============================================================================
# FINAL SUMMARY & CONCLUSIONS
# ============================================================================
print("=" * 70)
print("CONCLUSIONS: WHAT THE DATA SAYS ABOUT YOUR EDGE")
print("=" * 70)
print()
print(f"Over {total_sessions} sessions, you've accumulated £{total_profit:.2f} in profit.")
print(f"Your win rate of {win_rate:.1f}% and average profit of £{avg_profit:.2f}/session")
print(f"suggests a positive edge, though variance (σ = £{std_dev:.2f}) is significant.")
print()
print("Key findings:")
print(f"  • {prob_winning:.0f}% probability you're a long-term winning player")
print(f"  • Your results beat {percentile:.0f}% of what pure luck would produce")
print(f"  • Maximum drawdown of £{abs(max_drawdown):.2f} shows manageable risk")
print(f"  • Need ~{sessions_needed:.0f} more sessions to statistically confirm your edge")
print()

if p_value < 0.05:
    print("📊 VERDICT: Your edge appears REAL and statistically significant.")
elif p_value < 0.1:
    print("📊 VERDICT: Promising results, but more sessions needed to confirm edge.")
else:
    print("📊 VERDICT: Sample size too small to draw firm conclusions.")

print()
print("Charts saved to /home/claude/")

# Save summary statistics to CSV
summary_df = pd.DataFrame({
    'Metric': ['Total Sessions', 'Total Profit', 'Win Rate', 'Avg Profit/Session', 
               'Std Deviation', 'Biggest Win', 'Biggest Loss', 'Max Drawdown',
               'Win/Loss Ratio', 'P-value (t-test)'],
    'Value': [total_sessions, f'£{total_profit:.2f}', f'{win_rate:.1f}%', f'£{avg_profit:.2f}',
              f'£{std_dev:.2f}', f'£{biggest_win:.2f}', f'£{biggest_loss:.2f}', 
              f'£{max_drawdown:.2f}', f'{abs(avg_win/avg_loss):.2f}', f'{p_value:.4f}']
})
summary_df.to_csv('/home/claude/summary_stats.csv', index=False)

# Save full data
df.to_csv('/home/claude/poker_sessions_data.csv', index=False)
print("\n📁 Data exported to CSV files")
