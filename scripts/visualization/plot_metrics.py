import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Linear Reg', 'CNN', 'ConvLSTM', 'Vanilla ViT', 'Swin ViT', 'Multi-Var ViT']
rmse = [2.20, 2.26, 2.38, 2.28, 2.40, 2.27]
acc = [0.92, 0.92, 0.91, 0.91, 0.90, 0.91]

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot RMSE on primary y-axis
color = 'tab:red'
ax1.set_xlabel('Models', fontweight='bold')
ax1.set_ylabel('RMSE (Lower is Better)', color=color, fontweight='bold')
bars1 = ax1.bar(x - width/2, rmse, width, label='RMSE', color=color, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(2.1, 2.5)

# Plot ACC on secondary y-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('ACC (Higher is Better)', color=color, fontweight='bold')
bars2 = ax2.bar(x + width/2, acc, width, label='ACC', color=color, alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.88, 0.94)

# Add values on top of bars
def autolabel(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(bars1, ax1)
autolabel(bars2, ax2)

plt.title('T850 Prediction Performance (5-day lead time)', fontweight='bold', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15)
fig.tight_layout()

plt.savefig('metrics_comparison.png', dpi=300)
print("Saved metrics_comparison.png")
