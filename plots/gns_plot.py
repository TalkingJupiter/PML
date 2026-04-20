import matplotlib.pyplot as plt

# Enter your values here
width_multipliers = [1.0, 1.3, 1.6, 1.9, 2.2, 2.8, 3.4]
accuracies = [90.83, 91.53, 92.04, 92.90, 93.67, 93.18, 93.25]

plt.figure(figsize=(10,6))

plt.plot(
    width_multipliers,
    accuracies,
    marker='o',
    linestyle='-',
    linewidth=2,
    markersize=8,
    label="GhostNetV3-Small"
)

# Label each point
for x, y in zip(width_multipliers, accuracies):
    plt.text(x, y + 0.05, f"{y:.2f}%", ha='center', fontsize=12, color='blue')

plt.xlabel("Width Multiplier", fontsize=14, fontweight="bold")
plt.ylabel("Best Test Accuracy (%)", fontsize=14, fontweight="bold")

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=13)

plt.tight_layout()

# Save figure for paper
plt.savefig("ghostnet_width_accuracy.png", dpi=300)

# plt.show()