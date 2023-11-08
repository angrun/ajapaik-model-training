import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Sample data for initial and retrained models
categories = ["scene", "viewpoint elevation"]
initial_model_accuracy = [85.03, 80.07]
retrained_model_accuracy = [85.22, 80.64]

# New colors
initial_model_color = "#fd7f6f"
retrained_model_color = "#b2e061"

# Plot the data
width = 0.35
x = range(len(categories))
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
initial_bars = ax.bar(x, initial_model_accuracy, width, label='Trained Model', color=initial_model_color)
retrained_bars = ax.bar([i + width for i in x], retrained_model_accuracy, width, label='Retrained Model', color=retrained_model_color)

ax.set_xlabel('Image Category Class')
ax.set_ylabel('Accuracy')
ax.set_title('Trained and Retrained Model Accuracy per Image Category Class', fontsize=16, pad=20)  # Added 'pad' for spacing after the title
ax.set_xticks([i + width / 2 for i in x])
ax.set_xticklabels(categories)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)  # Adjusted 'bbox_to_anchor' for the legend
ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines

# Set the background to white
ax.set_facecolor('white')

# Customize the chart border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Add numeric values to the bars
for i, bar in enumerate(initial_bars):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{initial_model_accuracy[i]:.2f}",
            ha='center', va='bottom', fontsize=10)

for i, bar in enumerate(retrained_bars):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{retrained_model_accuracy[i]:.2f}",
            ha='center', va='bottom', fontsize=10)

# Save the chart to a file
plt.savefig('accuracy_comparison.png', bbox_inches='tight', dpi=300)  # Adjust DPI and file format as needed
plt.show()
