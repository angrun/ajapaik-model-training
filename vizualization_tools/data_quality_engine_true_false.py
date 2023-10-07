import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend (or use another backend that works for your system)

import matplotlib.pyplot as plt

# Define the data
data = {
    'True Positive': 427,
    'True Negative': 162,
    'False Positive': 453,
    'False Negative': 22
}

# Extract labels and values
labels = list(data.keys())
values = list(data.values())

# Create a bar chart
plt.figure(figsize=(8, 6))
plt.bar(labels, values, color=['green', 'green', 'red', 'red'])
plt.xlabel('Confusion Matrix')
plt.ylabel('Count')
plt.title('Confusion Matrix Visualization')
plt.ylim(0, max(values) + 50)  # Adjust the y-axis limit for better visualization
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Display the values on top of the bars
for i, v in enumerate(values):
    plt.text(i, v + 10, str(v), ha='center', va='bottom', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
