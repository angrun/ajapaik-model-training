import plotly.express as px
import plotly.io as pio

# Define the data
data = {
    'True Positive': 464,
    'True Negative': 74,
    'False Positive': 922,
    'False Negative': 40
}

# Extract labels and values
labels = list(data.keys())
values = list(data.values())

# Specify a color map for each label
color_map = {
    'True Positive': '#96d665',
    'True Negative': '#96d665',
    'False Positive': '#db8467',
    'False Negative': '#db8467'
}

# Create a bar chart with the specified color map
fig = px.bar(
    x=labels,
    y=values,
    labels={"x": "Data Quality Engine v2 feedbacks processing", "y": "Count"},
    title="Data Quality Engine v2 feedbacks processing",
    color=color_map  # Specify color map
)

# Customize the background color to white
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis_title=None,
    yaxis_title=None
)

# Show the plot
fig.show()

# Save the figure as an image (optional)
pio.write_image(fig, 'confusion_matrix_plot.png')
