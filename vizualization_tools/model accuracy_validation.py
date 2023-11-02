import plotly.express as px
import plotly.io as pio

# Number of total images
total_images = 3057
# total_images = 3555

# Number of correctly predicted images
# correct_predictions = 955
# correct_predictions = 3
# correct_predictions = 3100 + 2500 + 2100
# correct_predictions = 305
# correct_predictions = 285
correct_predictions = 400 + 285 + 305

# Calculate the accuracy percentage
accuracy_percentage = (correct_predictions / total_images) * 100

# Create a bar chart
fig = px.bar(
    x=["Correct Predictions", "Incorrect Predictions"],
    y=[correct_predictions, total_images - correct_predictions],
    labels={"x": "Prediction Results", "y": "Number of Images"},
    title=f"Model Image Prediction Accuracy [total] ({accuracy_percentage:.2f}%)",
    color_discrete_sequence=["#72cf7e", "#72cf7e"]
)

# Customize the background color to white
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis_title="Total Images",
    bargap=0.5,  # Adjust the gap between columns
    bargroupgap=0.1,  # Adjust the gap between groups of columns
)

# Adjust the width of the columns
fig.update_xaxes(categoryarray=["Correct Predictions", "Incorrect Predictions"], title=None)

# Save the figure as a .jpg image
pio.write_image(fig, 'outputs/accuracy_plot.jpg')

# Save the figure as a .png image
pio.write_image(fig, 'outputs/accuracy_plot.png')
