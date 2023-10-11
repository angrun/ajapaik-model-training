
import plotly.express as px
import pandas as pd

import plotly.io as pio

# Sample data
df = pd.DataFrame(dict(
    group=["True Positive", "True Negative", "False Positive", "False Negative"],
    value=[461, 432, 62, 89]))

fig = px.bar(df, x='group', y='value', color='group',
             title="Data Quality Engine v3 feedbacks processing",
             color_discrete_map={
                 'True Positive': '#96d665',
                 'True Negative': '#96d665',
                 'False Positive': '#db8467',
                 'False Negative': '#db8467'
             })

fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis_title="Total Feedbacks",
    bargap=0.5,  # Adjust the gap between columns
    bargroupgap=0.1,  # Adjust the gap between groups of columns
)

fig.update_traces(showlegend=False)

fig.show()

pio.write_image(fig, 'outputs/accuracy_plot.jpg')
