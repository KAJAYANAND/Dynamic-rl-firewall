import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import stats

# Data for the learning curve
episode_rewards = [-105.23, -98.45, -92.12, -87.34, -82.67, -78.91, -75.23, -71.45, -68.12, -65.34, -62.67, -59.91, -57.23, -54.45, -51.12, -48.34, -45.67, -42.91, -40.23, -37.45, -70.43, -67.12, -63.34, -59.67, -55.91, -52.23, -48.45, -44.12, -40.34, -36.67, -32.91, -29.23, -25.45, -21.12, -17.34, -13.67, -9.91, -6.23, -2.45, 1.12, -43.04, -39.34, -35.67, -31.91, -28.23, -24.45, -20.12, -16.34, -12.67, -8.91, -5.23, -1.45, 2.12, 5.34, 8.67, 12.91, 16.23, 19.45, 22.12, 25.34, -9.15, -5.67, -2.91, 0.23, 3.45, 6.12, 9.34, 12.67, 15.91, 19.23, 22.45, 25.12, 28.34, 31.67, 34.91, 38.23, 41.45, 44.12, 47.34, 50.67, 51.01, 54.23, 57.45, 60.12, 63.34, 66.67, 69.91, 73.23, 76.45, 79.12, 82.34, 85.67, 88.91, 92.23, 95.45, 98.12, 101.34, 104.67, 107.91, 111.23]

episodes = list(range(1, len(episode_rewards) + 1))

# Calculate trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(episodes, episode_rewards)
trend_line = [slope * x + intercept for x in episodes]

# Identify episode reset points (large drops in rewards)
reset_points = []
for i in range(1, len(episode_rewards)):
    if episode_rewards[i] < episode_rewards[i-1] - 30:  # Threshold for reset detection
        reset_points.append(i+1)  # +1 because episodes start from 1

# Create the figure
fig = go.Figure()

# Add the episode rewards line
fig.add_trace(go.Scatter(
    x=episodes,
    y=episode_rewards,
    mode='lines+markers',
    name='Episode Reward',
    line=dict(color='#1FB8CD', width=2),
    marker=dict(size=3),
    hovertemplate='Episode: %{x}<br>Reward: %{y:.2f}<extra></extra>'
))

# Add trend line
fig.add_trace(go.Scatter(
    x=episodes,
    y=trend_line,
    mode='lines',
    name='Learning Trend',
    line=dict(color='#DB4545', width=3, dash='dash'),
    hovertemplate='Episode: %{x}<br>Trend: %{y:.2f}<extra></extra>'
))

# Add markers for episode resets
if reset_points:
    fig.add_trace(go.Scatter(
        x=reset_points,
        y=[episode_rewards[i-1] for i in reset_points],
        mode='markers',
        name='Episode Reset',
        marker=dict(color='#2E8B57', size=8, symbol='diamond'),
        hovertemplate='Reset at Ep: %{x}<br>Reward: %{y:.2f}<extra></extra>'
    ))

# Update layout with better spacing and gridlines
fig.update_layout(
    title='DRL Firewall Learning Progress',
    xaxis_title='Training Episode',
    yaxis_title='Episode Reward',
    legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5),
    showlegend=True,
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        dtick=10
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart as both PNG and SVG
fig.write_image("learning_curve.png")
fig.write_image("learning_curve.svg", format="svg")

print("Improved chart saved successfully!")
print(f"Learning trend: {slope:.3f} reward improvement per episode")
print(f"Episode resets detected at episodes: {reset_points}")