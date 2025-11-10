import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Since mermaid service is unavailable, create a network diagram using Plotly
# Define node positions for the Dynamic RL Firewall architecture

# Node definitions with coordinates
nodes = {
    'Network Traffic': (0, 8, '#1FB8CD'),
    'Feature Extract': (0, 6, '#2E8B57'),
    'LSTM-CNN Model': (-2, 4, '#DB4545'),
    'DQN Agent': (2, 4, '#DB4545'),
    'Reward System': (4, 2, '#DB4545'),
    'Rule Engine': (0, 2, '#D2BA4C'),
    'SDN Controller': (0, 0, '#D2BA4C'),
    'Firewall Actions': (0, -2, '#5D878F'),
    'Performance Metrics': (2, -1, '#5D878F'),
    'States (MDP)': (-4, 4, '#B4413C'),
    'Actions (MDP)': (-4, 2, '#B4413C'),
    'Rewards (MDP)': (-4, 0, '#B4413C')
}

# Extract coordinates and create traces
x_coords = [pos[0] for pos in nodes.values()]
y_coords = [pos[1] for pos in nodes.values()]
colors = [pos[2] for pos in nodes.values()]
node_names = list(nodes.keys())

# Create edges (connections)
edges = [
    ('Network Traffic', 'Feature Extract'),
    ('Feature Extract', 'LSTM-CNN Model'),
    ('Feature Extract', 'DQN Agent'),
    ('LSTM-CNN Model', 'Rule Engine'),
    ('DQN Agent', 'Rule Engine'),
    ('Rule Engine', 'SDN Controller'),
    ('SDN Controller', 'Firewall Actions'),
    ('Firewall Actions', 'Performance Metrics'),
    ('Performance Metrics', 'Reward System'),
    ('Reward System', 'DQN Agent'),
    ('Feature Extract', 'States (MDP)'),
    ('Rule Engine', 'Actions (MDP)'),
    ('Performance Metrics', 'Rewards (MDP)'),
    ('States (MDP)', 'DQN Agent'),
    ('Actions (MDP)', 'SDN Controller'),
    ('Rewards (MDP)', 'DQN Agent')
]

# Create edge traces
edge_x = []
edge_y = []
for edge in edges:
    x0, y0, _ = nodes[edge[0]]
    x1, y1, _ = nodes[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# Create the figure
fig = go.Figure()

# Add edges
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=2, color='#333333'),
    hoverinfo='none',
    mode='lines',
    showlegend=False
))

# Add nodes
fig.add_trace(go.Scatter(
    x=x_coords, y=y_coords,
    mode='markers+text',
    marker=dict(
        size=40,
        color=colors,
        line=dict(width=2, color='white')
    ),
    text=node_names,
    textposition="middle center",
    textfont=dict(size=10, color='white'),
    hoverinfo='text',
    hovertext=node_names,
    showlegend=False
))

# Update layout
fig.update_layout(
    title="Dynamic RL Firewall Architecture",
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    annotations=[
        dict(
            x=-4, y=5,
            text="Markov Decision<br>Process (MDP)",
            showarrow=False,
            font=dict(size=12, color='#B4413C')
        ),
        dict(
            x=0, y=9,
            text="Input Layer",
            showarrow=False,
            font=dict(size=10, color='#666')
        ),
        dict(
            x=0, y=5,
            text="AI Core",
            showarrow=False,
            font=dict(size=10, color='#666')
        ),
        dict(
            x=0, y=1,
            text="Decision Layer",
            showarrow=False,
            font=dict(size=10, color='#666')
        ),
        dict(
            x=0, y=-3,
            text="Output & Feedback",
            showarrow=False,
            font=dict(size=10, color='#666')
        )
    ]
)

# Save as PNG and SVG
fig.write_image("rl_firewall_diagram.png")
fig.write_image("rl_firewall_diagram.svg", format="svg")

print("Dynamic RL Firewall architecture diagram created successfully!")