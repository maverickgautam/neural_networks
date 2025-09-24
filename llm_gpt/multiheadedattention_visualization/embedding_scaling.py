import plotly.graph_objects as go
import numpy as np

# Define vector and scalar
v = np.array([2, 3, 4])
s = 2
scaled_v = s * v

# Plot traces
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
    mode='lines', line=dict(color='red', width=5), name='v'
))
fig.add_trace(go.Scatter3d(
    x=[0, scaled_v[0]], y=[0, scaled_v[1]], z=[0, scaled_v[2]],
    mode='lines', line=dict(color='purple', width=5), name='s * v'
))

# Create arrows (markers) for the vector heads
fig.add_trace(go.Scatter3d(x=[v[0]], y=[v[1]], z=[v[2]], mode='markers', marker=dict(size=5, color='red'), name='v head'))
fig.add_trace(go.Scatter3d(x=[scaled_v[0]], y=[scaled_v[1]], z=[scaled_v[2]], mode='markers', marker=dict(size=5, color='purple'), name='s*v head'))

# Configure layout
fig.update_layout(
    title='Interactive 3D Vector Scaling (Plotly)',
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
        aspectmode='cube',
    )
)

fig.show()
