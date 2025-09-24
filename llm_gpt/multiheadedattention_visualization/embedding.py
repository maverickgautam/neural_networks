import plotly.graph_objects as go
import numpy as np

# Define vectors
v = np.array([2, 3, 4])
w = np.array([1, 5, -2])
v_plus_w = v + w

# Plot traces
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
    mode='lines', line=dict(color='red', width=5), name='v'
))
fig.add_trace(go.Scatter3d(
    x=[0, w[0]], y=[0, w[1]], z=[0, w[2]],
    mode='lines', line=dict(color='green', width=5), name='w'
))
fig.add_trace(go.Scatter3d(
    x=[0, v_plus_w[0]], y=[0, v_plus_w[1]], z=[0, v_plus_w[2]],
    mode='lines', line=dict(color='blue', width=5), name='v + w'
))
# Optional: add dashed lines to form a parallelogram
fig.add_trace(go.Scatter3d(
    x=[v[0], v_plus_w[0]], y=[v[1], v_plus_w[1]], z=[v[2], v_plus_w[2]],
    mode='lines', line=dict(color='green', width=2, dash='dot'), showlegend=False
))
fig.add_trace(go.Scatter3d(
    x=[w[0], v_plus_w[0]], y=[w[1], v_plus_w[1]], z=[w[2], v_plus_w[2]],
    mode='lines', line=dict(color='red', width=2, dash='dot'), showlegend=False
))

# Create arrows (markers) for the vector heads
fig.add_trace(go.Scatter3d(x=[v[0]], y=[v[1]], z=[v[2]], mode='markers', marker=dict(size=5, color='red'), name='v head'))
fig.add_trace(go.Scatter3d(x=[w[0]], y=[w[1]], z=[w[2]], mode='markers', marker=dict(size=5, color='green'), name='w head'))
fig.add_trace(go.Scatter3d(x=[v_plus_w[0]], y=[v_plus_w[1]], z=[v_plus_w[2]], mode='markers', marker=dict(size=5, color='blue'), name='v+w head'))

# Configure layout
fig.update_layout(
    title='Interactive 3D Vector Addition (Plotly)',
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
        aspectmode='cube',
    )
)

fig.show()
