import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd


def render_keyword_network(cooccurrence_df: pd.DataFrame):
    """
    Renders an interactive network graph from a hashtag co-occurrence dataframe.
    Expects columns: hashtag_1, hashtag_2, count
    """
    if cooccurrence_df.empty:
        st.info("Not enough co-occurring hashtags to build a network graph.")
        return

    G = nx.Graph()
    for _, row in cooccurrence_df.iterrows():
        G.add_edge(row["hashtag_1"], row["hashtag_2"], weight=row["count"])

    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Build edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    # Build node traces
    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"#{node}")
        node_size.append(10 + G.degree(node) * 5)  # bigger node = more connections

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color="#1f77b4",
            line=dict(width=2, color="white")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)