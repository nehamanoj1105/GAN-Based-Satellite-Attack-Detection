import streamlit as st
import pandas as pd
import os
import sys
import plotly.graph_objects as go

# -----------------------------
# FIX IMPORT PATH
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.server import run_detection

# -----------------------------
# UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("🛰️ SAT-GUARD Dashboard")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    if st.button("Run Detection"):
        try:
            result = run_detection(df)

            st.subheader("Detection Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Max Error", round(result["max_reconstruction_error"], 4))
            col2.metric("Mean Error", round(result["mean_reconstruction_error"], 4))
            col3.metric("Threshold", round(result["threshold"], 4))

            if result["is_anomaly"]:
                st.error("ANOMALY DETECTED")
            else:
                st.success("NORMAL")

            # -----------------------------
            # GRAPH
            # -----------------------------
            scores = result["scores"]
            threshold = result["threshold"]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                y=scores,
                mode="lines",
                name="Reconstruction Error"
            ))

            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold"
            )

            anomaly_idx = [i for i, s in enumerate(scores) if s > threshold]

            if anomaly_idx:
                fig.add_trace(go.Scatter(
                    x=anomaly_idx,
                    y=[scores[i] for i in anomaly_idx],
                    mode="markers",
                    name="Anomalies",
                    marker=dict(color="red", size=8)
                ))

            fig.update_layout(
                title="Reconstruction Error",
                xaxis_title="Time Step",
                yaxis_title="Error",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
