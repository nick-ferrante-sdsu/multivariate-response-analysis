import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_plotly_events import plotly_events
import plotly.io as pio
pio.templates.default = "plotly"

st.set_page_config(page_title="Analysis Tool", page_icon="📊", layout="wide")
st.title("Multivariate Response Analysis Tool")

df: pd.DataFrame = pd.read_pickle("response_data.pkl")
df =df.replace("Don't Remember", np.nan)

columns_df = pd.read_excel("columns.xlsx")

st.subheader("Dataset")
view_selection = st.radio("Dataset View", options=["None", "Selected Keys", "Full Dataset"], index=0, horizontal=True)

view_keys = []
if view_selection == "None":
    pass
elif view_selection == "Selected Keys":
    view_keys = st.multiselect("Select Keys to view", df.keys())
    st.write(df[view_keys])
elif view_selection == "Full Dataset":
    st.write(df)

st.subheader("Data Plotting")
st.text("Please select keys to analyze")
plot_keys_options = [
    "nickname", 
    "gender", 
    "race", 
    "age", 
    "height", 
    "bodybuild", 
    "bodyframe", 
    "hair", 
    "hairstyle", 
    "hairtype", 
    "eye", 
    "glasses", 
    "voice", 
    "smell", 
    "notablefeatures", 
    "tattoo", 
    "jewelry", 
    "worktime", 
    "relation", 
    "location"
]
plot_keys = st.multiselect("", options=plot_keys_options)
N_plot = len(plot_keys)
N_cols = N_plot
N_rows = N_plot

selected_points = []

if N_plot > 0:
    cols = st.columns(N_cols)
    for ii in reversed(range(N_rows)):
        for jj in reversed(range(ii+1)):
            k1 = plot_keys[jj]
            k2 = plot_keys[ii]

            with cols[jj]:
                fig = go.Figure()
                if ii==jj:
                    fig.add_trace(
                        go.Histogram(x=df[k1]),
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df[k1],
                            y=df[k2],
                            mode="markers"
                        )
                    )
                
                tmp_points = plotly_events(fig, click_event=True, key=f"interactive-figure-{ii}_{jj}")
                tmp_points[0]["kx"] = k1
                tmp_points[0]["ky"] = k2
                selected_points.append(tmp_points[0])
    
    st.write(df.iloc[list(set.intersection(*[set(df[df[point["kx"]] == point["x"]].index.tolist()) for point in selected_points]))])
