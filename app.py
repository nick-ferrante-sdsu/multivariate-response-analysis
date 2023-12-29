import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

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
N_cols = 2
N_rows = int(np.ceil(N_plot/N_cols))
if N_plot > 0:

    fig = make_subplots(
            rows=N_rows,
            cols=N_cols,
    )
    fig.update_layout(showlegend=False)
    for ii, key in enumerate(plot_keys):
        row, col = ii // N_cols + 1, ii % N_cols + 1
        
        fig.add_trace(
            go.Histogram(x=df[key]),
            row=row, col=col,
        )

    st.plotly_chart(fig)

    fig = go.Figure()
    dimensions=[
                dict(
                    label=key,
                    values=df[key],
                ) for key in plot_keys
            ]
    fig.add_trace(
        go.Splom(
            dimensions=dimensions,
            diagonal_visible=False, # remove plots on diagonal
            showupperhalf=False, # remove plots on diagonal
            )
    )
    st.plotly_chart(fig)

    plot = sns.pairplot(df, vars=plot_keys, corner=True)
    st.pyplot(plot)
