import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_plotly_events import plotly_events
import plotly.io as pio
from utils import compare_responses

pio.templates.default = "plotly"

st.set_page_config(page_title="Analysis Tool", page_icon="📊", layout="wide")
st.title("Multivariate Response Analysis Tool")

df: pd.DataFrame = pd.read_pickle("response_data.pkl")
df =df.replace("Don't Remember", np.nan)

columns_df = pd.read_excel("columns.xlsx")

st.subheader("Dataset Viewing")
st.write("""
         In this section you may look at the dataset collected.
         Within the first expander you will find the questions used to populate the keys provided in this dataset.
         Each respondent was allowed to provide up to 5 responses per submission.
         Each response is included as a separate entry with their ID column allowing for later collation of data to the original dataset.
         """
    )
view_selection = st.radio("Dataset View", options=["None", "Selected Keys", "Full Dataset"], index=0, horizontal=True)

with st.expander("Open to view questions used to populate columns", expanded=False):
    st.write(columns_df)

view_keys = []
if view_selection == "None":
    pass
elif view_selection == "Selected Keys":
    view_keys = st.multiselect("Select Keys to view", df.keys())
    st.write(df[view_keys])
elif view_selection == "Full Dataset":
    st.write(df)

st.subheader("Data Plotting")
st.markdown(
    """
    In this section, first select keys from the dataset to visualize.
    * If only one key is selected, a histogram of responses is shown.
    * With more than one key selected, the user is then presented a [scatter plot matrix (SPLOM) arrangement of data](https://medium.com/plotly/what-is-a-splom-chart-make-scatterplot-matrices-in-python-8dc4998921c3) to investigate relationships between client responses.
    * Finally, you may select a point from any combination of plots and the dataset will be queried to show responses containing the selected traits subsequent to the plotting field.
    * Refresh the page to begin a new point selection (future features will bring a refresh button)
    """ 
)
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
                    tmpdf = df[[k1, k2]]
                    tt = tmpdf.groupby(tmpdf.columns.tolist(),as_index=False).size()
                    fig.add_trace(
                        go.Scatter(
                            x=tt[k1],
                            y=tt[k2],
                            mode="markers",
                            text = [f"Count: {cval}" for cval in tt["size"]],
                            marker=dict(
                                color=tt["size"], #set color equal to a variable
                                colorscale='portland', # one of plotly colorscales
                                showscale=True
                            ),
                        )
                    )
                
                tmp_points = plotly_events(fig, click_event=True, key=f"interactive-figure-{ii}_{jj}")

                if tmp_points:
                    tmp_points[0]["kx"] = k1
                    tmp_points[0]["ky"] = k2
                    selected_points.append(tmp_points[0])
    if selected_points:
        st.write("The following individuals were determined to fit your selected criterion")
        st.write(df.iloc[list(set.intersection(*[set(df[df[point["kx"]] == point["x"]].index.tolist()) for point in selected_points]))])

st.subheader("Data Comparator")
st.markdown(
    """
    In this section we perform a cross correlation analysis to determine similarities between responses

    """ 
)

compare_keys_default = ['nickname', 'years', 'gender', 'race', 'age', 'height', 'bodybuild', 'bodyframe', 'hair', 'hairstyle', 'hairtype', 'eye', 'glasses', 'voice', 'smell', 'notablefeatures', 'tattoo', 'jewelry', 'worktime', 'relation', 'location']
compare_keys = st.multiselect("Keys to analyze", options=compare_keys_default, default=compare_keys_default)
threshold = st.slider("Similarity Threshold [%]", min_value=0, max_value=100, step=1, value=40, format="%d%%")

similarity_matrix, score_matrix = compare_responses(df=df, compare_keys=compare_keys)

xi, yi = np.where(similarity_matrix > threshold)

best_indx = np.argmax(np.max(similarity_matrix[xi, yi]))
best_value = similarity_matrix[xi[best_indx], yi[best_indx]]
st.write(f"Found {len(xi)} pairs with at least {threshold}% similarity. The best similarity was {best_value}% similarity for respondents {xi[best_indx]} and {yi[best_indx]}.")

with st.expander("Similarity Response Diagram"):
    fig = px.imshow(similarity_matrix)
    st.plotly_chart(fig, use_container_width=True)
with st.expander("Show Similarity Results"):
    for ii, jj in zip(xi, yi):
        s = score_matrix[ii, jj]
        sk = [k for k, v in s.items() if v]
        x1 = df.loc[ii]
        x2 = df.loc[jj]
        ss = pd.Series(x1[sk].values, index=sk)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.dataframe(x1, use_container_width=True)
        with c2:
            st.dataframe(x2, use_container_width=True)
        with c3:
            st.json(dict(ss))
        