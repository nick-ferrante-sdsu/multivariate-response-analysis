import streamlit as st
import pandas as pd
import numpy as np

def compare_list(x1, x2):
    print(x1, x2)
    for y in x1:
        if y in x2:
            return True
    else:
        return False

@st.cache_data
def compare_responses(df: pd.DataFrame, compare_keys):
    st.spinner("Comparing responses")
    N_sub = len(df)

    similarity_matrix = np.zeros([N_sub, N_sub])
    score_matrix = np.zeros([N_sub, N_sub], dtype=pd.Series)
    for ii in range(N_sub):
        x1 = df.loc[ii]
        for jj in range(ii):
            x2 = df.loc[jj]
            tmp_dict = {}
            for kk in compare_keys:
                v1, v2 = x1[kk], x2[kk]
                tmp_score = 0
                if not pd.isna([v1, v2]).all():
                    if v1 == v2:
                        if type(v1) == list:
                            if compare_list(v1, v2):
                                tmp_score = 1
                        else:
                            tmp_score = 1
                    else:
                        pass
                    
                tmp_dict[kk] = tmp_score
            s = pd.Series(tmp_dict)
            similarity_matrix[ii, jj] = s.sum()
            score_matrix[ii, jj] = s
    similarity_matrix /= len(compare_keys)
    similarity_matrix *= 100
    return similarity_matrix, score_matrix