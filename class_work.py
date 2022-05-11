# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title = "Credit Default App",
    page_icon = "🦈",
    layout = "wide")

@st.cache()
def load_data():
    data = pd.read_csv("prosper_data_app_dev.csv")
    return(data.dropna())

data = load_data()
################################

st.title("💸💸Sharky's Credit Default App💸💸")
st.markdown("this dashboard can be used to **analyze** and **predict** credit default")
st.write("My first Streamlit App to predict credit risk")


###################################

st.header("Customer Explorer")

row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])

rate = row1_col1.slider("Interest to pay",
                 min_value=data["borrower_rate"].min(),
                 max_value=data["borrower_rate"].max(),
                 value=(0.01,0.12)
)
row1_col1.markdown(rate)


income = row1_col2.slider("Monthly income of customers",
                 min_value=data["monthly_income"].min(),
                 max_value=data["monthly_income"].max(),
                 value=(2500.00, 10000.00)
                 )

row1_col1.markdown(income)
#######################################################################
mask = ~data.columns.isin(["loan_default", "borrower_rate", "employment_status"])
st.markdown(mask)
names = data.loc[:, mask].columns

#######################################################################
variable = row1_col3.selectbox("select variable to compare", names)

row1_col3.write(variable)

#######################################################################
filtered_data = data.loc[(data["borrower_rate"] >= rate[0]) &
                         (data["borrower_rate"] <= rate[1])&
                         (data["monthly_income"] >= income[0]) &
                         (data["monthly_income"] <= income[1])
                         ,:]

#######################################################################               
if st.checkbox("show filtered data", False):
    st.subheader("Raw Data")
    st.write(filtered_data)
    
row2_col1, row2_col2 = st.columns([1,1])

#######################################################################
barplotdata = filtered_data[["loan_default", variable]].groupby("loan_default").mean()
fig1, ax = plt.subplots(figsize=(8,3.7))
ax.bar(barplotdata.index.astype(str), barplotdata[variable], color = "#fc8d62")
ax.set_ylabel(variable)

row2_col1.subheader("Compare Customer Groups")
row2_col1.pyplot(fig1, use_container_width=True)


#### seaborn plot #########################

fig2 = sns.lmplot(y="borrower_rate", x = variable, data = filtered_data, order=2,
                  height=4, aspect=1/1, col="loan_default", hue="loan_default", palette = "Set2")


row2_col2.subheader("Borrower Rate Correlations")
row2_col2.pyplot(fig2, use_container_width=True)

#######################################################################

st.header("Predicting customer Default")
uploaded_data = st.file_uploader("Choose a file with Customer Data for predicting default")
