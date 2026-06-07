import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn


st.title("LifeMetrics Dashboard 🏥")
st.write(
    "Gareth Liu | https://github.com/gl3084/gl-app-summer-26 | June 2026"
)

st.set_page_config(
    page_title = "LifeMetrics 🏥",
    layout = "centered",
    page_icon = "🏥",
)

## Step 01 - Setup
st.sidebar.title("LifeMetrics - Life Expectancy Agency 🏥")
page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊", "Automated Report 📑","Prediction"])


st.image("life_metrics.png")

df = pd.read_csv("Life Expectancy Data.csv")
df_cleaned = df.dropna()

## Step 02 - Load dataset
if page == "Introduction 📘":

    st.subheader("01 Introduction 📘")

    st.markdown("##### 🎯 Objectives")
    st.write("LifeMetrics aims to understand the factors that influence life expectancy around the world. By analyzing health, economic, and social indicators, we are attempting to identify the key drivers of population longevity and facilitate data-supported decisions that improve lives.")

    st.markdown("##### ℹ️ Data Preview")
    st.write("Our analysis is based on a global life expectancy dataset under the World Health Organization (WHO). It contains demographic, economic, healthcare, and educational information from various countries (around 193) across 2000 to 2015. A preview of the dataset is shown below:")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))

    st.markdown("##### ❗ Data Cleaning")
    st.write("To ensure complete records, rows containing missing values were removed in the data visualization and predication pages. This reduced the dataset from 2938 to 1649 entries. One limitation of this approach is that countries with incomplete reporting were disproportionately excluded, which may introduce bias into the analysis. A comparison between the original and cleaned dataset can be viewed below:")

    tab1, tab2 = st.tabs(["Original Data Set", "Cleaned Data Set"])

    with tab1:
        st.markdown("##### Missing values")
        missing = df.isnull().sum()
        st.write(missing)

        if missing.sum() == 0:
            st.success("✅ No missing values found")
        else:
            st.warning("⚠️ You have missing values")
        
        st.markdown("##### 📈 Summary Statistics")
        if st.button("Show Original Dataset (Rows, Columns)"):
            st.write(df.shape)
        if st.button("Show Original Dataset Stats"):
            st.dataframe(df.describe())

    with tab2:
        st.markdown("##### Missing values")
        missing = df_cleaned.isnull().sum()
        st.write(missing)

        if missing.sum() == 0:
            st.success("✅ No missing values found")
        else:
            st.warning("⚠️ You have missing values")

        st.markdown("##### 📈 Summary Statistics")
        if st.button("Show Cleaned Dataset (Rows, Columns)"):
            st.write(df_cleaned.shape)
        if st.button("Show Cleaned Dataset Stats"):
            st.dataframe(df_cleaned.describe())
