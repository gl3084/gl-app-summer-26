import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn


st.title("Wellness4Youth Dashboard 🧠❤️‍🩹")
st.write(
    "Gareth Liu | https://github.com/gl3084/gl-app-summer-26 | June 2026"
)

st.set_page_config(
    page_title = "Wellness4Youth 🧠❤️‍🩹",
    layout = "centered",
    page_icon = "🧘",
)

## Step 01 - Setup
st.sidebar.title("Teen Mental Health Awareness Co. 🧠❤️‍🩹")
page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊", "Automated Report 📑","Prediction"])


st.image("social_media_stress.png")

df = pd.read_csv("Teen_Mental_Health_Dataset.csv")

## Step 02 - Load dataset
if page == "Introduction 📘":

    st.subheader("01 Introduction 📘")

    st.markdown("##### 🎯 Objectives")
    st.write("Wellness4Youth aims to identify how teens' digital habits influence their overall well-being and academic success using data-driven analysis and predictive modeling. For this specific study, we analyze the relationship between social media use, mental health, physical activity, and academic peformance in adolescents.")

    st.markdown("##### ℹ️ Data Preview")
    st.write("Wellnes4Youth has collected data from 1200 teenagers aged 13-19. A preview of the data can be seen below:")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ You have missing values")

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

    df2 = df.drop(["gender", "platform_usage", "social_interaction_level"], axis = 1)
    
    st.subheader("Correlation Matrix")
    df_numeric = df.select_dtypes(include=np.number)

    fig_corr, ax_corr = plt.subplots(figsize=(18,14))
    # create the plot, in this case with seaborn 
    sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='coolwarm')
    ## render the plot in streamlit 
    st.pyplot(fig_corr)
