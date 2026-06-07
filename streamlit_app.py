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
page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊", "Prediction 🔮"])


st.image("life_metrics.png")

df_original = pd.read_csv("Life Expectancy Data.csv")
df = df_original.dropna()

## Step 02 - Load dataset
if page == "Introduction 📘":

    st.subheader("01 Introduction 📘")

    st.markdown("##### 🎯 Objectives")
    st.write("LifeMetrics aims to understand the factors that influence life expectancy around the world. By analyzing health, economic, and social indicators, we are attempting to identify the key drivers of population longevity and facilitate data-supported decisions that improve lives.")

    st.markdown("##### ℹ️ Data Preview")
    st.write("Our analysis is based on a global life expectancy dataset under the World Health Organization (WHO). It contains demographic, economic, healthcare, and educational information from various countries (around 193) across 2000 to 2015. A preview of the dataset is shown below:")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df_original.head(rows))

    st.markdown("##### ❗ Data Cleaning")
    st.write("To ensure complete records, rows containing missing values were removed in the data visualization and predication pages. This reduced the dataset from 2938 to 1649 entries. One limitation of this approach is that countries with incomplete reporting were disproportionately excluded, which may introduce bias into the analysis. A comparison between the original and cleaned dataset can be viewed below:")

    tab1, tab2 = st.tabs(["Original Data Set", "Cleaned Data Set"])

    with tab1:
        st.markdown("##### Missing values")
        missing = df_original.isnull().sum()
        st.write(missing)

        if missing.sum() == 0:
            st.success("✅ No missing values found")
        else:
            st.warning("⚠️ You have missing values")
        
        st.markdown("##### 📈 Summary Statistics")
        if st.button("Show Original Dataset (Rows, Columns)"):
            st.write(df_original.shape)
        if st.button("Show Original Dataset Stats"):
            st.dataframe(df_original.describe())

    with tab2:
        st.markdown("##### Missing values")
        missing = df.isnull().sum()
        st.write(missing)

        if missing.sum() == 0:
            st.success("✅ No missing values found")
        else:
            st.warning("⚠️ You have missing values")

        st.markdown("##### 📈 Summary Statistics")
        if st.button("Show Cleaned Dataset (Rows, Columns)"):
            st.write(df.shape)
        if st.button("Show Cleaned Dataset Stats"):
            st.dataframe(df.describe())


elif page == "Visualization 📊":

    ## Step 03 - Data Viz
    st.subheader("02 Data Visualization")

    col_x = st.selectbox("Select X-axis variable",df.columns,index=0)
    col_y = st.selectbox("Select Y-axis variable",df.columns.drop(["Country", "Status"]),index=1)
    grouped_data = df.groupby(col_x)[col_y].mean().reset_index()

    tab1, tab2, tab3, tab4 = st.tabs(["Bar Chart 📊","Line Chart 📈","Scatterplot 🔵","Correlation Heatmap 🔥"])

    with tab1:
        st.subheader("Bar Chart")
        st.bar_chart(grouped_data.sort_values(by=col_x), x=col_x, y=col_y, use_container_width=True)

    with tab2:
        st.subheader("Line Chart")
        st.line_chart(grouped_data.sort_values(by=col_x), x=col_x, y=col_y, use_container_width=True)

    with tab3:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=col_x, y=col_y)
        st.pyplot(fig)
        
    with tab4:
        st.subheader("Correlation Matrix")
        df_numeric = df.select_dtypes(include=np.number)

        fig_corr, ax_corr = plt.subplots(figsize=(18,14))
        # create the plot, in this case with seaborn 
        sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='coolwarm')
        ## render the plot in streamlit 
        st.pyplot(fig_corr)


elif page == "Prediction 🔮":
    st.subheader("03 Prediction with Linear Regression")

    ### Label Encoder to change text categories into number categories
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    df["Country"] = le.fit_transform(df["Country"])
    df["Status"] = le.fit_transform(df["Status"])

    list_var = list(df.columns)

    features_selection = st.sidebar.multiselect("Select features (X)", list_var, default=list_var)
    target_selection  = st.sidebar.selectbox("Select target variable (Y))", list_var)
    selected_metrics = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"], default=["Mean Absolute Error (MAE)"])

    ### i) X and y
    X = df[features_selection]
    y = df[target_selection]

    st.dataframe(X.head())
    st.dataframe(y.head())

    ### ii) train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


    ## Model 

    ### i) Definition model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()

    ### ii) Training model
    model.fit(X_train,y_train)

    ### iii) Prediction
    predictions = model.predict(X_test)

    ### iv) Evaluation 
    from sklearn import metrics 
    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE** {mse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE** {mae:,.2f}")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R2** {r2:,.3f}")

    st.success(f"My model performance is of {np.round(mae,2)}")

    fig, ax = plt.subplots()
    ax.scatter(y_test,predictions,alpha=0.5)
    ax.plot([y_test.min(),y_test.max()],
           [y_test.min(),y_test.max() ],"--r",linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
