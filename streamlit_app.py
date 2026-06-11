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
    tabhead, tabtail, tabrand = st.tabs(["First rows", "Last rows", "Random sample"])
    with tabhead:
        st.dataframe(df.head(rows))
    with tabtail:
        st.dataframe(df.tail(rows))   
    with tabrand:
        st.dataframe(df.sample(rows))


    variable_dict = pd.DataFrame({
        "Variable": [
            "Country", "Year", "Status", "Life expectancy", "Adult Mortality", "Infant deaths", "Alcohol", "Percentage expenditure", "Hepatitis B", "Measles", "BMI", "Under-five deaths", "Polio", "Total expenditure", "Diphtheria", "HIV/AIDS", "GDP", "Population", "Thinness 1-19 years", "Thinness 5-9 years", "Income com of resources", "Schooling"
        ],
        "Description": [
            "Country where data was collected",
            "Year of observation",
            "Developed or Developing status",
            "Average number of years a person is expected to live",
            "Probability of dying between ages 15 and 60 per 1000 population",
            "Number of Infant Deaths per 1000 population",
            "Liters of pure alcohol consumed per capita",
            "Health expenditure as percentage of Gross Domestic Product per capita",
            "Hep B immunization coverage among 1-year-olds percentage",
            "Number of reported cases per 1000 population",
            "Average Body Mass Index of entire population",
            "Deaths under age five per 1000 live births",
            "Polio immunization coverage among 1-year-olds percentage",
            "General government expenditure on health as a percentage of total government expenditure",
            "DTP3 immunization coverage among 1-year-olds percentage",
            "Deaths from HIV/AIDS per 1000 live births",
            "Gross Domestic Product per capita in USD",
            "Population of the country",
            "Percentage of adolescents aged 10-19 classified as thin",
            "Percentage of children aged 5-9 classified as thin",
            "Index measuring income/resource composition from 0 to 1",
            "Average number of years of schooling"
        ],
        "Type": [str(df[col].dtype) for col in df.columns]
    })
    st.markdown("##### 📙 Variable Dictionary")
    st.dataframe(variable_dict, use_container_width = True)


    st.markdown("##### ❗ Data Cleaning")
    st.write("To ensure complete records, rows containing missing values were removed in the data visualization and predication pages. This reduced the dataset from 2938 to 1649 entries. One limitation of this approach is that countries with incomplete reporting were disproportionately excluded, which may introduce bias into the analysis. A comparison between the original and cleaned dataset can be viewed below:")

    tab1, tab2 = st.tabs(["Original Data Set", "Cleaned Data Set"])

    with tab1:
        st.markdown("##### General Information")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df_original.shape[0])
        with col2:
            st.metric("Features", df_original.shape[1])
        with col3:
            st.metric("Target", "Life exp")
        with col4:
            st.metric("Source", "WHO")

        st.markdown("##### Missing values")
        missing = df_original.isnull().sum()
        st.write(missing)

        if missing.sum() == 0:
            st.success("✅ No missing values found")
        else:
            st.warning("⚠️ You have missing values")
        
        st.markdown("##### 📈 Summary Statistics")
        if st.button("Show Original Dataset Stats"):
            st.dataframe(df_original.describe())

    with tab2:
        st.markdown("##### General Information")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            st.metric("Target", "Life exp")
        with col4:
            st.metric("Source", "WHO")

        st.markdown("##### Missing values")
        missing = df.isnull().sum()
        st.write(missing)

        if missing.sum() == 0:
            st.success("✅ No missing values found")
        else:
            st.warning("⚠️ You have missing values")

        st.markdown("##### 📈 Summary Statistics")
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

    st.divider()
    st.subheader("Distribution of Life Expectancy")
    figh, axh = plt.subplots()
    sns.histplot(data=df, x="Life expectancy ", bins=20, kde=True)
    st.pyplot(figh)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Mean", round(df["Life expectancy "].mean(), 2))
    with col2:
        st.metric("Median", round(df["Life expectancy "].median(), 2))
    with col3:
        st.metric("Std", round(df["Life expectancy "].std(), 2))
    with col4:
        st.metric("Min", df["Life expectancy "].min())
    with col5:
        st.metric("Max", df["Life expectancy "].max())

    st.divider()

    st.markdown("##### Key Insights")
    st.write("Schooling and income composition of resources showed the strongest positive corrleations with life expectancy (r ≈ 0.70)")
    st.write("Developed countries generally exhibited higher life expectancy than developing countries.")
    st.write("Adult mortality had the strongest negative correlation with life expectancy as expected")
    

elif page == "Prediction 🔮":
    st.subheader("03 Prediction with Linear Regression")

    ### Label Encoder to change text categories into number categories
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    df["Country"] = le.fit_transform(df["Country"])
    df["Status"] = le.fit_transform(df["Status"])

    list_var = list(df.columns)

    target_selection  = st.sidebar.selectbox("Select target variable (Y)", list_var)
    features_selection = st.sidebar.multiselect("Select features (X)", list(df.columns.drop(target_selection)), default=list(df.columns.drop(target_selection)))
    selected_metrics = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"], default=["Mean Absolute Error (MAE)"])
    test_size = st.sidebar.slider("Choose test size (%)",10,40,20)

    ### i) X and y
    X = df[features_selection]
    y = df[target_selection]

    st.dataframe(X.head())
    st.dataframe(y.head())

    ### ii) train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=(test_size/100))

    st.write(f"Training set: {X_train.shape[0]}")
    st.write(f"Testing set: {X_test.shape[0]}")
    st.divider()

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

    st.success(f"My model performance is of {np.round(mae,2)} years")

    fig, ax = plt.subplots()
    ax.scatter(y_test,predictions,alpha=0.5)
    ax.plot([y_test.min(),y_test.max()],
           [y_test.min(),y_test.max() ],"--r",linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    st.divider()
    st.markdown("##### Coefficient Values")
    coeff_df = pd.DataFrame(model.coef_, X.columns, columns = ["Coefficient"]).sort_values(by="Coefficient", key=abs, ascending=False)
    st.dataframe(coeff_df, use_container_width = True)
