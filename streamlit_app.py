import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("final_data.csv", parse_dates=['Date_time'])
    df['hour'] = df['Date_time'].dt.hour
    df['day'] = df['Date_time'].dt.date
    df['month'] = df['Date_time'].dt.month
    return df

data = load_data()

# Título de la aplicación
st.title("Análisis de Datos Atmosféricos con Streamlit")

variables = ['CO', 'DIR', 'HUM', 'LLU', 'NO2', 'O3', 'PM10', 'PM2', 'PRE', 'RS', 'SO2', 'TMP', 'VEL']
variable = st.selectbox("Selecciona una variable para visualizar", variables)

# Variación diaria por estación
def plot_variable(variable):
    plot_data = data[["hour", "station", variable]].dropna()
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=plot_data, x="hour", y=variable, hue="station", marker="o")
    plt.title(f"Evolución de {variable} por hora")
    plt.xlabel("Hora del día")
    plt.ylabel(variable)
    st.pyplot(plt)

plot_variable(variable)

# Tendencias temporales
st.header("Tendencias Temporales")

def plot_trends(variable):
    trend_data = data.groupby("day")[variable].mean().reset_index()
    plt.figure(figsize=(14, 8))
    sns.lineplot(x=trend_data["day"], y=trend_data[variable])
    plt.title(f"Tendencia Temporal de {variable}")
    plt.xlabel("Fecha")
    plt.ylabel(variable)
    plt.xticks(rotation=45)
    st.pyplot(plt)

plot_trends(variable)

# Patrones estacionales
st.header("Patrones Estacionales")

def plot_seasonal_patterns(variable):
    seasonal_data = data.groupby("month")[variable].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=seasonal_data["month"], y=seasonal_data[variable], palette="coolwarm")
    plt.title(f"Patrón Estacional de {variable}")
    plt.xlabel("Mes")
    plt.ylabel(variable)
    st.pyplot(plt)

plot_seasonal_patterns(variable)

# Análisis de Componentes Principales (PCA)
st.header("Análisis de Componentes Principales (PCA)")
selected_variables = st.multiselect("Selecciona variables para PCA", variables, default=variables[:5])
n_components = st.slider("Número de Componentes", 1, min(len(selected_variables), 10), 2)

if selected_variables:
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[selected_variables].dropna())
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)

    df_pca = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(n_components)])
    st.write("Varianza explicada:", pca.explained_variance_ratio_)

    if n_components >= 2:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df_pca.iloc[:, 0], y=df_pca.iloc[:, 1], alpha=0.5)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Proyección PCA")
        st.pyplot(plt)
