import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Pingüinos",
    page_icon="🐧",
    layout="wide"
)

# Título principal
st.title("🐧 Análisis de Pingüinos: Clasificación y Clustering")

# Carga de datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('penguins.csv')
        df = df.dropna(subset=['species'])
        df['sex'] = df['sex'].fillna('unknown')
        return df
    except FileNotFoundError:
        st.error("Archivo 'penguins.csv' no encontrado.")
        return None

df = load_data()

if df is not None:
    # Preprocesamiento
    numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    categorical_features = ['island', 'sex']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    # Dividir en pestañas
    tab1, tab2, tab3 = st.tabs(["📊 Datos", "🤖 Clasificación", "🔍 Clustering (K-Means)"])

    with tab1:
        st.header("Exploración de Datos")
        st.dataframe(df.head())
        
        # Visualizaciones
        st.subheader("Distribución de Características")
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        sns.boxplot(data=df, x='species', y='bill_length_mm', ax=ax[0, 0])
        sns.boxplot(data=df, x='species', y='flipper_length_mm', ax=ax[0, 1])
        sns.scatterplot(data=df, x='bill_length_mm', y='bill_depth_mm', hue='species', ax=ax[1, 0])
        sns.scatterplot(data=df, x='flipper_length_mm', y='body_mass_g', hue='species', ax=ax[1, 1])
        st.pyplot(fig)

    with tab2:
        st.header("Clasificación con Random Forest")
        
        # Entrenamiento del modelo
        X = df.drop('species', axis=1)
        y = df['species']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.metric("Precisión", f"{accuracy_score(y_test, y_pred):.2%}")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

    with tab3:
        st.header("Clustering con K-Means")
        
        # Selección de características
        features = st.multiselect(
            "Selecciona características para clustering",
            numeric_features,
            default=numeric_features
        )
        
        # Preprocesamiento para clustering
        X_cluster = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Selección de K
        k = st.slider("Número de clusters (K)", 2, 5, 3)
        
        # Entrenar K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Métricas
        st.metric("Inercia", f"{kmeans.inertia_:.2f}")
        st.metric("Silhouette Score", f"{silhouette_score(X_scaled, clusters):.2f}")
        
        # Visualización con PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        ax.set_title(f"Proyección PCA de Clusters (K={k})")
        ax.set_xlabel("Componente Principal 1")
        ax.set_ylabel("Componente Principal 2")
        plt.colorbar(scatter, label='Cluster')
        st.pyplot(fig)
        
        # Comparación con especies reales
        if st.checkbox("Mostrar comparación con especies reales"):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df.loc[X_cluster.index, 'species'], palette='Set2', ax=ax)
            ax.set_title("PCA Coloreado por Especies Reales")
            st.pyplot(fig)

else:
    st.warning("No se pudieron cargar los datos. Sube el archivo 'penguins.csv'.")

# Footer
st.markdown("---")
st.markdown("Datos de [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/)")
