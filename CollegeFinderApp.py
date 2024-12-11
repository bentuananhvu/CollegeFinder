import streamlit as st
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("miced_matching_data.csv")

college_data = load_data()
college_data.columns = ['InstitutionName', 'StateCode', 'Control', 'Size', 'Selectivity',
       'StudentFacultyRatioBin', 'TestOptional', 'SATVRMID', 'SATMTMID',
       'ACTENMID', 'ACTMTMID', 'Tuition', 'LivingCost', 'CrimeRate',
       'Diversity', 'Population']

# Define preprocessing
num_cols = ["Size", "Selectivity", "StudentFacultyRatioBin", "SATVRMID", "SATMTMID", "ACTENMID", "ACTMTMID", "Tuition", "LivingCost", "CrimeRate", "Diversity", "Population"]
cat_cols = ["StateCode", "Control"]

num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Standardize numerical values
])

cat_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical values
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)

# Preprocess data
college_preprocessed = preprocessor.fit_transform(college_data)

# Convert sparse matrix to dense if necessary
if isinstance(college_preprocessed, scipy.sparse.spmatrix):
    college_preprocessed = college_preprocessed.toarray()

# Function to build and train the autoencoder
@st.cache_resource
def build_autoencoder(data):
    input_dim = data.shape[1]

    autoencoder = Sequential([
        Input(shape=(input_dim,)),  # Specify the input shape
        Dense(10, activation='relu', name="encoding_layer"),  # Encoding layer
        Dense(input_dim, activation='sigmoid')  # Decoding layer
    ])

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(data, data, epochs=50, batch_size=32, verbose=0)

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(name="encoding_layer").output)
    encoded_data = encoder.predict(data)
    return autoencoder, encoder, encoded_data

# Function to apply PCA
@st.cache_resource
def apply_pca(data):
    pca = PCA()
    pca.fit(data)
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    n_components = (cumulative_variance >= 0.95).argmax() + 1

    pca_optimal = PCA(n_components=n_components)
    reduced_data = pca_optimal.fit_transform(data)
    return pca_optimal, reduced_data

# Function to set up KNN model
@st.cache_resource
def build_knn(data):
    knn = NearestNeighbors(n_neighbors=len(data), metric='euclidean')
    knn.fit(data)
    return knn

# Train Autoencoder and PCA
autoencoder, encoder, college_encodings = build_autoencoder(college_preprocessed)
pca_optimal, college_pca = apply_pca(college_preprocessed)
knn = build_knn(college_preprocessed)

# Placeholder for student preprocessing and PCA transformation
def calculate_pca_similarity(student_preprocessed):
    student_pca = pca_optimal.transform(student_preprocessed)
    similarities_pca = cosine_similarity(student_pca, college_pca).flatten()
    sorted_indices_pca = similarities_pca.argsort()[::-1]  # Sort in descending order
    sorted_similarities_pca = similarities_pca[sorted_indices_pca]
    sorted_college_indices_pca = college_data.index[sorted_indices_pca]
    return pd.DataFrame({
        "InstitutionName": college_data.iloc[sorted_college_indices_pca]["InstitutionName"].values,
        "PCACosine": sorted_similarities_pca,
        "PCARank": range(1, len(sorted_college_indices_pca) + 1)
    })

# Define function for calculating KNN distances and ranks
def calculate_knn_similarity(student_preprocessed):
    distances_knn, indices_knn = knn.kneighbors(student_preprocessed)
    all_distances = distances_knn.flatten()  # All distances to colleges
    all_indices = indices_knn.flatten()  # All college indices

    sorted_indices_knn = all_distances.argsort()
    sorted_college_indices_knn = all_indices[sorted_indices_knn]

    return pd.DataFrame({
        "InstitutionName": college_data.iloc[sorted_college_indices_knn]["InstitutionName"].values,
        "KnnDistance": all_distances[sorted_indices_knn],
        "KnnRank": range(1, len(sorted_college_indices_knn) + 1)
    })

# Define function for calculating cosine similarity and sorting scores
def calculate_cosine_similarity(student_encoding):
    similarities_autoenc = cosine_similarity(student_encoding, college_encodings).flatten()
    sorted_indices_autoenc = similarities_autoenc.argsort()[::-1]  # Sort in descending order
    sorted_similarities_autoenc = similarities_autoenc[sorted_indices_autoenc]
    sorted_college_indices_autoenc = college_data.index[sorted_indices_autoenc]
    return pd.DataFrame({
        "InstitutionName": college_data.iloc[sorted_college_indices_autoenc]["InstitutionName"].values,
        "AutoencCosine": sorted_similarities_autoenc,
        "AutoencRank": range(1, len(sorted_college_indices_autoenc) + 1)
    })

# Placeholder function for compatibility score calculation
def calculate_compatibility(student_input):
    # Process student input
    student_preprocessed = preprocessor.transform(student_input)
    if isinstance(student_preprocessed, scipy.sparse.spmatrix):
        student_preprocessed = student_preprocessed.toarray()
    student_encoding = encoder.predict(student_preprocessed)
    autoenc_results = calculate_cosine_similarity(student_encoding)
    pca_results = calculate_pca_similarity(student_preprocessed)
    knn_results = calculate_knn_similarity(student_preprocessed)

    # Merge the three DataFrames on InstitutionName
    merged_scores = pd.merge(autoenc_results, pca_results, on="InstitutionName", suffixes=("_Autoenc", "_PCA"))
    merged_scores = pd.merge(merged_scores, knn_results, on="InstitutionName")

    # Calculate the average rank
    merged_scores["AvgRank"] = merged_scores[["AutoencRank", "PCARank", "KnnRank"]].mean(axis=1)

    # Sort by the average rank in ascending order (best rank first)
    merged_scores = merged_scores.sort_values(by="AvgRank").reset_index(drop=True)

    # Normalize KNN Distance Score
    min_knn_dist = merged_scores["KnnDistance"].min()
    max_knn_dist = merged_scores["KnnDistance"].max()
    merged_scores["KNN_Normalized"] = 1 - (merged_scores["KnnDistance"] - min_knn_dist) / (max_knn_dist - min_knn_dist)

    # Calculate Aggregated Score
    merged_scores["AggregatedScore"] = round((merged_scores["PCACosine"] + merged_scores["AutoencCosine"] + merged_scores["KNN_Normalized"]) / 3 * 100, 3)

    # Sort by Aggregated Score
    merged_scores = merged_scores.sort_values(by="AggregatedScore", ascending=False)

    return merged_scores

# Streamlit app
st.title("College Finder Application Demo")
st.write("Provide your details below to see the top 10 recommended universities based on your profile.")

# Inputs from the user
state_code = st.selectbox("State Code:", options=["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"], index=21)
control = st.selectbox("Control (1 = Public, 2 = Private):", options=["1", "2"], index=1)
size = st.slider("University Size (1 = <1,000, 2 = 1,000-4,999, 3 = 5,000-9,999, 4 = 10,000+):", 1, 4, 3)
selectivity = st.slider("Selectivity (1 = Least Selective, 5 = Most Selective):", 1, 5, 4)
student_faculty_ratio_bin = st.slider("Student-Faculty Ratio Bin (1 = Low, 5 = High):", 1, 5, 2)
test_optional = st.radio("Test Optional (1 = Yes, 0 = No):", options=[1, 0], index=0)
sat_vr_mid = st.slider("SAT English (200-800):", 200, 800, 650)
sat_mt_mid = st.slider("SAT Math (200-800):", 200, 800, 677)
act_en_mid = st.slider("ACT English (1-36):", 1, 36, 29)
act_mt_mid = st.slider("ACT Math (1-36):", 1, 36, 31)
tuition = st.slider("Tuition Bin (1 = Least Expensive, 5 = Most Expensive):", 1, 5, 5)
living_cost = st.slider("Living Cost Bin (1 = Lowest, 5 = Highest):", 1, 5, 3)
crime_rate = st.slider("Crime Rate Bin (1 = Lowest, 5 = Highest):", 1, 5, 4)
diversity = st.slider("Diversity Index Bin (1 = Lowest, 5 = Highest):", 1, 5, 4)
population = st.slider("Population Bin (1 = Lowest, 5 = Highest):", 1, 5, 2)

# Aggregating student inputs
student_input = pd.DataFrame({
    "StateCode": [state_code],
    "Control": [control],
    "Size": [size],
    "Selectivity": [selectivity],
    "StudentFacultyRatioBin": [student_faculty_ratio_bin],
    "TestOptional": [test_optional],
    "SATVRMID": [sat_vr_mid],
    "SATMTMID": [sat_mt_mid],
    "ACTENMID": [act_en_mid],
    "ACTMTMID": [act_mt_mid],
    "Tuition": [tuition],
    "LivingCost": [living_cost],
    "CrimeRate": [crime_rate],
    "Diversity": [diversity],
    "Population": [population]
})

# Calculate compatibility scores
if st.button("Show Recommendations"):
    top_universities = calculate_compatibility(student_input)
    st.write("### Top 10 Recommended Universities")
    st.dataframe(top_universities)
