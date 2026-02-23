import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Tourism Experience Analytics", layout="wide", page_icon="🌍")

# --- 2. LOAD & PREPARE DATA ---
@st.cache_data
def load_data():
    # encoding='latin1' prevents crashes with special characters in city/country names
    df = pd.read_csv('tourism_master_data.csv', encoding='latin1')
    # Drop rows with missing values in critical columns to avoid training errors
    df = df.dropna(subset=['VisitMode', 'Continent', 'Country', 'AttractionType'])
    return df

df = load_data()

# --- 3. HELPER FUNCTION: AUTO-TRAIN CLASSIFIER ---
@st.cache_resource
def get_trained_model():
    features_clf = ['Rating', 'VisitYear', 'VisitMonth', 'Continent', 'Country', 'AttractionType']
    X = df[features_clf].copy()
    
    # Label Encoding for categorical features
    le_dict = {}
    for col in ['Continent', 'Country', 'AttractionType']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    le_target = LabelEncoder()
    y = le_target.fit_transform(df['VisitMode'].astype(str))
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    return model, le_dict, le_target

model, encoders, target_encoder = get_trained_model()

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/826/826070.png", width=100)
st.sidebar.title("Tourism Dashboard")
page = st.sidebar.selectbox("Select a Task", ["Market Insights", "Predict Visit Mode", "Attraction Recommender"])

# --- 5. PAGE: MARKET INSIGHTS ---
if page == "Market Insights":
    st.title("📊 Tourism Market Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Popularity by Attraction Type")
        type_counts = df['AttractionType'].value_counts().head(10)
        st.bar_chart(type_counts)
        
    with col2:
        st.subheader("Global Tourist Origins")
        country_counts = df['Country'].value_counts().head(10)
        st.write(country_counts)

# --- 6. PAGE: CLASSIFICATION (Visit Mode) ---
elif page == "Predict Visit Mode":
    st.title("🧳 Trip Persona Classifier")
    st.write("Fill in the travel details to predict the likely Visit Mode.")

    with st.form("clf_form"):
        c1, c2 = st.columns(2)
        with c1:
            cont = st.selectbox("Continent", sorted(df['Continent'].unique()))
            reg_rating = st.slider("Anticipated Rating", 1, 5, 5)
        with c2:
            attr = st.selectbox("Attraction Category", sorted(df['AttractionType'].unique()))
            month = st.selectbox("Month of Visit", list(range(1, 13)))
        
        # Filter countries based on continent for a better UX
        filtered_countries = sorted(df[df['Continent'] == cont]['Country'].unique())
        country = st.selectbox("Country", filtered_countries)

        if st.form_submit_button("Analyze Persona"):
            try:
                # Prepare input data for the model
                input_row = pd.DataFrame([[
                    reg_rating, 2024, month,
                    encoders['Continent'].transform([cont])[0],
                    encoders['Country'].transform([country])[0],
                    encoders['AttractionType'].transform([attr])[0]
                ]], columns=['Rating', 'VisitYear', 'VisitMonth', 'Continent', 'Country', 'AttractionType'])
                
                prediction = model.predict(input_row)
                result = target_encoder.inverse_transform(prediction)[0]
                
                st.success(f"Predicted Visit Mode: **{result}**")
            except Exception as e:
                st.error(f"Error in prediction: {e}")

# --- 7. PAGE: RECOMMENDATIONS ---
elif page == "Attraction Recommender":
    st.title("⭐️ Personalized Recommendations")
    
    # Clean unique attraction names
    attraction_list = sorted(df['Attraction'].unique())
    selected_name = st.selectbox("Select an attraction you've visited:", attraction_list)
    
    if st.button("Recommend Similar Places"):
        # Quick Content-Based logic
        items = df[['Attraction', 'AttractionType', 'Region']].drop_duplicates('Attraction').reset_index(drop=True)
        # Create a combined string for similarity comparison
        items['soup'] = items['AttractionType'].astype(str) + " " + items['Region'].astype(str)
        
        cv = CountVectorizer(stop_words='english')
        matrix = cv.fit_transform(items['soup'])
        sim = cosine_similarity(matrix)
        
        # Get index of the selected attraction
        idx = items[items['Attraction'] == selected_name].index[0]
        
        # Get similarity scores
        scores = list(enumerate(sim[idx]))
        # Sort and take top 5 (skipping the 1st one because it's the item itself)
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
        
        st.write("### We think you'll also like:")
        for i, score in sorted_scores:
            st.info(f"📍 **{items.iloc[i]['Attraction']}** — {items.iloc[i]['AttractionType']} in {items.iloc[i]['Region']}")