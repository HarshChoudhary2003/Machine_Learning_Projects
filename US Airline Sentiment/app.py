
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
import os
from preprocessing_library import clean_text
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Page Config ---
st.set_page_config(page_title="AIRLINE SENTIMENT AI", layout="wide", page_icon="✈️")

# --- Initialize Resources ---
@st.cache_resource
def load_model_artifacts():
    if os.path.exists('sentiment_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    return None, None

@st.cache_data
def load_data():
    return pd.read_csv("Tweets.csv")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def preprocess_input(text):
    cleaned = clean_text(text, lowercase_flag=True, remove_urls_flag=True, remove_mentions_flag=True)
    tokens = word_tokenize(cleaned)
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmatized)

# --- Custom Styles ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #e94560 !important;
        font-family: 'Outfit', sans-serif;
    }
    .stButton>button {
        background-color: #e94560;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 25px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/clouds/200/airport.png")
st.sidebar.title("✈️ Airline AI")
page = st.sidebar.radio("Navigate", ["Dashboard", "Sentiment Predictor", "About Data"])

df = load_data()
model, vectorizer = load_model_artifacts()

if page == "Dashboard":
    st.title("📊 Airline Sentiment Insights")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Tweets", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Airlines Tracked", df['airline'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        neg_count = (df['airline_sentiment'] == 'negative').sum()
        st.metric("Negative Sentiment %", f"{(neg_count/len(df)*100):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(df, names='airline_sentiment', hole=0.4, 
                         color_discrete_sequence=['#e94560', '#0f3460', '#4ecca3'])
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Sentiment by Airline")
        fig_bar = px.histogram(df, x='airline', color='airline_sentiment', barmode='group',
                               color_discrete_sequence=['#e94560', '#0f3460', '#4ecca3'])
        fig_bar.update_layout(template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Top Negative Reasons")
    neg_reasons = df[df['airline_sentiment'] == 'negative']['negativereason'].value_counts().reset_index()
    fig_reason = px.bar(neg_reasons, x='negativereason', y='count', color='count',
                        labels={'negativereason': 'Reason', 'count': 'Count'})
    fig_reason.update_layout(template="plotly_dark", showlegend=False)
    st.plotly_chart(fig_reason, use_container_width=True)

elif page == "Sentiment Predictor":
    st.title("🔮 AI Sentiment Predictor")
    st.write("Type a tweet to see what the AI thinks about it.")
    
    if model is None:
        st.error("Model artifacts not found! Please run the training script first.")
    else:
        user_input = st.text_area("Enter your tweet here:", placeholder="The flight was delayed but the food was okay.")
        
        if st.button("Predict Sentiment"):
            if user_input:
                processed = preprocess_input(user_input)
                tfidf_feat = vectorizer.transform([processed])
                prediction = model.predict(tfidf_feat)[0]
                probs = model.predict_proba(tfidf_feat)[0]
                
                # Result Display
                color = "#4ecca3" if prediction == "positive" else "#e94560" if prediction == "negative" else "#0f3460"
                st.markdown(f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: white !important;">PREDICTED: {prediction.upper()}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability Bars
                st.write("---")
                st.subheader("Confidence Scores")
                cols = st.columns(3)
                classes = model.classes_
                for i in range(len(classes)):
                    cols[i].metric(classes[i].capitalize(), f"{probs[i]*100:.1f}%")
            else:
                st.warning("Please enter some text.")

elif page == "About Data":
    st.title("📖 Dataset Overview")
    st.write("This dataset contains tweets about major US airlines from February 2015. It includes human-annotated sentiment scores.")
    st.dataframe(df.head(100), use_container_width=True)

