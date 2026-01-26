import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
# --- NEW IMPORTS for Machine Learning ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# ----------------------------------------

# --- Configuration and Theming ---
st.set_page_config(
    page_title="News Data Explorer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data with Caching for Performance ---
@st.cache_data
def load_data(file_path):
    """Loads the CSV data and performs initial data cleaning."""
    df = pd.read_csv(file_path)
    # Simple cleaning: converting date to datetime objects
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except:
        # Keep as string if conversion fails
        df['date'] = df['date'].astype(str) 
    return df

fake_df = load_data('Fake.csv')
true_df = load_data('True.csv')


# --- NEW: Model Training and Loading (Runs once on startup) ---
@st.cache_resource
def train_and_load_model(fake_df, true_df):
    """Combines data, trains a Logistic Regression classifier, and returns the pipeline."""
    
    # 1. Add Labels and Combine Data
    fake_df['label'] = 0  # 0 for Fake
    true_df['label'] = 1  # 1 for True
    df = pd.concat([fake_df, true_df])

    # 2. Prepare Feature Column
    # Combine title and text into a single feature for classification
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df = df.dropna(subset=['content'])
    
    # 3. Define Features (X) and Target (y)
    X = df['content']
    y = df['label']

    # 4. Create a Model Pipeline (TF-IDF Vectorizer + Classifier)
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('classifier', LogisticRegression(solver='liblinear'))
    ])

    # 5. Train the Model
   
    model_pipeline.fit(X, y)
   
    return model_pipeline

# Train the model when the app loads
classifier_model = train_and_load_model(fake_df, true_df)
# -------------------------------------------------------------


# --- Sidebar: Article Input Form (Unchanged) ---

st.sidebar.title("Analyze New Article 🤖")

# Use a form to capture the title and text input
with st.sidebar.form("article_input_form"):
    st.subheader("Paste your article details:")
    # Text input for the title
    input_title = st.text_input("Article Title (Optional)", value="")
    # Text area for the main body of the article
    input_text = st.text_area("Article Text", height=250, placeholder="Paste the full article text here...")
    
    # Submit button for the form
    submit_button = st.form_submit_button("Analyze Article")

# Check if the button was clicked and text was provided
if submit_button and input_text:
    # Store the user input in Streamlit's session state to persist it across reruns
    st.session_state['new_article'] = {
        'title': input_title if input_title else "N/A",
        'text': input_text
    }
    # Automatically switch the main view to the new analysis page
    selected_view = "New Article Analysis"
elif 'new_article' not in st.session_state:
    # Initialize the session state if it doesn't exist
    st.session_state['new_article'] = None
    selected_view = "Summary Dashboard"
else:
    # If a previous article exists, check if the user is viewing the analysis page
    if 'current_view' in st.session_state and st.session_state['current_view'] == "New Article Analysis":
         selected_view = "New Article Analysis"
    else:
        # Fall back to the default dashboard view
        selected_view = "Summary Dashboard"


# --- Sidebar: Navigation and Settings (Unchanged) ---
st.sidebar.markdown("---")
st.sidebar.title("App Navigation")

# Create a master view selector
master_view = st.sidebar.selectbox(
    "Select Analysis View",
    ["Summary Dashboard", "Fake News Explorer", "True News Explorer"]
)

# Only override selected_view if the user explicitly changes the selectbox
if submit_button and input_text:
    # Keep the auto-switch logic
    pass 
else:
    # Use the selectbox value for navigation
    selected_view = master_view
    st.session_state['current_view'] = master_view # Update session state for persistence

st.sidebar.expander("About This App ℹ️", expanded=False).markdown("""
    This interactive dashboard visualizes key statistics and trends from the 
    **Fake** and **True** news datasets. 
    \n\n- **Source:** Uploaded CSVs
    - **Created with:** Streamlit, Pandas, Plotly
""")
st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Use the tabs for detailed comparison!")


# --- Main Content: Title and Layout (Unchanged) ---

st.title("📰 News Dataset Visualizer: The Truth and The Fake")
st.markdown("""
    **A data exploration app to compare the key features and content 
    between two news datasets.**
    ---
""")


# --- Main Dashboard Logic (UPDATED Prediction Section) ---

if selected_view == "New Article Analysis":
    st.header("New Article Initial Analysis and Prediction ✨")
    
    if st.session_state['new_article'] and classifier_model:
        article = st.session_state['new_article']
        
        # Combine title and text for model input
        full_content = article['title'] + ' ' + article['text']
        
        # --- Prediction Logic ---
        # 1. Predict the label (0 or 1)
        prediction = classifier_model.predict([full_content])[0]
        
        # 2. Get the prediction probability
        prediction_proba = classifier_model.predict_proba([full_content])
        fake_proba = prediction_proba[0][0] * 100 # Probability of being Fake (0)
        true_proba = prediction_proba[0][1] * 100 # Probability of being True (1)
        
        # --- Display Results ---
        
        # Determine the visual based on prediction
        if prediction == 1:
            # TRUE NEWS
            st.success(f"**PREDICTION: REAL NEWS ✅**")
            st.info(f"The model is **{true_proba:.2f}%** confident this is a **TRUE** article.")
            prediction_delta = f"Prob. {true_proba:.2f}%"
            prediction_color = "normal"
        else:
            # FAKE NEWS
            st.error(f"**PREDICTION: FAKE NEWS ❌**")
            st.info(f"The model is **{fake_proba:.2f}%** confident this is a **FAKE** article.")
            prediction_delta = f"Prob. {fake_proba:.2f}%"
            prediction_color = "inverse"

        st.markdown("---")
        
        # Calculate key metrics for the new article
        text_length = len(article['text'])
        word_count = len(article['text'].split())
        
        st.subheader(f"Title: {article['title']}")
        
        # Use columns for metrics, including the new Prediction Metric
        col_pred, col_len, col_word = st.columns(3)
        col_pred.metric("Result", f"{'REAL' if prediction == 1 else 'FAKE'}", delta=prediction_delta, delta_color=prediction_color)
        col_len.metric("Text Length (Chars)", f"{text_length:,}")
        col_word.metric("Word Count", f"{word_count:,}")
        
        st.markdown("---")
        
        # Display the full text in an expander for review
        with st.expander("View Full Article Text 📝", expanded=False):
            st.code(article['text'])
            
    else:
        if not classifier_model:
            st.info("Model not loaded. Please check the terminal for errors during startup.")
        else:
            st.info("Paste an article in the sidebar and click 'Analyze Article' to see results here.")


# --- The rest of your code remains the same (Summary Dashboard, etc.) ---

elif selected_view == "Summary Dashboard":
    st.header("Global Metrics Overview 📊")
    
    # Use columns to showcase key metrics in a visually appealing row
    col1, col2, col3, col4 = st.columns(4)

    # Calculate and display metrics
    total_fake = len(fake_df)
    total_true = len(true_df)
    total_articles = total_fake + total_true
    fake_percent = (total_fake / total_articles) * 100
    
    col1.metric("Total Articles", f"{total_articles:,}")
    col2.metric("Fake Articles Count", f"{total_fake:,}", delta=f"{fake_percent:.1f}% of total")
    col3.metric("True Articles Count", f"{total_true:,}", delta=f"{-fake_percent:.1f}% of total", delta_color="inverse")
    
    # Calculate and display the number of unique subjects
    unique_subjects = pd.concat([fake_df['subject'], true_df['subject']]).nunique()
    col4.metric("Unique Subjects", unique_subjects)
    
    st.markdown("---")
    
    st.header("Detailed Explorers")
    
    # Use tabs for a cleaner, organized layout for deeper dives
    tab1, tab2, tab3 = st.tabs(["Fake News: Subject Distribution", "True News: Subject Distribution", "Text Length Analysis"])

    with tab1:
        st.subheader("Distribution by Subject in Fake News 🚨")
        # Create a visually rich bar chart
        fake_subject_counts = fake_df['subject'].value_counts().reset_index()
        fake_subject_counts.columns = ['Subject', 'Count']
        
        fig_fake = px.bar(
            fake_subject_counts, 
            x='Subject', 
            y='Count', 
            color='Subject', # Color by subject for visual distinction
            title='Fake News Articles by Subject',
            template='plotly_white'
        )
        st.plotly_chart(fig_fake, use_container_width=True)

    with tab2:
        st.subheader("Distribution by Subject in True News ✅")
        # Create a visually rich bar chart
        true_subject_counts = true_df['subject'].value_counts().reset_index()
        true_subject_counts.columns = ['Subject', 'Count']
        
        fig_true = px.bar(
            true_subject_counts, 
            x='Subject', 
            y='Count', 
            color='Subject', 
            title='True News Articles by Subject',
            template='plotly_white'
        )
        st.plotly_chart(fig_true, use_container_width=True)

    with tab3:
        st.subheader("Text Length Comparison")
        # Feature Engineering: Calculate article length
        fake_df['text_length'] = fake_df['text'].str.len()
        true_df['text_length'] = true_df['text'].str.len()
        
        # Combine for comparison
        fake_df['Type'] = 'Fake'
        true_df['Type'] = 'True'
        combined_df = pd.concat([fake_df[['text_length', 'Type']], true_df[['text_length', 'Type']]])
        
        # A box plot is great for comparing distributions
        fig_length = px.box(
            combined_df,
            x='Type',
            y='text_length',
            color='Type',
            title='Article Text Length Distribution',
            points='outliers' # Show outliers explicitly
        )
        st.plotly_chart(fig_length, use_container_width=True)

# --- Simple Display for Dedicated Explorers (Unchanged) ---
elif selected_view == "Fake News Explorer":
    st.header("Dive into Fake News Data 📉")
    st.write("First 100 rows of the Fake News dataset:")
    st.dataframe(fake_df.head(100), use_container_width=True)
    
elif selected_view == "True News Explorer":
    st.header("Dive into True News Data 📈")
    st.write("First 100 rows of the True News dataset:")
    st.dataframe(true_df.head(100), use_container_width=True)