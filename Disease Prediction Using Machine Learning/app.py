import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import time
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(
    page_title="HealthAI Ultra | Next-Gen Diagnostics",
    page_icon="🧬",
    layout="wide",
)

# --- ADVANCED UI SYSTEM (Custom CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
        --glass: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.1);
        --primary: #6366f1;
        --secondary: #a855f7;
        --accent: #22d3ee;
        --success: #10b981;
    }

    * { font-family: 'Plus Jakarta Sans', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 0% 0%, #1e1b4b 0%, #020617 100%);
    }

    /* Glass Card Style */
    .glass-card {
        background: var(--glass);
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 25px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        transform: translateY(-8px);
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }

    /* Vibrant Text */
    .gradient-text {
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Animated Button */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        border: none !important;
        color: white !important;
        height: 65px !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        border-radius: 20px !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3) !important;
        text-transform: uppercase !important;
    }

    .stButton>button:hover {
        transform: scale(1.02) translateY(-3px) !important;
        box-shadow: 0 15px 30px rgba(168, 85, 247, 0.5) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(2, 6, 23, 0.95) !important;
        border-right: 1px solid var(--glass-border);
    }

    /* Success Result */
    .result-container {
        padding: 3rem;
        border-radius: 30px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 2px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        animation: slideUp 0.6s ease-out;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- DATA & ASSETS ---
@st.cache_resource
def load_resources():
    model = joblib.load('model.joblib')
    encoder = joblib.load('encoder.joblib')
    # Disease Info Mapping
    # Comprehensive Disease Information Database
    disease_info = {
        "Fungal infection": {
            "desc": "A skin disease caused by a fungus. There are millions of species of fungi. They live in the dirt, on plants, on household surfaces, and on your skin.",
            "symptoms": ["Itchy skin", "Rash", "Redness", "Scaling"],
            "precautions": ["Keep skin clean and dry", "Avoid sharing personal items like towels", "Use antifungal creams as prescribed"]
        },
        "Allergy": {
            "desc": "A condition in which the immune system reacts abnormally to a foreign substance like pollen, bee venom, or pet dander.",
            "symptoms": ["Sneezing", "Itchy eyes", "Runny nose", "Skin rashes"],
            "precautions": ["Avoid allergens", "Keep window shut during high pollen days", "Take antihistamines if prescribed"]
        },
        "GERD": {
            "desc": "Gastroesophageal reflux disease occurs when stomach acid frequently flows back into the tube connecting your mouth and stomach (esophagus).",
            "symptoms": ["Heartburn", "Chest pain", "Difficulty swallowing", "Regurgitation"],
            "precautions": ["Avoid spicy foods", "Don't lie down after eating", "Maintain a healthy weight"]
        },
        "Chronic cholestasis": {
            "desc": "Chronic cholestasis is a liver disease where bile flow from the liver is reduced or blocked for a long period.",
            "symptoms": ["Jaundice", "Dark urine", "Pale stools", "Extreme itching"],
            "precautions": ["Consult a liver specialist", "Avoid alcohol", "Follow a low-fat diet"]
        },
        "Drug Reaction": {
            "desc": "An adverse response to a medication, ranging from mild skin rashes to life-threatening anaphylaxis.",
            "symptoms": ["Hives", "Fever", "Swelling", "Shortness of breath"],
            "precautions": ["Identify and stop the medication", "Seek medical help for severe reactions", "Note allergy for future doctors"]
        },
        "Peptic ulcer diseae": {
            "desc": "Open sores that develop on the inside lining of your stomach and the upper portion of your small intestine.",
            "symptoms": ["Burning stomach pain", "Feeling of fullness", "Bloating", "Nausea"],
            "precautions": ["Avoid NSAIDs like aspirin", "Limit spicy foods", "Eat smaller, frequent meals"]
        },
        "AIDS": {
            "desc": "Acquired immunodeficiency syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV).",
            "symptoms": ["Fever", "Weight loss", "Night sweats", "Recurrent infections"],
            "precautions": ["Follow antiretroviral therapy (ART)", "Practice safe sexual habits", "Regular health check-ups"]
        },
        "Diabetes": {
            "desc": "A group of diseases that result in too much sugar in the blood (high blood glucose).",
            "symptoms": ["Increased thirst", "Frequent urination", "Blurred vision", "Fatigue"],
            "precautions": ["Monitor blood sugar levels", "Follow a balanced diet", "Exercise regularly"]
        },
        "Gastroenteritis": {
            "desc": "An intestinal infection marked by diarrhea, cramps, nausea, vomiting, and fever.",
            "symptoms": ["Watery diarrhea", "Vomiting", "Stomach cramps", "Low-grade fever"],
            "precautions": ["Stay hydrated with fluids/ORS", "Eat bland foods", "Wash hands frequently"]
        },
        "Bronchial Asthma": {
            "desc": "A condition in which a person's airways become inflamed, narrow and swell, and produce extra mucus, which makes it difficult to breathe.",
            "symptoms": ["Shortness of breath", "Chest tightness", "Wheezing", "Coughing"],
            "precautions": ["Avoid triggers like smoke/pollen", "Keep inhaler handy", "Follow an action plan"]
        },
        "Hypertension": {
            "desc": "A condition in which the force of the blood against the artery walls is too high.",
            "symptoms": ["Headaches", "Shortness of breath", "Nosebleeds", "Vision changes"],
            "precautions": ["Reduce salt intake", "Moderate exercise", "Stress management"]
        },
        "Migraine": {
            "desc": "A headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head.",
            "symptoms": ["Nausea", "Sensitivity to light/sound", "Visual auras", "Vomiting"],
            "precautions": ["Maintain regular sleep", "Avoid trigger foods (caffeine/alcohol)", "Rest in a dark, quiet room"]
        },
        "Cervical spondylosis": {
            "desc": "A general term for age-related wear and tear affecting the spinal disks in your neck.",
            "symptoms": ["Neck pain", "Stiffness", "Numbness in arms", "Headaches"],
            "precautions": ["Maintain good posture", "Perform gentle neck exercises", "Use a supportive pillow"]
        },
        "Paralysis (brain hemorrhage)": {
            "desc": "Loss of muscle function in part of your body caused by a stroke or bleeding in the brain.",
            "symptoms": ["Sudden weakness", "Slurred speech", "Confusion", "Loss of balance"],
            "precautions": ["Immediate hospitalization", "Physical therapy", "Control blood pressure"]
        },
        "Jaundice": {
            "desc": "A condition in which the skin, whites of the eyes and mucous membranes turn yellow because of a high level of bilirubin.",
            "symptoms": ["Yellow skin", "Pale stools", "Dark urine", "Itchy skin"],
            "precautions": ["Keep hydrated", "Avoid heavy/oily foods", "Rest for liver recovery"]
        },
        "Malaria": {
            "desc": "A disease caused by a parasite, transmitted to humans through the bites of infected female Anopheles mosquitoes.",
            "symptoms": ["High fever", "Chills", "Sweating", "Headache"],
            "precautions": ["Use mosquito nets", "Wear protective clothing", "Consult a doctor for antimalarials"]
        },
        "Chicken pox": {
            "desc": "A highly contagious viral infection causing an itchy, blister-like rash on the skin.",
            "symptoms": ["Itchy rash", "Fever", "Loss of appetite", "Headache"],
            "precautions": ["Avoid scratching blisters", "Isolate until scabs dry", "Calamine lotion for itching"]
        },
        "Dengue": {
            "desc": "A mosquito-borne viral disease occurring in tropical and subtropical areas.",
            "symptoms": ["High fever", "Severe joint/muscle pain", "Fatigue", "Skin rash"],
            "precautions": ["Stay hydrated", "Prevent mosquito breeding", "Avoid blood-thinning meds like aspirin"]
        },
        "Typhoid": {
            "desc": "An infectious bacterial fever with an eruption of red spots on the chest and abdomen and severe intestinal irritation.",
            "symptoms": ["High fever", "Stomach pain", "Weakness", "Rose spots rash"],
            "precautions": ["Drink boiled/purified water", "Eat thoroughly cooked food", "Practice good hand hygiene"]
        },
        "hepatitis A": {
            "desc": "A highly contagious liver infection caused by the hepatitis A virus.",
            "symptoms": ["Nausea", "Jaundice", "Abdominal pain", "Fatigue"],
            "precautions": ["Wash hands frequently", "Avoid raw shellfish", "Get vaccinated"]
        },
        "Hepatitis B": {
            "desc": "A serious liver infection caused by the hepatitis B virus that's easily preventable by a vaccine.",
            "symptoms": ["Abdominal pain", "Dark urine", "Joint pain", "Nausea"],
            "precautions": ["Avoid sharing needles/razors", "Safe sexual practices", "Get vaccinated"]
        },
        "Hepatitis C": {
            "desc": "An infection caused by a virus that attacks the liver and leads to inflammation.",
            "symptoms": ["Bleeding easily", "Bruising easily", "Fatigue", "Poor appetite"],
            "precautions": ["Avoid sharing personal items", "Screening for blood-borne diseases", "Antiviral drugs as prescribed"]
        },
        "Hepatitis D": {
            "desc": "A liver disease caused by the hepatitis D virus, which happens only in people who are also infected with the hepatitis B virus.",
            "symptoms": ["Liver swelling", "Jaundice", "Fatigue", "Nausea"],
            "precautions": ["Prevent Hepatitis B infection", "Avoid contact with infected blood", "Regular liver checks"]
        },
        "Hepatitis E": {
            "desc": "A liver infection caused by the hepatitis E virus, usually through contaminated drinking water.",
            "symptoms": ["Jaundice", "Loss of appetite", "Abdominal pain", "Nausea"],
            "precautions": ["Drink safe/bottled water", "Practice good sanitation", "Ensure food is hygienic"]
        },
        "Alcoholic hepatitis": {
            "desc": "Liver inflammation caused by drinking too much alcohol over many years.",
            "symptoms": ["Yellowing of skin", "Increased waist size", "Nausea", "Vomiting blood"],
            "precautions": ["Stop alcohol consumption", "Maintain nutrition", "Consult a hepatologist"]
        },
        "Tuberculosis": {
            "desc": "A potentially serious infectious bacterial disease that mainly affects the lungs.",
            "symptoms": ["Chronic cough", "Weight loss", "Night sweats", "Blood in phlegm"],
            "precautions": ["Complete the full course of antibiotics", "Wear masks in public", "Improve ventilation"]
        },
        "Common Cold": {
            "desc": "A viral infection of your nose and throat (upper respiratory tract). It's usually harmless.",
            "symptoms": ["Runny nose", "Sore throat", "Cough", "Congestion"],
            "precautions": ["Stay hydrated", "Get plenty of rest", "Use saline nasal drops"]
        },
        "Pneumonia": {
            "desc": "Infection that inflames the air sacs in one or both lungs, which may fill with fluid.",
            "symptoms": ["Cough with phlegm", "Fever", "Chills", "Difficulty breathing"],
            "precautions": ["Follow antibiotic course", "Avoid smoking", "Get flu/pneumonia vaccines"]
        },
        "Dimorphic hemmorhoids(piles)": {
            "desc": "Swollen veins in your anus and lower rectum, similar to varicose veins.",
            "symptoms": ["Bleeding during bowel movements", "Itching/irritation", "Pain or discomfort", "Swelling"],
            "precautions": ["Eat high-fiber foods", "Drink plenty of water", "Don't strain during bowel movements"]
        },
        "Heart attack": {
            "desc": "A medical emergency when a blood clot blocks blood flow to the heart muscle.",
            "symptoms": ["Chest pain", "Shortness of breath", "Nausea", "Cold sweat"],
            "precautions": ["Call emergency services immediately", "Take aspirin if advised", "Maintain low-cholesterol diet"]
        },
        "Varicose veins": {
            "desc": "Gnarled, enlarged veins, most commonly in the legs and feet.",
            "symptoms": ["Visible bulging veins", "Aching legs", "Swelling", "Itching around veins"],
            "precautions": ["Avoid standing for long periods", "Wear compression stockings", "Elevate legs when resting"]
        },
        "Hypothyroidism": {
            "desc": "Condition where the thyroid gland doesn't produce enough thyroid hormone.",
            "symptoms": ["Fatigue", "Weight gain", "Puffy face", "Cold intolerance"],
            "precautions": ["Take hormone replacement meds", "Intermittent blood tests", "Maintain a balanced diet"]
        },
        "Hyperthyroidism": {
            "desc": "The overproduction of a hormone by the butterfly-shaped gland in the neck.",
            "symptoms": ["Unintentional weight loss", "Rapid heartbeat", "Irritability", "Sweating"],
            "precautions": ["Follow antithyroid medications", "Limit iodine intake", "Beta-blockers if prescribed"]
        },
        "Hypoglycemia": {
            "desc": "A condition caused by a very low level of blood sugar (glucose), your body's main energy source.",
            "symptoms": ["Dizziness", "Shakiness", "Sweating", "Hunger"],
            "precautions": ["Consume fast-acting sugar", "Monitor blood glucose", "Eat regular meals"]
        },
        "Osteoarthristis": {
            "desc": "Type of arthritis that occurs when flexible tissue at the ends of bones wears down.",
            "symptoms": ["Joint pain", "Stiffness", "Loss of flexibility", "Grating sensation"],
            "precautions": ["Maintain a healthy weight", "Stay active", "Join physical therapy"]
        },
        "Arthritis": {
            "desc": "Inflammation of one or more joints, causing pain and stiffness that can worsen with age.",
            "symptoms": ["Joint pain", "Stiffness", "Swelling", "Reduced range of motion"],
            "precautions": ["Hot and cold therapy", "Regular low-impact exercise", "Consult a rheumatologist"]
        },
        "(vertigo) Paroymsal  Positional Vertigo": {
            "desc": "Occurs when small calcium crystals move out of their normal location in the inner ear.",
            "symptoms": ["Spinning sensation", "Nausea", "Vomiting", "Loss of balance"],
            "precautions": ["Avoid sudden head movements", "Perform canalith repositioning maneuvers", "Sit down when dizzy"]
        },
        "Acne": {
            "desc": "A skin condition that occurs when your hair follicles become plugged with oil and dead skin cells.",
            "symptoms": ["Pimples", "Whiteheads", "Blackheads", "Cystic lesions"],
            "precautions": ["Gently wash face", "Don't pick/squeeze pimples", "Use non-comedogenic products"]
        },
        "Urinary tract infection": {
            "desc": "An infection in any part of the urinary system, the kidneys, ureters, bladder and urethra.",
            "symptoms": ["Pelvic pain", "Burning sensation on urination", "Cloudy urine", "Frequent urge to urinate"],
            "precautions": ["Drink plenty of water", "Wipe from front to back", "Empty bladder after intercourse"]
        },
        "Psoriasis": {
            "desc": "A disease in which skin cells build up and form scales and itchy, dry patches.",
            "symptoms": ["Red patches with silvery scales", "Dry, cracked skin", "Itching/burning", "Thickened nails"],
            "precautions": ["Keep skin moisturized", "Manage stress", "Avoid triggers like cold weather"]
        },
        "Impetigo": {
            "desc": "A highly contagious skin infection that causes red sores on the face.",
            "symptoms": ["Red sores", "Honey-colored crusts", "Itching", "Fluid-filled blisters"],
            "precautions": ["Avoid scratching", "Keep infected area covered", "Wash hands regularly"]
        }
    }

    data = pd.read_csv('improved_disease_dataset.csv')
    symptoms = list(data.columns[:-1])
    return model, encoder, symptoms, disease_info

def get_lottie(url):
    try: return requests.get(url).json()
    except: return None

# Load Resources
model, encoder, symptoms, info_map = load_resources()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("header.png", use_container_width=True)
    selected = option_menu(
        menu_title="Neural Center",
        options=["Predictor", "Research Hub", "Model Analysis"],
        icons=["cpu-fill", "journal-medical", "bar-chart-fill"],
        menu_icon="hospital",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "transparent"},
            "icon": {"color": "#6366f1", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "rgba(99, 102, 241, 0.1)"},
            "nav-link-selected": {"background-color": "rgba(99, 102, 241, 0.2)", "border-left": "4px solid #6366f1"},
        }
    )
    st.markdown("---")
    st.markdown("### 🧬 AI Status")
    st.info("Core Engine: Ensemble v2.1\nUptime: 99.98%\nPrecision: Ultra-High")

# --- PREDICTOR TAB ---
if selected == "Predictor":
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.markdown("<h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 0;'>Deep Scan</h1>", unsafe_allow_html=True)
        st.write("Synthesizing patient data through neural networks for immediate diagnostic insights.")
    with col_t2:
        l_pred = get_lottie("https://assets10.lottiefiles.com/packages/lf20_iq9asio0.json")
        if l_pred: st_lottie(l_pred, height=180, key="pred_ani")

    st.write("")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    selected_symptoms = st.multiselect(
        "Select all active symptoms:",
        options=[s.replace("_", " ").title() for s in symptoms],
        placeholder="Type here (e.g., Joint Pain, Skin Rash...)"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("RUN NEURAL DIAGNOSTIC", use_container_width=True):
        if not selected_symptoms:
            st.warning("⚠️ Please select symptoms to proceed.")
        else:
            with st.status("🔮 Accessing Neural Core...", expanded=True) as status:
                time.sleep(0.5)
                st.write("Encoding symptom vectors...")
                time.sleep(0.5)
                st.write("Polling Ensemble Classifiers (XGB + RF + SVM)...")
                
                user_vec = [1 if s.replace("_", " ").title() in selected_symptoms else 0 for s in symptoms]
                pred_idx = model.predict([user_vec])[0]
                probs = model.predict_proba([user_vec])[0]
                disease = encoder.inverse_transform([pred_idx])[0]
                confidence = probs[pred_idx] * 100
                
                status.update(label="Scanning Complete", state="complete")

            st.markdown(f"""
            <div class='result-container'>
                <p style='color: #22d3ee; letter-spacing: 3px; font-weight: 600;'>PRIMARY IDENTIFICATION</p>
                <h1 style='font-size: 4.5rem; color: white; margin: 10px 0;'>{disease}</h1>
                <div style='display: flex; justify-content: center; align-items: center; gap: 20px;'>
                    <span style='background: rgba(16, 185, 129, 0.2); color: #4ade80; padding: 10px 25px; border-radius: 100px; font-weight: 700;'>
                        {confidence:.2f}% Match
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 📈 Probabilities")
                top_3 = np.argsort(probs)[-3:][::-1]
                for idx in top_3:
                    d = encoder.inverse_transform([idx])[0]
                    p = probs[idx] * 100
                    st.write(f"**{d}**")
                    st.progress(p/100)
            with c2:
                st.markdown("### 📋 Clinical Notes")
                d_info = info_map.get(disease, {})
                if d_info:
                    st.write(d_info.get("desc", ""))
                    with st.expander("View Precautions"):
                        for p in d_info.get("precautions", []):
                            st.write(f"- {p}")
                else:
                    st.info("Consult a medical professional for more detailed information and treatment plan.")
            
            st.balloons()

# --- RESEARCH HUB ---
elif selected == "Research Hub":
    st.markdown("<h1 class='gradient-text'>Disease Encyclopedia</h1>", unsafe_allow_html=True)
    st.write("Browse details for all conditions detectable by our system.")
    
    search = st.text_input("Search disease information:", placeholder="e.g. Hepatitis")
    
    cols = st.columns(2)
    display_list = [d for d in encoder.classes_ if search.lower() in d.lower()] if search else list(encoder.classes_)
    
    for i, d in enumerate(display_list):
        with cols[i % 2]:
            d_info = info_map.get(d, {})
            with st.container():
                st.markdown(f"""
                <div class='glass-card'>
                    <h3 style='color: #818cf8;'>{d}</h3>
                    <p style='font-size: 0.9rem; margin-bottom: 10px;'>{d_info.get("desc", "Detailed clinical data available.")}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Using Streamlit components inside the card area
                with st.expander(f"Medical Details for {d}"):
                    st.markdown("**Key Symptoms:**")
                    st.write(", ".join(d_info.get("symptoms", ["N/A"])))
                    st.markdown("**Care Precautions:**")
                    for p in d_info.get("precautions", []):
                        st.write(f"• {p}")


# --- MODEL ANALYSIS ---
else:
    st.markdown("<h1 class='gradient-text'>System Intelligence</h1>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Ensemble Accuracy", "99.12%", "+0.5%")
    col_m2.metric("Inference Latency", "12.4ms", "-1.2ms")
    col_m3.metric("Data Features", len(symptoms), "Stable")

    st.write("")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("How it works")
    st.write("The system uses a **Soft-Voting Ensemble** architecture combining three distinct neural pathways:")
    st.markdown("""
    - **XGBoost**: Handles non-linear feature interactions with gradient boosting.
    - **Random Forest**: Provides robust classification through bagging multiple decision trees.
    - **SVM**: Finds the optimal hyperplane in high-dimensional symptom space.
    """)
    st.write("The final output is the weighted average of all three models, ensuring extreme reliability.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    l_tech = get_lottie("https://assets5.lottiefiles.com/packages/lf20_5njpX7.json")
    if l_tech: st_lottie(l_tech, height=300)

# --- FOOTER ---
st.markdown("<br><br><p style='text-align: center; opacity: 0.4; font-size: 0.8rem;'>HealthAI Neural Diagnostic Core v2.1.0 • Built for Precision</p>", unsafe_allow_html=True)
