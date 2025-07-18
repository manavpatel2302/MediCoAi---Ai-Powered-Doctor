import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
import json
import requests
import asyncio # Import asyncio for running async function

# Suppress all warnings for a cleaner demo output
warnings.filterwarnings('ignore')

# --- 1. Streamlit Page Configuration ---
st.set_page_config(
    page_title="Unified Health AI Assistant",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better aesthetics (Combined from both files, prioritizing ml.py's general styles) ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        cursor: pointer;
        transform: scale(1.02); /* Slight hover effect */
    }
    .stAlert {
        border-radius: 8px;
    }
    /* Updated styling for radio buttons to ensure better visibility and spacing */
    .stRadio > label {
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 5px; /* Space between label and options */
        display: block; /* Ensure label takes full width */
    }
    .stRadio > div > div { /* Container for radio options */
        display: flex; /* Arrange options horizontally */
        gap: 20px; /* Space between radio buttons */
        justify-content: center; /* Center the radio buttons */
        margin-top: 5px;
        margin-bottom: 15px; /* Space after the radio buttons */
    }
    .stRadio div[data-baseweb="radio"] { /* Individual radio button */
        padding: 8px 15px;
        border: 1px solid #ccc;
        border-radius: 8px;
        background-color: #ffffff;
        transition: all 0.2s ease-in-out;
    }
    .stRadio div[data-baseweb="radio"]:hover {
        background-color: #f0f0f0;
    }
    .stRadio div[data-baseweb="radio"][aria-checked="true"] {
        background-color: #d4edda; /* Light green when selected */
        border-color: #4CAF50; /* Green border when selected */
        color: #2c3e50;
    }

    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div>div {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    h1 {
        color: #00FFBC; /* Original from ml.py */
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    h2, h3 {
        color: #2c3e50;
        text-align: center;
    }
    /* Specific styles for prediction/recommendation boxes */
    .prediction-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        margin-top: 20px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
    }
    .recommendation-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2196F3;
        margin-top: 15px;
        font-size: 16px;
        color: #2c3e50;
    }
    /* Styles for AI Medical Assistant specific elements */
    .stTextArea label {
        font-weight: bold;
        color: #004d40;
    }
    .stSuccess {
        background-color: #e8f5e9; /* Light green for success */
        border-left: 5px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
    }
    .stWarning {
        background-color: #fff3e0; /* Light orange for warning */
        border-left: 5px solid #ff9800;
        padding: 10px;
        border-radius: 5px;
    }
    .stInfo {
        background-color: #e3f2fd; /* Light blue for info */
        border-left: 5px solid #2196f3;
        padding: 10px;
        border-radius: 5px;
    }
    .stError {
        background-color: #ffebee; /* Light red for error */
        border-left: 5px solid #f44336;
        padding: 10px;
        border-radius: 5px;
    }
    /* Watermark style */
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 0.8em;
        color: rgba(0, 0, 0, 0.3); /* Semi-transparent black */
        z-index: 1000; /* Ensure it's on top */
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Simulate Data and Train a Dummy ML Model for Chronic Disease Prediction ---
@st.cache_resource # Cache the model training to avoid re-running on every interaction
def train_dummy_chronic_disease_model():
    # Create a synthetic dataset for chronic disease (e.g., Diabetes) prediction
    np.random.seed(42)
    num_samples = 1000
    data = {
        'Age': np.random.randint(20, 70, num_samples),
        'BMI': np.random.uniform(18.0, 40.0, num_samples),
        'Glucose': np.random.uniform(70.0, 200.0, num_samples),
        'BloodPressure': np.random.uniform(80.0, 180.0, num_samples),
        'FamilyHistory': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]), # 0: No, 1: Yes
        'Smoking': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),       # 0: No, 1: Yes
        'PhysicalActivity_hours_week': np.random.uniform(1, 10, num_samples),
        'DiabetesRisk': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]) # 0: Low Risk, 1: High Risk
    }
    df = pd.DataFrame(data)

    # Introduce some correlation for 'DiabetesRisk'
    df.loc[df['Glucose'] > 140, 'DiabetesRisk'] = 1
    df.loc[df['BMI'] > 30, 'DiabetesRisk'] = 1
    df.loc[(df['Age'] > 50) & (df['FamilyHistory'] == 1), 'DiabetesRisk'] = 1
    df.loc[(df['PhysicalActivity_hours_week'] < 3) & (df['BMI'] > 28), 'DiabetesRisk'] = 1

    X = df[['Age', 'BMI', 'Glucose', 'BloodPressure', 'FamilyHistory', 'Smoking', 'PhysicalActivity_hours_week']]
    y = df['DiabetesRisk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Use RandomForestClassifier as a robust dummy model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.success(f"Chronic Disease Model Trained! Accuracy: {accuracy:.2f}")

    return model, X.columns

# Load and train the dummy chronic disease model
chronic_disease_model, chronic_disease_features = train_dummy_chronic_disease_model()

# --- Mock JSON Data for AI Medical Assistant (from medicin.py) ---
MOCK_JSON_EXAMPLES = [
    {
        "disease": "Common Cold",
        "medicines": ["Paracetamol", "Ibuprofen", "Decongestant nasal spray"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Influenza (Flu)",
        "medicines": ["Oseltamivir (prescription)", "Paracetamol", "Plenty of fluids"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Allergic Rhinitis",
        "medicines": ["Antihistamines", "Nasal corticosteroids", "Saline nasal spray"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Migraine",
        "medicines": ["Ibuprofen", "Sumatriptan (prescription)", "Caffeine"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Tension Headache",
        "medicines": ["Paracetamol", "Ibuprofen", "Muscle relaxants (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Type 2 Diabetes",
        "medicines": ["Metformin (prescription)", "Insulin (prescription)", "Diet and exercise"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Hypertension (High Blood Pressure)",
        "medicines": ["Diuretics (prescription)", "ACE inhibitors (prescription)", "Lifestyle changes"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Coronary Artery Disease",
        "medicines": ["Aspirin", "Statins (prescription)", "Beta-blockers (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Asthma",
        "medicines": ["Albuterol (inhaler, prescription)", "Corticosteroid inhalers (prescription)", "Leukotriene modifiers (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Depression",
        "medicines": ["SSRIs (prescription)", "SNRIs (prescription)", "Therapy"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Anxiety Disorder",
        "medicines": ["SSRIs (prescription)", "Benzodiazepines (prescription, short-term)", "Therapy"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Osteoporosis",
        "medicines": ["Bisphosphonates (prescription)", "Calcium supplements", "Vitamin D supplements"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Anemia (Iron Deficiency)",
        "medicines": ["Iron supplements", "Vitamin C (to aid absorption)", "Dietary changes"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Gout",
        "medicines": ["NSAIDs", "Colchicine (prescription)", "Allopurinol (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Osteoarthritis",
        "medicines": ["NSAIDs", "Acetaminophen", "Topical pain relievers"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Rheumatoid Arthritis",
        "medicines": ["DMARDs (prescription)", "NSAIDs", "Corticosteroids (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Hypothyroidism",
        "medicines": ["Levothyroxine (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Hyperthyroidism",
        "medicines": ["Antithyroid drugs (prescription)", "Beta-blockers (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Gastroesophageal Reflux Disease (GERD)",
        "medicines": ["Antacids", "PPIs (prescription)", "H2 blockers"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Irritable Bowel Syndrome (IBS)",
        "medicines": ["Antispasmodics (prescription)", "Fiber supplements", "Probiotics"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Chronic Obstructive Pulmonary Disease (COPD)",
        "medicines": ["Bronchodilators (prescription)", "Corticosteroids (inhaler, prescription)", "Oxygen therapy (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Pneumonia",
        "medicines": ["Antibiotics (prescription)", "Fever reducers", "Cough medicine"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Urinary Tract Infection (UTI)",
        "medicines": ["Antibiotics (prescription)", "Pain relievers", "Increased fluid intake"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Sinusitis",
        "medicines": ["Decongestants", "Nasal corticosteroids", "Pain relievers"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Dermatitis (Eczema)",
        "medicines": ["Topical corticosteroids", "Moisturizers", "Antihistamines"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Acne Vulgaris",
        "medicines": ["Benzoyl peroxide", "Salicylic acid", "Topical retinoids (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Psoriasis",
        "medicines": ["Topical corticosteroids", "Vitamin D analogs (prescription)", "Phototherapy"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Kidney Stones",
        "medicines": ["Pain relievers", "Alpha-blockers (prescription)", "Increased fluid intake"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Gallstones",
        "medicines": ["Pain relievers", "Ursodeoxycholic acid (prescription, for small stones)", "Surgery (for severe cases)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Depression",
        "medicines": ["SSRIs (prescription)", "SNRIs (prescription)", "Therapy"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Anxiety Disorder",
        "medicines": ["SSRIs (prescription)", "Benzodiazepines (prescription, short-term)", "Therapy"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Insomnia",
        "medicines": ["Sleep aids (prescription, short-term)", "Melatonin", "CBT-I"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Parkinson's Disease",
        "medicines": ["Levodopa (prescription)", "Dopamine agonists (prescription)", "MAO-B inhibitors (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Alzheimer's Disease",
        "medicines": ["Cholinesterase inhibitors (prescription)", "Memantine (prescription)", "Supportive care"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Multiple Sclerosis (MS)",
        "medicines": ["Disease-modifying therapies (prescription)", "Corticosteroids (prescription, for relapses)", "Symptom management"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Epilepsy",
        "medicines": ["Antiepileptic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Glaucoma",
        "medicines": ["Eye drops (prescription)", "Laser therapy (procedure)", "Surgery (procedure)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Cataracts",
        "medicines": ["Surgery (primary treatment)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Diabetic Retinopathy",
        "medicines": ["Laser treatment (procedure)", "Anti-VEGF injections (prescription)", "Blood sugar control"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Chronic Kidney Disease (CKD)",
        "medicines": ["Blood pressure medication (prescription)", "Diuretics (prescription)", "Dialysis (advanced stages)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Liver Cirrhosis",
        "medicines": ["Diuretics (prescription)", "Lactulose (prescription)", "Liver transplant (severe cases)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Pancreatitis",
        "medicines": ["Pain relievers", "IV fluids", "Enzyme supplements (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Stroke",
        "medicines": ["Thrombolytics (emergency, prescription)", "Antiplatelet drugs (prescription)", "Rehabilitation"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Heart Failure",
        "medicines": ["ACE inhibitors (prescription)", "Beta-blockers (prescription)", "Diuretics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Arrhythmia (e.g., Atrial Fibrillation)",
        "medicines": ["Beta-blockers (prescription)", "Anticoagulants (prescription)", "Antiarrhythmic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Peripheral Artery Disease (PAD)",
        "medicines": ["Statins (prescription)", "Antiplatelet drugs (prescription)", "Exercise program"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Atherosclerosis",
        "medicines": ["Statins (prescription)", "Aspirin", "Lifestyle changes"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Prostate Enlargement (BPH)",
        "medicines": ["Alpha-blockers (prescription)", "5-alpha reductase inhibitors (prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Polycystic Ovary Syndrome (PCOS)",
        "medicines": ["Birth control pills (prescription)", "Metformin (prescription)", "Lifestyle changes"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Endometriosis",
        "medicines": ["Pain relievers", "Hormone therapy (prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Uterine Fibroids",
        "medicines": ["Pain relievers", "Hormone therapy (prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Menopause Symptoms",
        "medicines": ["Hormone replacement therapy (HRT, prescription)", "SSRIs (prescription)", "Lifestyle changes"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Chronic Fatigue Syndrome (CFS)",
        "medicines": ["Symptom management", "Graded exercise therapy", "CBT"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Lupus (Systemic Lupus Erythematosus)",
        "medicines": ["NSAIDs", "Corticosteroids (prescription)", "Immunosuppressants (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Sjogren's Syndrome",
        "medicines": ["Artificial tears", "Saliva substitutes", "Immunosuppressants (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Scleroderma",
        "medicines": ["Immunosuppressants (prescription)", "Medications for specific symptoms (e.g., GERD)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Psoriatic Arthritis",
        "medicines": ["NSAIDs", "DMARDs (prescription)", "Biologics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Ankylosing Spondylitis",
        "medicines": ["NSAIDs", "DMARDs (prescription)", "Biologics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Raynaud's Phenomenon",
        "medicines": ["Calcium channel blockers (prescription)", "Avoid cold", "Gloves"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Deep Vein Thrombosis (DVT)",
        "medicines": ["Anticoagulants (prescription)", "Compression stockings"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Pulmonary Embolism",
        "medicines": ["Anticoagulants (prescription)", "Thrombolytics (prescription, emergency)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Hypertensive Crisis",
        "medicines": ["Emergency blood pressure lowering medication (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Myocardial Infarction (Heart Attack)",
        "medicines": ["Aspirin", "Nitroglycerin (prescription)", "Beta-blockers (prescription)", "Emergency medical attention"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Angina",
        "medicines": ["Nitroglycerin (prescription)", "Beta-blockers (prescription)", "Calcium channel blockers (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Congestive Heart Failure",
        "medicines": ["Diuretics (prescription)", "ACE inhibitors (prescription)", "Beta-blockers (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Valve Disease (Heart)",
        "medicines": ["Diuretics (prescription)", "Blood pressure medication (prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Pericarditis",
        "medicines": ["NSAIDs", "Colchicine (prescription)", "Corticosteroids (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Endocarditis",
        "medicines": ["Antibiotics (IV, prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Pneumothorax (Collapsed Lung)",
        "medicines": ["Oxygen therapy", "Chest tube insertion (procedure)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Pleural Effusion",
        "medicines": ["Diuretics (prescription)", "Thoracentesis (procedure)", "Treat underlying cause"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Croup",
        "medicines": ["Corticosteroids (prescription)", "Humidifier", "Cool mist"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Epiglottitis",
        "medicines": ["Antibiotics (prescription)", "Airway management (emergency)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Bronchiolitis",
        "medicines": ["Supportive care", "Nasal suctioning", "Fluids"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Cystic Fibrosis",
        "medicines": ["Mucus-thinning drugs (prescription)", "Antibiotics (prescription)", "Enzyme supplements (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Tonsillitis",
        "medicines": ["Antibiotics (if bacterial, prescription)", "Pain relievers", "Gargle with salt water"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Laryngitis",
        "medicines": ["Voice rest", "Humidifier", "Pain relievers"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Pharyngitis (Sore Throat)",
        "medicines": ["Pain relievers", "Throat lozenges", "Warm fluids"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Oral Thrush (Candidiasis)",
        "medicines": ["Antifungal mouthwash/lozenge (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Gingivitis",
        "medicines": ["Improved oral hygiene", "Antiseptic mouthwash"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Periodontitis",
        "medicines": ["Deep cleaning (scaling and root planing)", "Antibiotics (prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Dental Caries (Cavities)",
        "medicines": ["Fillings", "Crowns", "Improved oral hygiene"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Abscessed Tooth",
        "medicines": ["Antibiotics (prescription)", "Root canal", "Tooth extraction"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Temporomandibular Joint (TMJ) Disorder",
        "medicines": ["Pain relievers", "Muscle relaxants (prescription)", "Mouthguard"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Bell's Palsy",
        "medicines": ["Corticosteroids (prescription)", "Antivirals (prescription)", "Eye care"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Trigeminal Neuralgia",
        "medicines": ["Anticonvulsants (prescription)", "Muscle relaxants (prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Sciatica",
        "medicines": ["NSAIDs", "Muscle relaxants (prescription)", "Physical therapy"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Herniated Disc",
        "medicines": ["NSAIDs", "Physical therapy", "Corticosteroid injections (prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Spinal Stenosis",
        "medicines": ["NSAIDs", "Physical therapy", "Corticosteroid injections (prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Scoliosis",
        "medicines": ["Bracing", "Physical therapy", "Surgery (severe cases)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Osteomyelitis",
        "medicines": ["Antibiotics (IV, prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Septic Arthritis",
        "medicines": ["Antibiotics (IV, prescription)", "Joint drainage (procedure)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Bursitis",
        "medicines": ["NSAIDs", "Rest", "Corticosteroid injections (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Tendinitis",
        "medicines": ["NSAIDs", "Rest", "Physical therapy"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Rotator Cuff Injury",
        "medicines": ["NSAIDs", "Physical therapy", "Surgery (severe tears)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Meniscus Tear",
        "medicines": ["RICE (Rest, Ice, Compression, Elevation)", "Physical therapy", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "ACL Tear",
        "medicines": ["RICE (Rest, Ice, Compression, Elevation)", "Physical therapy", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Fracture",
        "medicines": ["Immobilization (cast/splint)", "Pain relievers", "Surgery (if needed)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Dislocation",
        "medicines": ["Reduction (repositioning)", "Immobilization", "Pain relievers"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Concussion",
        "medicines": ["Rest (physical and cognitive)", "Pain relievers (acetaminophen)", "Gradual return to activity"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Head Injury (Minor)",
        "medicines": ["Rest", "Pain relievers (acetaminophen)", "Monitoring for worsening symptoms"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Spinal Cord Injury",
        "medicines": ["Emergency stabilization", "Corticosteroids (prescription)", "Rehabilitation"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Brain Tumor",
        "medicines": ["Surgery", "Radiation therapy", "Chemotherapy"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Huntington's Disease",
        "medicines": ["Medications for symptoms (e.g., chorea, depression)", "Therapy"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Amyotrophic Lateral Sclerosis (ALS)",
        "medicines": ["Riluzole (prescription)", "Edaravone (prescription)", "Supportive care"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Guillain-Barré Syndrome",
        "medicines": ["IV immunoglobulin (IVIG)", "Plasma exchange", "Rehabilitation"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Myasthenia Gravis",
        "medicines": ["Cholinesterase inhibitors (prescription)", "Corticosteroids (prescription)", "Immunosuppressants (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Food Allergy",
        "medicines": ["Antihistamines", "Epinephrine auto-injector (prescription)", "Avoidance of allergen"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Drug Allergy",
        "medicines": ["Antihistamines", "Corticosteroids (prescription)", "Avoidance of allergen"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Seasonal Allergies (Hay Fever)",
        "medicines": ["Antihistamines", "Nasal corticosteroids", "Decongestants"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Poison Ivy/Oak/Sumac Rash",
        "medicines": ["Calamine lotion", "Corticosteroid creams", "Antihistamines"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Insect Bites/Stings",
        "medicines": ["Antihistamines", "Calamine lotion", "Hydrocortisone cream"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Sunburn",
        "medicines": ["Aloe vera gel", "Moisturizers", "Pain relievers (e.g., ibuprofen)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Heat Rash",
        "medicines": ["Cool compresses", "Loose clothing", "Calamine lotion"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Diabetic Foot Ulcer",
        "medicines": ["Wound care", "Antibiotics (prescription)", "Offloading (special footwear)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Pressure Ulcers (Bedsores)",
        "medicines": ["Wound care", "Repositioning", "Special mattresses"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Varicose Ulcers",
        "medicines": ["Compression therapy", "Wound care", "Treat underlying venous insufficiency"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Burns (Minor)",
        "medicines": ["Cool water", "Aloe vera", "Pain relievers (e.g., acetaminophen)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Burns (Severe)",
        "medicines": ["Emergency medical attention", "Pain management", "Wound care"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Hypoglycemia (Low Blood Sugar)",
        "medicines": ["Glucose tablets", "Sugary drinks/food", "Glucagon (prescription, emergency)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Hyperglycemia (High Blood Sugar)",
        "medicines": ["Insulin (prescription)", "Increased fluid intake", "Dietary adjustments"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Diabetic Ketoacidosis (DKA)",
        "medicines": ["Emergency medical attention", "Insulin (IV, prescription)", "IV fluids", "Electrolyte replacement"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Thyroid Storm",
        "medicines": ["Emergency medical attention", "Antithyroid drugs (prescription)", "Beta-blockers (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Adrenal Crisis",
        "medicines": ["Emergency medical attention", "Corticosteroids (IV, prescription)", "IV fluids"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Pneumocystis Pneumonia (PCP)",
        "medicines": ["Antibiotics (prescription)", "Corticosteroids (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Toxoplasmosis",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Cryptosporidiosis",
        "medicines": ["Antiparasitic drugs (prescription)", "Oral rehydration salts"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Giardiasis",
        "medicines": ["Antiparasitic drugs (prescription)", "Oral rehydration salts"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Amoebiasis",
        "medicines": ["Antiparasitic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Schistosomiasis",
        "medicines": ["Praziquantel (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Filariasis",
        "medicines": ["Antiparasitic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Leishmaniasis",
        "medicines": ["Antiparasitic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Chagas Disease",
        "medicines": ["Antiparasitic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Sleeping Sickness (African Trypanosomiasis)",
        "medicines": ["Antiparasitic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Listeriosis",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Botulism",
        "medicines": ["Antitoxin (prescription)", "Supportive care"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Anthrax",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Plague",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Tularemia",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Brucellosis",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Q Fever",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Legionnaires' Disease",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Psittacosis",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Mycoplasma Pneumonia",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Chlamydia Pneumonia",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Histoplasmosis",
        "medicines": ["Antifungal drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Coccidioidomycosis (Valley Fever)",
        "medicines": ["Antifungal drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Blastomycosis",
        "medicines": ["Antifungal drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Aspergillosis",
        "medicines": ["Antifungal drugs (prescription)", "Surgery (if needed)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Cryptococcosis",
        "medicines": ["Antifungal drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Candidemia",
        "medicines": ["Antifungal drugs (IV, prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Mucormycosis",
        "medicines": ["Antifungal drugs (IV, prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Zygomycosis",
        "medicines": ["Antifungal drugs (IV, prescription)", "Surgery"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Pneumocystis Pneumonia (PCP)",
        "medicines": ["Antibiotics (prescription)", "Corticosteroids (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Toxoplasmosis",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Cryptosporidiosis",
        "medicines": ["Antiparasitic drugs (prescription)", "Oral rehydration salts"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Giardiasis",
        "medicines": ["Antiparasitic drugs (prescription)", "Oral rehydration salts"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Amoebiasis",
        "medicines": ["Antiparasitic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Schistosomiasis",
        "medicines": ["Praziquantel (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Filariasis",
        "medicines": ["Antiparasitic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Leishmaniasis",
        "medicines": ["Antiparasitic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Chagas Disease",
        "medicines": ["Antiparasitic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Sleeping Sickness (African Trypanosomiasis)",
        "medicines": ["Antiparasitic drugs (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    },
    {
        "disease": "Listeriosis",
        "medicines": ["Antibiotics (prescription)"],
        "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
    }
]


# Function to call the Gemini API for text generation (from medicin.py)
async def generate_text_from_gemini(prompt):
    """
    Calls the Gemini API to generate text based on a given prompt.

    Args:
        prompt (str): The input prompt for the language model.

    Returns:
        dict: A dictionary containing the parsed JSON response from the Gemini model,
              or an error message if the API call fails or the response is malformed.
    """
    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "disease": {"type": "STRING"},
                    "medicines": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "disclaimer": {"type": "STRING"}
                },
                "propertyOrdering": ["disease", "medicines", "disclaimer"]
            }
        }
    }

    # IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual Google Cloud API Key.
    # For production, use environment variables or a secure secret management system.
    # The API key from medicin.py is used here.
    api_key = "AIzaSyDfJMS4Ee7U6Y5jKdP5Tih3OcItX2QNxd4"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    try:
        response = requests.post(
            api_url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json()

        if result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           len(result["candidates"][0]["content"]["parts"]) > 0:
            
            json_string = result["candidates"][0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(json_string)
            return parsed_json
        else:
            return {"error": "Unexpected API response structure or no content."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}. Please check your internet connection or API access."}
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON response from API. The model might not have returned valid JSON."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


# --- Main Streamlit Application Layout ---

st.title("⚕️ Unified Health AI Assistant")
st.markdown("### Your AI-powered tool for chronic disease risk prediction and symptom-based medical assistance.")
st.markdown("---")

st.markdown("""
    <p style='font-size: 16px; text-align: center; color: #e74c3c;'>
    <b>Disclaimer: This is a demonstration model and should NOT be used for actual medical diagnosis or treatment decisions. Always consult a qualified healthcare professional.</b>
    </p>
    """, unsafe_allow_html=True)

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📊 Chronic Disease Risk Predictor", "🩺 AI Medical Assistant", "💊 Medicine Usage Lookup"])

with tab1:
    st.header("Chronic Disease Risk Predictor")
    st.markdown("""
        <p style='font-size: 18px; text-align: center;'>
        Analyze patient data to predict the likelihood of developing chronic diseases (e.g., Diabetes) and get personalized health recommendations.
        </p>
        """, unsafe_allow_html=True)

    # --- User Input Form for Chronic Disease Predictor ---
    with st.form("chronic_disease_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (Years)", min_value=1, max_value=120, value=35, step=1)
            bmi = st.number_input("BMI (e.g., 25.0)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=90.0, step=0.1)
        with col2:
            blood_pressure = st.number_input("Blood Pressure (Systolic, mmHg)", min_value=70.0, max_value=250.0, value=120.0, step=0.1)
            # Changed st.selectbox to st.radio for better rendering as per user's image issue
            family_history = st.radio("Family History of Chronic Disease?", ("No", "Yes"), horizontal=True)
            smoking = st.radio("Smoker?", ("No", "Yes"), horizontal=True)
            physical_activity = st.number_input("Physical Activity (Hours/Week)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)

        # Convert categorical inputs to numerical for the model
        family_history_encoded = 1 if family_history == "Yes" else 0
        smoking_encoded = 1 if smoking == "Yes" else 0

        submitted_chronic = st.form_submit_button("Predict Risk & Get Recommendations")

        if submitted_chronic:
            # Prepare input for the model
            input_data = pd.DataFrame([[age, bmi, glucose, blood_pressure, family_history_encoded, smoking_encoded, physical_activity]],
                                      columns=chronic_disease_features)

            # Make prediction
            prediction_proba = chronic_disease_model.predict_proba(input_data)[0][1] # Probability of being high risk (class 1)
            prediction_class = chronic_disease_model.predict(input_data)[0]

            st.markdown("---")
            st.subheader("Prediction Results")

            risk_level = ""
            recommendations = []

            if prediction_class == 1: # High Risk
                risk_level = "High"
                st.markdown(f"<div class='prediction-box' style='background-color:#ffebee; border-color:#e74c3c;'>Likelihood of Diabetes: <span style='color:#e74c3c;'>{risk_level} ({prediction_proba*100:.1f}%)</span></div>", unsafe_allow_html=True)
                recommendations.append("Immediate consultation with a healthcare specialist is highly recommended for further assessment and personalized management plan.")
                recommendations.append("Consider detailed diagnostic tests (e.g., HbA1c, fasting blood sugar) as advised by your doctor.")
                recommendations.append("Adopt a balanced, low-sugar, low-processed food diet. Focus on whole grains, lean proteins, and plenty of vegetables.")
                recommendations.append("Aim for at least 150 minutes of moderate-intensity aerobic exercise per week (e.g., brisk walking, cycling).")
                recommendations.append("Monitor your blood sugar levels regularly if advised by your physician.")
                recommendations.append("Manage stress through mindfulness, yoga, or meditation.")
            else: # Low Risk
                risk_level = "Low"
                st.markdown(f"<div class='prediction-box'>Likelihood of Diabetes: <span style='color:#4CAF50;'>{risk_level} ({prediction_proba*100:.1f}%)</span></div>", unsafe_allow_html=True)
                recommendations.append("Continue maintaining a healthy lifestyle to keep your risk low.")
                recommendations.append("Regular physical activity (at least 30 minutes most days of the week) is beneficial.")
                recommendations.append("Maintain a balanced diet rich in fruits, vegetables, and whole foods.")
                recommendations.append("Schedule regular health check-ups with your doctor for preventive care.")
                recommendations.append("Stay hydrated and ensure adequate sleep.")

            st.subheader("Personalized Health Recommendations")
            for i, rec in enumerate(recommendations):
                st.markdown(f"<div class='recommendation-box'>{i+1}. {rec}</div>", unsafe_allow_html=True)

with tab2:
    st.header("AI Medical Assistant")
    st.markdown("### Your AI-powered symptom checker and medicine suggester.")

    st.markdown("""
    **How it works:**
    1.  **Enter Symptoms:** Describe the patient's symptoms in detail.
    2.  **Get Diagnosis:** Our AI will analyze the symptoms to suggest a probable disease.
    3.  **Medicine Suggestions:** Receive a list of common over-the-counter or generally prescribed medicines.
    4.  **Crucial Disclaimer:** Always remember, this is AI-generated information and **not a substitute for professional medical advice.**
    """)

    st.write("---")

    # Input for symptoms
    symptoms = st.text_area(
        "📝 **Describe the patient's symptoms here:** \n(e.g., 'high fever, persistent cough, sore throat, body aches, chills')",
        height=150,
        placeholder="Type symptoms like 'headache, nausea, fatigue'..."
    )

    if st.button("🚀 Get AI Diagnosis & Medicine"):
        if symptoms:
            st.info("🧠 Analyzing symptoms and consulting vast medical knowledge... Please wait.")
            
            # Construct the prompt for the LLM - Improved for broader common disease coverage
            prompt = f"""
            As an expert AI medical doctor, analyze the following patient symptoms thoroughly.
            Identify the most probable common or normal disease, such as loose motion, asthma, common cold, fever, headache, etc.
            Then, provide:
            1. The most probable disease.
            2. A list of 3-5 common over-the-counter or generally prescribed medicines for this disease.
               (Strictly do not include prescription-only drugs unless absolutely necessary and clearly state if any are prescription-only).
            3. A clear and prominent disclaimer stating that this is an AI-generated suggestion and not a substitute for professional medical advice, and that a qualified doctor must be consulted for accurate diagnosis and treatment.

            Patient Symptoms: {symptoms}

            Please provide the output in a JSON format with keys: "disease", "medicines" (an array of strings), and "disclaimer".
            Example JSON format:
            {{
                "disease": "Common Cold",
                "medicines": ["Paracetamol", "Ibuprofen", "Decongestant nasal spray"],
                "disclaimer": "This is an AI-generated suggestion and not a substitute for professional medical advice. Consult a qualified doctor for accurate diagnosis and treatment."
            }}
            """

            # Use st.spinner for a loading indicator
            with st.spinner("Processing your request..."):
                # Call the async function and wait for its result
                result = asyncio.run(generate_text_from_gemini(prompt))

            st.write("---") # Separator for results

            if result and not result.get("error"):
                st.markdown("### ✨ **AI Diagnosis Result:**")
                st.success(f"**Probable Disease:** {result.get('disease', 'Not determined')}")

                st.markdown("### 💊 **Suggested Medicines:**")
                if result.get("medicines"):
                    for i, medicine in enumerate(result["medicines"]):
                        st.markdown(f"- **{medicine}**")
                else:
                    st.warning("No specific medicines suggested or could not parse the medicine list.")

                st.markdown("### ⚠️ **Important Medical Disclaimer:**")
                with st.expander("Read Full Disclaimer"):
                    st.warning(
                        result.get("disclaimer", "This is an AI-generated suggestion and not a substitute for professional medical advice. "
                                                 "Always consult a qualified doctor for accurate diagnosis and treatment. "
                                                 "Do not self-medicate based on AI suggestions.")
                    )
            else:
                st.error(f"❌ **Error:** {result.get('error', 'Could not get a valid response from the AI.')}")
                st.error("Please refine your symptoms or try again later.")
        else:
            st.warning("⚠️ Please enter some symptoms in the text area above to get a diagnosis.")

    st.write("---")

    # --- Section to display JSON examples ---
    with st.expander("📚 Browse Example Diagnoses & Medicines"):
        st.markdown("Here are some examples of AI-generated diagnoses and medicine suggestions:")

        # Prepare data for DataFrame
        example_data = []
        for i, item in enumerate(MOCK_JSON_EXAMPLES):
            # Create a simple symptom string for display in the table
            symptom_placeholder = f"Symptoms for {item['disease']}"
            example_data.append({
                "ID": i + 1,
                "Probable Disease": item.get("disease", "N/A"),
                "Suggested Medicines": ", ".join(item.get("medicines", ["N/A"])),
                "Example Symptoms": symptom_placeholder # Placeholder for symptoms
            })

        df = pd.DataFrame(example_data)
        # Display the DataFrame as a table
        st.dataframe(df, height=300, use_container_width=True)

        st.markdown("""
        <p style='font-size: 0.8em; color: #888;'>
            Note: The 'Example Symptoms' are placeholders derived from the disease name for illustrative purposes.
            The AI generates detailed symptoms based on your input.
        </p>
        """, unsafe_allow_html=True)

with tab3:
    st.header("💊 Medicine Usage Lookup")
    st.markdown("### Find out what diseases a specific medicine is commonly used for based on our data.")

    medicine_name_input = st.text_input(
        "🔍 **Enter Medicine Name:**",
        placeholder="e.g., Paracetamol, Ibuprofen, Metformin"
    )

    if st.button("Search Medicine Usage"):
        if medicine_name_input:
            search_term = medicine_name_input.strip().lower()
            found_uses = []

            for entry in MOCK_JSON_EXAMPLES:
                for medicine in entry.get("medicines", []):
                    if search_term in medicine.lower():
                        found_uses.append(entry)
                        break # Found in this entry, move to the next example

            if found_uses:
                st.subheader(f"Diseases where '{medicine_name_input}' is suggested:")
                for i, use in enumerate(found_uses):
                    st.markdown(f"**Disease:** {use['disease']}")
                    st.markdown(f"**Suggested Medicines for {use['disease']}:** {', '.join(use['medicines'])}")
                    st.markdown("---")
            else:
                st.warning(f"No diseases found where '{medicine_name_input}' is suggested in our current data. Please try another medicine name.")
        else:
            st.warning("Please enter a medicine name to search.")

# --- 4. Sidebar for Additional Information (Combined) ---
st.sidebar.title("About This Unified Demo")
st.sidebar.info(
    """
    This Streamlit application combines three functionalities:
    1.  **Chronic Disease Risk Predictor:** Uses a dummy ML model to predict chronic disease risk (e.g., Diabetes) based on simulated patient data and provides health recommendations.
    2.  **AI Medical Assistant:** Uses the Google Gemini API to provide AI-generated diagnoses and medicine suggestions based on user-entered symptoms.
    3.  **Medicine Usage Lookup:** Allows users to search for diseases associated with a given medicine name based on predefined mock data.

    **Important Considerations (for a real solution):**
    -   **Data Privacy (DPDP Act, India):** A real solution would require explicit, granular consent from patients for data collection, processing, and sharing, adhering strictly to the Digital Personal Data Protection Act, 2023. Data would be securely stored (encrypted at rest and in transit) and access controlled.
    -   **Clinical Validation:** All ML models and AI-generated suggestions would undergo rigorous clinical validation with large, diverse real-world datasets.
    -   **Ethical AI:** Continuous monitoring for algorithmic bias and ensuring fairness across different demographics would be paramount.
    -   **Integration:** A full solution would integrate with Electronic Health Records (EHR) systems and potentially other healthcare APIs.
    -   **Scalability:** For production, this would involve robust backend services, MLOps pipelines for model monitoring and retraining, and scalable cloud infrastructure.
    -   **API Key Management:** For the AI Medical Assistant, the API key is hardcoded for demo purposes. In a production environment, this must be managed securely (e.g., environment variables, secret management services).
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed as a comprehensive demonstration for an AI-powered healthcare solution.")

# --- Watermark ---
st.markdown("""
    <div style="color: #1BE687;" class="watermark">
        Developed by Manav Patel ❤️‍🩹
    </div>
    """, unsafe_allow_html=True)