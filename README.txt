# MediPredict AI v2 — Complete Setup Guide
# 37 Diseases | Login System | Patient Records | Doctor Booking

## Quick Start (3 steps)

### Step 1 — Install
pip install -r requirements.txt

### Step 2 — Train model (first time only)
python model/train.py

### Step 3 — Run
python app.py
Then open: http://localhost:5000

## Project Structure
medipredict2/
├── app.py                  ← Main Flask app
├── requirements.txt
├── database/
│   ├── db.py               ← SQLite database (users, consultations, bookings)
│   └── medipredict.db      ← Auto-created on first run
├── model/
│   ├── generate_data.py    ← Dataset generator
│   ├── train.py            ← Model training
│   ├── model.pkl           ← Trained model
│   ├── tfidf.pkl / scaler.pkl / label_encoder.pkl
│   └── model_meta.json
├── templates/
│   ├── base.html           ← Sidebar layout
│   ├── login.html          ← Login page
│   ├── register.html       ← Registration with medical info
│   ├── dashboard.html      ← Patient dashboard
│   ├── diagnose.html       ← Symptom analysis
│   ├── history.html        ← Consultation history
│   ├── booking.html        ← Doctor booking
│   ├── emergency.html      ← Emergency contacts
│   └── profile.html        ← Profile editor
└── static/css/main.css, static/js/main.js

## 37 Diseases
Cardiovascular: Coronary Artery Disease, Stroke, Hypertension, Heart Failure, Peripheral Artery Disease
Respiratory: COPD, Asthma, Pneumonia, Tuberculosis, Lung Cancer
Metabolic: Diabetes Type 1, Diabetes Type 2, Chronic Kidney Disease, Liver Cirrhosis, Obesity, Alzheimer's
Infectious: HIV/AIDS, Malaria, Dengue, Hepatitis B, Hepatitis C, Influenza, COVID-19, Cholera, Typhoid, Meningitis
Cancers: Breast, Colorectal, Prostate, Pancreatic, Cervical
General: Migraine, Anemia, Gastroenteritis, UTI, Depression, Appendicitis

## Pages
/ or /login     → Login
/register       → Create account
/dashboard      → Home dashboard
/diagnose       → AI symptom analysis
/history        → Patient consultation history (saved to SQLite)
/booking        → Doctor booking
/emergency      → Emergency contacts
/profile        → Edit profile, medical info
