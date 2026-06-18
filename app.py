"""
MediPredict AI v2 — Flask Application
37 diseases | Login | District-wise Doctor DB | Patient Records
Run: python app.py  →  http://localhost:5000
"""
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import joblib, json, numpy as np, os, uuid
from analyzer.report_analyzer import analyze_report
from scipy.sparse import hstack
from functools import wraps
import sys
sys.path.insert(0, os.path.dirname(__file__))
from database.db import (init_db, register_user, login_user, get_user,
                          save_consultation, get_consultations,
                          save_booking, get_bookings, update_profile)

app = Flask(__name__)
app.secret_key = "medipredict-v2-secret-2024"

# ── Load Model ───────────────────────────────────────────────────────────────
BASE_MODEL = os.path.join(os.path.dirname(__file__), "model")
model  = joblib.load(os.path.join(BASE_MODEL, "model.pkl"))
tfidf  = joblib.load(os.path.join(BASE_MODEL, "tfidf.pkl"))
scaler = joblib.load(os.path.join(BASE_MODEL, "scaler.pkl"))
le     = joblib.load(os.path.join(BASE_MODEL, "label_encoder.pkl"))
with open(os.path.join(BASE_MODEL, "model_meta.json")) as f:
    meta = json.load(f)
print(f"✅ Model loaded — {len(meta['classes'])} diseases | Accuracy: {meta['accuracy']*100:.0f}%")

# ── Load Doctor Dataset ───────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "india_doctors_dataset.json")
with open(DATA_PATH) as f:
    ALL_DOCTORS = json.load(f)

# Build state → districts map
DISTRICT_MAP = {}
STATES       = set()
SPECIALTIES  = set()
DISEASES_LIST= set()
for d in ALL_DOCTORS:
    STATES.add(d["state"])
    SPECIALTIES.add(d["specialty"])
    DISEASES_LIST.add(d["disease"])
    DISTRICT_MAP.setdefault(d["state"], set()).add(d["district"])
DISTRICT_MAP = {k: sorted(list(v)) for k, v in DISTRICT_MAP.items()}
print(f"✅ Doctors loaded — {len(ALL_DOCTORS)} doctors | {len(STATES)} states | {len(DISTRICT_MAP)} state entries")

# ── Severity / Emergency ──────────────────────────────────────────────────────
SEVERITY_MAP = {
    "Appendicitis":"critical","Stroke":"critical","Meningitis":"critical","Cholera":"critical",
    "Dengue Fever":"high","Pneumonia":"high","COVID-19":"high","Malaria":"high",
    "Heart Failure":"high","Coronary Artery Disease":"high","Lung Cancer":"high",
    "Breast Cancer":"high","Colorectal Cancer":"high","Prostate Cancer":"high",
    "Pancreatic Cancer":"high","Cervical Cancer":"high","Tuberculosis":"high","HIV/AIDS":"high",
    "Hypertension":"moderate","Diabetes Type 1":"moderate","Diabetes Type 2":"moderate",
    "Chronic Kidney Disease":"moderate","Liver Cirrhosis":"moderate","COPD":"moderate",
    "Asthma":"moderate","Hepatitis B":"moderate","Hepatitis C":"moderate",
    "Peripheral Artery Disease":"moderate","Typhoid Fever":"moderate","Obesity":"moderate",
    "Depression":"moderate","Anemia":"moderate","Alzheimer's Disease":"moderate",
    "Influenza":"low","Migraine":"low","Gastroenteritis":"low","UTI":"low",
}
EMERGENCY_KW = ["chest pain","can't breathe","cannot breathe","heart attack","stroke",
                 "unconscious","severe bleeding","seizure","convulsion","overdose",
                 "poisoning","choking","suicidal","paralysis","face drooping"]

DISEASE_CATEGORIES = {
    "❤️ Cardiovascular": ["Coronary Artery Disease","Stroke","Hypertension","Heart Failure","Peripheral Artery Disease"],
    "🌬️ Respiratory":    ["COPD","Asthma","Pneumonia","Tuberculosis","Lung Cancer"],
    "🍽️ Metabolic":      ["Diabetes Type 1","Diabetes Type 2","Chronic Kidney Disease","Liver Cirrhosis","Obesity","Alzheimer's Disease"],
    "🦠 Infectious":     ["HIV/AIDS","Malaria","Dengue Fever","Hepatitis B","Hepatitis C","Influenza","COVID-19","Cholera","Typhoid Fever","Meningitis"],
    "🎗️ Cancers":        ["Breast Cancer","Colorectal Cancer","Prostate Cancer","Pancreatic Cancer","Cervical Cancer"],
    "🩺 General":        ["Migraine","Anemia","Gastroenteritis","UTI","Depression","Appendicitis"],
}

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def predict(text, age, gender, fever, sys_bp, dia_bp, hr, duration):
    import pandas as pd
    X_text = tfidf.transform([text])
    COLS = ["age","fever_celsius","systolic_bp","diastolic_bp","heart_rate_bpm","duration_days","gender_enc"]
    struct = pd.DataFrame([[age, fever, sys_bp, dia_bp, hr, duration, int(gender=="Female")]], columns=COLS)
    X_s    = scaler.transform(struct)
    X      = hstack([X_text, X_s])
    probs  = model.predict_proba(X)[0]
    top_idx = np.argsort(probs)[::-1][:6]
    conditions = [{"name": le.classes_[i], "probability": round(float(probs[i])*100, 1)}
                  for i in top_idx if probs[i] > 0.005]
    primary  = conditions[0]["name"] if conditions else "Unknown"
    severity = SEVERITY_MAP.get(primary, "low")
    return primary, severity, conditions

def match_doctors_for_disease(primary):
    """Return top 4 doctors for the diagnosed disease."""
    matched = [d for d in ALL_DOCTORS if d["disease"] == primary]
    matched.sort(key=lambda x: -x["rating"])
    return matched[:4]

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("dashboard") if "user_id" in session else url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        user = login_user(request.form.get("email",""), request.form.get("password",""))
        if user:
            session["user_id"]   = user["id"]
            session["user_name"] = user["full_name"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        ok, msg = register_user(
            full_name=request.form.get("full_name",""), email=request.form.get("email",""),
            password=request.form.get("password",""), dob=request.form.get("dob",""),
            gender=request.form.get("gender",""), phone=request.form.get("phone",""),
            blood_group=request.form.get("blood_group",""), allergies=request.form.get("allergies",""),
        )
        flash(msg, "success" if ok else "error")
        if ok: return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))

# ── App Pages ──────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    user = get_user(session["user_id"])
    consultations = get_consultations(session["user_id"])
    bookings = get_bookings(session["user_id"])
    stats = {"total_consultations": len(consultations),
             "critical": sum(1 for c in consultations if c["severity"] in ["critical","high"]),
             "bookings": len(bookings),
             "last_visit": consultations[0]["date"] if consultations else "No visits yet"}
    return render_template("dashboard.html", user=user, stats=stats,
                           recent=consultations[:3], categories=DISEASE_CATEGORIES,
                           total_diseases=len(meta["classes"]))

@app.route("/diagnose")
@login_required
def diagnose():
    return render_template("diagnose.html")

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    d        = request.json
    text     = d.get("symptom_text","").strip()
    if not text:
        return jsonify({"error":"Please enter symptoms"}), 400

    # ── Llama Prompt Guard Validation ──
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        completion = client.chat.completions.create(
            model="meta-llama/llama-prompt-guard-2-86m",
            messages=[{"role": "user", "content": text}],
            temperature=1, max_completion_tokens=1, top_p=1, stream=False, stop=None
        )
        guard_res = completion.choices[0].message.content.strip().lower()
        if "unsafe" in guard_res:
            return jsonify({"error": "Unsafe or malicious input detected."}), 400
    except Exception as e:
        print(f"Prompt Guard Warning: {e}")
    age      = int(d.get("age",30)); gender = d.get("gender","Male")
    fever    = float(d.get("fever",37.0))
    sys_bp   = int(d.get("systolic_bp",120)); dia_bp = int(d.get("diastolic_bp",80))
    hr       = int(d.get("heart_rate",80)); duration = int(d.get("duration",2))

    has_scan   = bool(d.get("has_scan"))
    has_report = bool(d.get("has_report"))

    emergency_kw = [k for k in EMERGENCY_KW if k in text.lower()]
    primary, severity, conditions = predict(text, age, gender, fever, sys_bp, dia_bp, hr, duration)
    if emergency_kw: severity = "critical"

    recs = {"critical":["🚨 Call 112 immediately","Do not move the patient","Keep calm","Loosen clothing","Monitor breathing"],
             "high":    ["🏥 Emergency room today","Rest completely","Monitor closely","Take prescribed medication","Avoid exertion"],
             "moderate":["📅 Book doctor in 2–3 days","Rest and hydrate","Track daily symptoms","Avoid self-medication"],
             "low":     ["💊 Home care recommended","Stay hydrated","Monitor temperature","See doctor if worsening"]}.get(severity,[])

    if has_scan or has_report:
        recs = ["🔬 Scan/report was analyzed by AI — share with your doctor for confirmation"] + recs

    doctors = match_doctors_for_disease(primary)

    input_modes = []
    if text and "[SCAN FINDINGS]" not in text and "[REPORT FINDINGS]" not in text: input_modes.append("Symptoms")
    if has_scan:   input_modes.append("Scan")
    if has_report: input_modes.append("Lab Report")
    input_modes.append("Vitals")

    save_consultation(session["user_id"], {"symptom_text":text[:500],"primary_diagnosis":primary,"severity":severity,
        "conditions":conditions,"recommendations":recs,"age":age,"fever":fever,
        "systolic_bp":sys_bp,"diastolic_bp":dia_bp,"heart_rate":hr,"duration":duration})

    return jsonify({"primary_diagnosis":primary,"severity":severity,"conditions":conditions,
                    "recommendations":recs,"emergency_keywords":emergency_kw,"doctors":doctors,
                    "input_modes":input_modes,
                    "follow_up":{"critical":"Immediately","high":"Today","moderate":"Within 3 days","low":"If worsening"}.get(severity,"")})

@app.route("/history")
@login_required
def history():
    return render_template("history.html", records=get_consultations(session["user_id"]),
                           user=get_user(session["user_id"]))

@app.route("/api/history/clear", methods=["POST"])
@login_required
def clear_history():
    from database.db import get_conn
    conn = get_conn()
    conn.execute("DELETE FROM consultations WHERE user_id=?", (session["user_id"],))
    conn.commit(); conn.close()
    return jsonify({"ok": True})

@app.route("/booking")
@login_required
def booking():
    return render_template("booking.html",
                           doctors_json=json.dumps(ALL_DOCTORS),
                           district_map=json.dumps(DISTRICT_MAP),
                           states=list(STATES), specialties=list(SPECIALTIES),
                           diseases=list(DISEASES_LIST),
                           total_doctors=len(ALL_DOCTORS),
                           user=get_user(session["user_id"]),
                           my_bookings=get_bookings(session["user_id"]))

@app.route("/api/book", methods=["POST"])
@login_required
def api_book():
    import datetime
    d    = request.json
    ref  = f"MP{uuid.uuid4().hex[:6].upper()}"
    booking_data = {"ref":ref, "doctor":d.get("doctor_name","Doctor"),
                    "specialty":d.get("specialty",""), "hospital":d.get("hospital",""),
                    "slot":d.get("slot",""), "date":datetime.datetime.now().strftime("%d %B %Y"), "color":"#0EA5E9"}
    save_booking(session["user_id"], booking_data)
    return jsonify({"booking": booking_data})

@app.route("/emergency")
def emergency():
    return render_template("emergency.html")

@app.route("/profile", methods=["GET","POST"])
@login_required
def profile():
    user = get_user(session["user_id"])
    if request.method == "POST":
        update_profile(session["user_id"], {
            "full_name":request.form.get("full_name",""), "phone":request.form.get("phone",""),
            "blood_group":request.form.get("blood_group",""), "allergies":request.form.get("allergies",""),
            "dob":request.form.get("dob",""), "gender":request.form.get("gender",""),
        })
        session["user_name"] = request.form.get("full_name","")
        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))
    return render_template("profile.html", user=user,
                           consultation_count=len(get_consultations(session["user_id"])),
                           booking_count=len(get_bookings(session["user_id"])))


@app.route("/api/analyze-report", methods=["POST"])
@login_required
def api_analyze_report():
    if "report" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["report"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    file_bytes = f.read()
    result = analyze_report(file_bytes, f.filename, f.content_type or "")
    return jsonify(result)

# ── Chatbot Features ─────────────────────────────────────────────────────────
@app.route("/chat")
@login_required
def chat():
    from database.db import get_chat_history
    history = get_chat_history(session["user_id"], limit=50)
    return render_template("chatbot.html", user=get_user(session["user_id"]), history=json.dumps(history))

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    from chatbot_service import generate_chat_response
    user_message = request.form.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message is required."}), 400
        
    extracted_report = None
    if "report" in request.files:
        f = request.files["report"]
        if f.filename:
            file_bytes = f.read()
            result = analyze_report(file_bytes, f.filename, f.content_type or "")
            if "raw_text" in result and result["raw_text"]:
                extracted_report = result["raw_text"] + "\n\nML Insights: " + result.get("symptom_summary", "")
                
    # Generate reply
    response_text = generate_chat_response(session["user_id"], user_message, extracted_report)
    return jsonify({"response": response_text})

if __name__ == "__main__":
    init_db()
    print(f"\n🏥 MediPredict AI v2 — {len(meta['classes'])} diseases | {len(ALL_DOCTORS)} doctors")
    print("📍 Open http://localhost:5000\n")
    app.run(debug=True, port=5000)
