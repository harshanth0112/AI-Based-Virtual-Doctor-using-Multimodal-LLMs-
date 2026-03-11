"""
Multimodal Report Analyzer
Handles: X-ray images, lab reports (PDF/image), doctor prescriptions, scan reports
Extracts text via OCR + identifies key medical values
"""
import os, re, base64, io, json
from PIL import Image
import pytesseract
import pdfplumber

# ── Medical value patterns ────────────────────────────────────────────────────
LAB_PATTERNS = {
    "hemoglobin":      r"h[ae]moglobin[:\s]+(\d+\.?\d*)",
    "wbc":             r"(?:wbc|white blood cell|leukocyte)[:\s]+(\d+\.?\d*)",
    "rbc":             r"(?:rbc|red blood cell|erythrocyte)[:\s]+(\d+\.?\d*)",
    "platelets":       r"platelet[:\s]+(\d+\.?\d*)",
    "glucose":         r"(?:glucose|blood sugar|fbs|rbs)[:\s]+(\d+\.?\d*)",
    "hba1c":           r"hba1c[:\s]+(\d+\.?\d*)",
    "creatinine":      r"creatinine[:\s]+(\d+\.?\d*)",
    "urea":            r"(?:urea|bun)[:\s]+(\d+\.?\d*)",
    "sodium":          r"sodium[:\s]+(\d+\.?\d*)",
    "potassium":       r"potassium[:\s]+(\d+\.?\d*)",
    "cholesterol":     r"(?:total\s)?cholesterol[:\s]+(\d+\.?\d*)",
    "ldl":             r"ldl[:\s]+(\d+\.?\d*)",
    "hdl":             r"hdl[:\s]+(\d+\.?\d*)",
    "triglycerides":   r"triglyceride[s]?[:\s]+(\d+\.?\d*)",
    "bilirubin":       r"bilirubin[:\s]+(\d+\.?\d*)",
    "alt":             r"(?:alt|sgpt)[:\s]+(\d+\.?\d*)",
    "ast":             r"(?:ast|sgot)[:\s]+(\d+\.?\d*)",
    "tsh":             r"tsh[:\s]+(\d+\.?\d*)",
    "t3":              r"\bt3\b[:\s]+(\d+\.?\d*)",
    "t4":              r"\bt4\b[:\s]+(\d+\.?\d*)",
    "uric_acid":       r"uric acid[:\s]+(\d+\.?\d*)",
    "esr":             r"esr[:\s]+(\d+\.?\d*)",
    "crp":             r"crp[:\s]+(\d+\.?\d*)",
    "systolic_bp":     r"(?:systolic|sys)[:\s]+(\d+)",
    "diastolic_bp":    r"(?:diastolic|dia)[:\s]+(\d+)",
    "heart_rate":      r"(?:heart rate|pulse|hr)[:\s]+(\d+)",
    "spo2":            r"(?:spo2|oxygen saturation|o2 sat)[:\s]+(\d+\.?\d*)",
    "temperature":     r"(?:temperature|temp)[:\s]+(\d+\.?\d*)",
    "weight":          r"weight[:\s]+(\d+\.?\d*)",
    "bmi":             r"bmi[:\s]+(\d+\.?\d*)",
}

# Abnormal value thresholds for flagging
ABNORMAL_THRESHOLDS = {
    "hemoglobin":    {"low": 12.0, "high": 17.5, "unit": "g/dL"},
    "glucose":       {"low": 70,   "high": 126,   "unit": "mg/dL"},
    "hba1c":         {"low": 0,    "high": 6.5,   "unit": "%"},
    "creatinine":    {"low": 0,    "high": 1.2,   "unit": "mg/dL"},
    "cholesterol":   {"low": 0,    "high": 200,   "unit": "mg/dL"},
    "ldl":           {"low": 0,    "high": 130,   "unit": "mg/dL"},
    "hdl":           {"low": 40,   "high": 999,   "unit": "mg/dL"},
    "triglycerides": {"low": 0,    "high": 150,   "unit": "mg/dL"},
    "bilirubin":     {"low": 0,    "high": 1.2,   "unit": "mg/dL"},
    "alt":           {"low": 0,    "high": 40,    "unit": "U/L"},
    "ast":           {"low": 0,    "high": 40,    "unit": "U/L"},
    "tsh":           {"low": 0.4,  "high": 4.0,   "unit": "mIU/L"},
    "spo2":          {"low": 95,   "high": 100,   "unit": "%"},
    "systolic_bp":   {"low": 90,   "high": 140,   "unit": "mmHg"},
    "diastolic_bp":  {"low": 60,   "high": 90,    "unit": "mmHg"},
    "platelets":     {"low": 150,  "high": 400,   "unit": "x10³/µL"},
    "wbc":           {"low": 4.5,  "high": 11.0,  "unit": "x10³/µL"},
}

# Medical keywords that suggest diseases
DISEASE_KEYWORDS = {
    "diabetes":     ["hyperglycemia","hba1c","fasting glucose","insulin","diabetes","diabetic","glucometer"],
    "cardiac":      ["troponin","ecg","ekg","st elevation","angina","myocardial","ischemia","ejection fraction","lvef"],
    "respiratory":  ["spirometry","fev1","fvc","copd","wheezing","consolidation","infiltrate","pleural","pneumonia"],
    "liver":        ["hepatitis","cirrhosis","jaundice","bilirubin elevated","alt elevated","ast elevated","fatty liver"],
    "kidney":       ["creatinine elevated","egfr","proteinuria","hematuria","renal","nephro"],
    "thyroid":      ["hypothyroid","hyperthyroid","tsh","goiter","thyroid nodule"],
    "anemia":       ["hemoglobin low","iron deficiency","microcytic","normocytic","ferritin","low hb"],
    "infection":    ["wbc elevated","neutrophilia","positive culture","bacteria","virus","sepsis","fever"],
    "cancer":       ["malignant","tumor","carcinoma","biopsy","metastasis","oncology","staging"],
    "dengue":       ["ns1","dengue","platelet low","thrombocytopenia"],
}

def extract_text_from_image(image_bytes):
    """OCR from image bytes."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if needed
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        # Enhance for OCR
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        text = pytesseract.image_to_string(img, config='--psm 6')
        return text.strip()
    except Exception as e:
        return f"[Image OCR failed: {str(e)}]"

def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF bytes."""
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
                # Also extract tables
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            text_parts.append(" | ".join([str(c) for c in row if c]))
        return "\n".join(text_parts).strip()
    except Exception as e:
        return f"[PDF extraction failed: {str(e)}]"

def extract_lab_values(text):
    """Extract numeric lab values from OCR text."""
    text_lower = text.lower()
    found = {}
    for key, pattern in LAB_PATTERNS.items():
        match = re.search(pattern, text_lower)
        if match:
            try:
                found[key] = float(match.group(1))
            except:
                pass
    return found

def flag_abnormal_values(lab_values):
    """Flag values outside normal range."""
    flags = []
    for key, val in lab_values.items():
        if key in ABNORMAL_THRESHOLDS:
            t = ABNORMAL_THRESHOLDS[key]
            if val < t["low"]:
                flags.append({"parameter": key.replace("_"," ").title(),
                               "value": f"{val} {t['unit']}", "status": "LOW",
                               "normal": f"{t['low']}–{t['high']} {t['unit']}"})
            elif val > t["high"]:
                flags.append({"parameter": key.replace("_"," ").title(),
                               "value": f"{val} {t['unit']}", "status": "HIGH",
                               "normal": f"{t['low']}–{t['high']} {t['unit']}"})
    return flags

def detect_report_type(text):
    """Detect what kind of medical document this is."""
    text_lower = text.lower()
    if any(k in text_lower for k in ["x-ray","xray","radiograph","chest pa","lateral view","opacity","consolidation"]):
        return "X-Ray Report"
    if any(k in text_lower for k in ["ecg","ekg","electrocardiogram","rhythm","qt interval","p wave","qrs"]):
        return "ECG Report"
    if any(k in text_lower for k in ["mri","magnetic resonance","t1","t2","flair","lesion","signal intensity"]):
        return "MRI Report"
    if any(k in text_lower for k in ["ct scan","computed tomography","axial","coronal","hounsfield"]):
        return "CT Scan Report"
    if any(k in text_lower for k in ["ultrasound","usg","sonography","echogenicity","gallbladder","liver size"]):
        return "Ultrasound Report"
    if any(k in text_lower for k in ["hemoglobin","wbc","rbc","platelet","hematocrit","blood count","cbc"]):
        return "Complete Blood Count (CBC)"
    if any(k in text_lower for k in ["glucose","hba1c","insulin","lipid","cholesterol","triglyceride"]):
        return "Biochemistry / Metabolic Panel"
    if any(k in text_lower for k in ["urine","protein","ketone","bilirubin","urinalysis","microscopy"]):
        return "Urine Analysis"
    if any(k in text_lower for k in ["prescription","tablet","capsule","mg","ml","dose","syrup","injection","rx"]):
        return "Doctor Prescription"
    if any(k in text_lower for k in ["discharge summary","admitted","diagnosis","treatment given","follow up"]):
        return "Discharge Summary"
    if any(k in text_lower for k in ["biopsy","histopathology","pathology","specimen","tissue","malignant"]):
        return "Pathology Report"
    return "Medical Document"

def detect_disease_hints(text):
    """Find disease-related keywords in the report."""
    text_lower = text.lower()
    hints = []
    for disease_group, keywords in DISEASE_KEYWORDS.items():
        matches = [k for k in keywords if k in text_lower]
        if matches:
            hints.append({"category": disease_group, "keywords_found": matches})
    return hints

def build_symptom_summary(text, lab_values, abnormal_flags, disease_hints):
    """Build a natural language summary of findings to feed into ML model."""
    parts = []

    # Abnormal flags
    for flag in abnormal_flags:
        param = flag["parameter"].lower()
        status = flag["status"].lower()
        if "glucose" in param:
            parts.append("elevated blood sugar" if status=="high" else "low blood sugar")
        elif "hemoglobin" in param:
            parts.append("low hemoglobin anemia fatigue" if status=="low" else "high hemoglobin")
        elif "creatinine" in param:
            parts.append("elevated creatinine kidney dysfunction")
        elif "alt" in param or "ast" in param:
            parts.append("elevated liver enzymes liver damage")
        elif "cholesterol" in param:
            parts.append("high cholesterol hyperlipidemia")
        elif "ldl" in param:
            parts.append("high ldl cardiovascular risk")
        elif "tsh" in param:
            if status == "high": parts.append("high tsh hypothyroidism fatigue weight gain")
            else: parts.append("low tsh hyperthyroidism palpitations")
        elif "platelet" in param:
            parts.append("low platelet thrombocytopenia dengue" if status=="low" else "high platelet")
        elif "wbc" in param:
            parts.append("high white blood cell infection inflammation" if status=="high" else "low wbc immune deficiency")
        elif "bilirubin" in param:
            parts.append("elevated bilirubin jaundice liver disease")
        elif "spo2" in param:
            parts.append("low oxygen saturation breathing difficulty respiratory distress")
        elif "systolic" in param:
            parts.append("high blood pressure hypertension")
        elif "triglycerides" in param:
            parts.append("high triglycerides metabolic syndrome")
        elif "hba1c" in param:
            parts.append("elevated hba1c diabetes poor glucose control")

    # Disease hints from text
    for hint in disease_hints:
        cat = hint["category"]
        if cat == "cardiac": parts.append("chest pain cardiac symptoms ecg changes")
        elif cat == "respiratory": parts.append("breathing difficulty lung function impaired")
        elif cat == "liver": parts.append("liver disease jaundice nausea")
        elif cat == "kidney": parts.append("kidney function impaired fatigue swelling")
        elif cat == "thyroid": parts.append("thyroid dysfunction fatigue weight changes")
        elif cat == "anemia": parts.append("anemia fatigue weakness pale skin")
        elif cat == "infection": parts.append("infection fever elevated white cells")
        elif cat == "diabetes": parts.append("diabetes blood sugar elevated thirst fatigue")
        elif cat == "dengue": parts.append("dengue fever low platelets rash body pain")
        elif cat == "cancer": parts.append("abnormal growth malignancy tumor findings")

    return " ".join(parts) if parts else ""

def analyze_report(file_bytes, filename, mime_type):
    """
    Main entry point. Returns dict with:
    - report_type, extracted_text, lab_values, abnormal_flags,
      disease_hints, symptom_summary, vitals_override
    """
    filename_lower = filename.lower()

    # Extract text
    if mime_type == "application/pdf" or filename_lower.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_bytes)
    elif mime_type.startswith("image/") or any(filename_lower.endswith(x) for x in [".jpg",".jpeg",".png",".bmp",".tiff",".webp"]):
        raw_text = extract_text_from_image(file_bytes)
    elif filename_lower.endswith(".txt"):
        raw_text = file_bytes.decode("utf-8", errors="ignore")
    else:
        raw_text = extract_text_from_image(file_bytes)  # Try OCR as fallback

    if not raw_text or len(raw_text) < 20:
        return {"error": "Could not extract readable text from this file. Please ensure it is a clear image or text-based PDF.", "raw_text": ""}

    # Analyze
    report_type    = detect_report_type(raw_text)
    lab_values     = extract_lab_values(raw_text)
    abnormal_flags = flag_abnormal_values(lab_values)
    disease_hints  = detect_disease_hints(raw_text)
    symptom_summary = build_symptom_summary(raw_text, lab_values, abnormal_flags, disease_hints)

    # Extract vitals for override
    vitals_override = {}
    if "systolic_bp"  in lab_values: vitals_override["systolic_bp"]  = int(lab_values["systolic_bp"])
    if "diastolic_bp" in lab_values: vitals_override["diastolic_bp"] = int(lab_values["diastolic_bp"])
    if "heart_rate"   in lab_values: vitals_override["heart_rate"]   = int(lab_values["heart_rate"])
    if "temperature"  in lab_values: vitals_override["fever"]        = float(lab_values["temperature"])
    if "spo2"         in lab_values: vitals_override["spo2"]         = float(lab_values["spo2"])

    return {
        "report_type":     report_type,
        "raw_text":        raw_text[:2000],   # truncate for display
        "lab_values":      lab_values,
        "abnormal_flags":  abnormal_flags,
        "disease_hints":   disease_hints,
        "symptom_summary": symptom_summary,
        "vitals_override": vitals_override,
        "has_findings":    len(abnormal_flags) > 0 or len(disease_hints) > 0,
    }
