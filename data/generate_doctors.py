"""
India District-Wise Doctor Dataset Generator
- 37 diseases
- 50 doctors per disease
- Spread across all major Indian districts/states
- Includes: name, hospital, district, state, specialty, experience, degree, rating, reviews, contact
"""

import pandas as pd
import numpy as np
import json
import random
import os

random.seed(42)
np.random.seed(42)

# ── Indian Districts by State ─────────────────────────────────────────────────
INDIA_DISTRICTS = {
    "Tamil Nadu": [
        "Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem",
        "Tirunelveli", "Erode", "Vellore", "Thoothukudi", "Dindigul",
        "Thanjavur", "Ranipet", "Sivaganga", "Virudhunagar", "Namakkal",
        "Karur", "Nagapattinam", "Tiruppur", "Kancheepuram", "Krishnagiri"
    ],
    "Maharashtra": [
        "Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad",
        "Solapur", "Kolhapur", "Amravati", "Nanded", "Sangli",
        "Satara", "Jalgaon", "Akola", "Latur", "Dhule",
        "Ahmednagar", "Chandrapur", "Parbhani", "Ratnagiri", "Wardha"
    ],
    "Karnataka": [
        "Bangalore", "Mysore", "Hubli", "Mangalore", "Belgaum",
        "Gulbarga", "Davanagere", "Bellary", "Bijapur", "Shimoga",
        "Tumkur", "Raichur", "Bidar", "Hassan", "Udupi",
        "Chitradurga", "Kolar", "Mandya", "Chikmagalur", "Dharwad"
    ],
    "Uttar Pradesh": [
        "Lucknow", "Kanpur", "Varanasi", "Agra", "Prayagraj",
        "Meerut", "Ghaziabad", "Aligarh", "Moradabad", "Bareilly",
        "Saharanpur", "Gorakhpur", "Firozabad", "Jhansi", "Mathura",
        "Muzaffarnagar", "Shahjahanpur", "Rampur", "Hardoi", "Etawah"
    ],
    "Gujarat": [
        "Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar",
        "Jamnagar", "Gandhinagar", "Junagadh", "Anand", "Navsari",
        "Bharuch", "Amreli", "Patan", "Mehsana", "Surendranagar",
        "Botad", "Morbi", "Kheda", "Narmada", "Porbandar"
    ],
    "Rajasthan": [
        "Jaipur", "Jodhpur", "Kota", "Bikaner", "Ajmer",
        "Udaipur", "Bhilwara", "Alwar", "Bharatpur", "Sikar",
        "Pali", "Sri Ganganagar", "Tonk", "Sawai Madhopur", "Nagaur",
        "Chittorgarh", "Jhunjhunu", "Barmer", "Hanumangarh", "Dausa"
    ],
    "West Bengal": [
        "Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri",
        "Malda", "Bardhaman", "Kharagpur", "Haldia", "Darjeeling",
        "Murshidabad", "Nadia", "North 24 Parganas", "South 24 Parganas", "Hooghly",
        "Bankura", "Purulia", "Jalpaiguri", "Cooch Behar", "Midnapore"
    ],
    "Andhra Pradesh": [
        "Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Kurnool",
        "Kadapa", "Tirupati", "Rajahmundry", "Kakinada", "Anantapur",
        "Ongole", "Eluru", "Machilipatnam", "Chittoor", "Srikakulam",
        "Vizianagaram", "West Godavari", "East Godavari", "Krishna", "Prakasam"
    ],
    "Telangana": [
        "Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam",
        "Ramagundam", "Mahbubnagar", "Nalgonda", "Adilabad", "Suryapet",
        "Mancherial", "Jagtial", "Sangareddy", "Siddipet", "Medak",
        "Vikarabad", "Wanaparthy", "Narayanpet", "Jogulamba", "Bhadradri"
    ],
    "Kerala": [
        "Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam",
        "Kannur", "Alappuzha", "Palakkad", "Kottayam", "Malappuram",
        "Idukki", "Pathanamthitta", "Wayanad", "Kasaragod", "Ernakulam"
    ],
    "Delhi": [
        "New Delhi", "Central Delhi", "North Delhi", "South Delhi", "East Delhi",
        "West Delhi", "North East Delhi", "North West Delhi", "South East Delhi", "South West Delhi"
    ],
    "Punjab": [
        "Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda",
        "Hoshiarpur", "Mohali", "Firozpur", "Gurdaspur", "Ropar",
        "Fatehgarh Sahib", "Moga", "Sangrur", "Barnala", "Faridkot"
    ],
    "Haryana": [
        "Faridabad", "Gurgaon", "Panipat", "Ambala", "Yamunanagar",
        "Rohtak", "Hisar", "Karnal", "Sonipat", "Panchkula",
        "Bhiwani", "Sirsa", "Jhajjar", "Rewari", "Mewat"
    ],
    "Madhya Pradesh": [
        "Bhopal", "Indore", "Jabalpur", "Gwalior", "Ujjain",
        "Sagar", "Dewas", "Satna", "Ratlam", "Rewa",
        "Katni", "Singrauli", "Burhanpur", "Khandwa", "Bhind",
        "Morena", "Chhindwara", "Guna", "Shivpuri", "Vidisha"
    ],
    "Bihar": [
        "Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Purnia",
        "Darbhanga", "Ara", "Begusarai", "Katihar", "Munger",
        "Chapra", "Hajipur", "Dehri", "Siwan", "Motihari",
        "Nawada", "Buxar", "Sitamarhi", "Vaishali", "Nalanda"
    ],
    "Odisha": [
        "Bhubaneswar", "Cuttack", "Rourkela", "Brahmapur", "Sambalpur",
        "Puri", "Balasore", "Bhadrak", "Baripada", "Jharsuguda",
        "Koraput", "Mayurbhanj", "Kendrapara", "Jajpur", "Khordha"
    ],
    "Assam": [
        "Guwahati", "Silchar", "Dibrugarh", "Jorhat", "Nagaon",
        "Tinsukia", "Tezpur", "Karimganj", "Hailakandi", "Cachar",
        "Barpeta", "Nalbari", "Kamrup", "Sonitpur", "Dhubri"
    ],
    "Jharkhand": [
        "Ranchi", "Jamshedpur", "Dhanbad", "Bokaro", "Deoghar",
        "Phusro", "Hazaribagh", "Giridih", "Ramgarh", "Medininagar",
        "Chatra", "Gumla", "Simdega", "Pakur", "Dumka"
    ],
    "Himachal Pradesh": [
        "Shimla", "Manali", "Dharamsala", "Solan", "Mandi",
        "Kullu", "Hamirpur", "Una", "Bilaspur", "Chamba"
    ],
    "Uttarakhand": [
        "Dehradun", "Haridwar", "Roorkee", "Haldwani", "Rudrapur",
        "Kashipur", "Rishikesh", "Kotdwar", "Ramnagar", "Pithoragarh"
    ],
}

# ── Doctor Name Components ─────────────────────────────────────────────────────
FIRST_NAMES_MALE = [
    "Rajesh","Suresh","Ramesh","Mahesh","Ganesh","Dinesh","Naresh","Umesh","Lokesh","Mukesh",
    "Anil","Sunil","Patel","Vijay","Sanjay","Ajay","Uday","Manoj","Rakesh","Deepak",
    "Ashok","Vinod","Pramod","Harish","Girish","Manish","Satish","Ritesh","Nilesh","Paresh",
    "Arjun","Rahul","Rohan","Kiran","Varun","Tarun","Arun","Vikas","Nikhil","Akhil",
    "Pradeep","Sandeep","Kuldeep","Pardeep","Navdeep","Gurdeep","Amitabh","Amit","Sumit","Rohit",
    "Vivek","Alok","Trilok","Ashish","Sachin","Navin","Praveen","Naveen","Jayesh","Alpesh",
    "Ravi","Kavi","Davi","Rajan","Mohan","Sohan","Johan","Krishnan","Murugan","Selvam",
    "Senthil","Karthi","Balamurugan","Sivakumar","Ramakrishnan","Venkatesan","Subramanian","Pandian","Muthukumar","Chellapan",
    "Abdul","Mohammed","Imran","Farhan","Salman","Irfan","Rizwan","Azhar","Tariq","Bilal",
    "Pawan","Bhawan","Chawan","Gopal","Murali","Balaji","Subramaniam","Narayanan","Thyagarajan","Parthasarathy",
]

FIRST_NAMES_FEMALE = [
    "Priya","Kavya","Divya","Shreya","Freya","Vidya","Nidhi","Riddhi","Siddhi","Vriddhi",
    "Anitha","Sunitha","Savitha","Kavitha","Kaveri","Gayatri","Saraswati","Lakshmi","Sarojini","Kamakshi",
    "Meena","Seena","Leena","Reena","Veena","Sheena","Meera","Geera","Neera","Heera",
    "Anjali","Shobha","Rekha","Radha","Sudha","Madhu","Sindhu","Bindu","Indu","Rendu",
    "Padma","Padmini","Kamini","Yamini","Rohini","Mohini","Sohini","Bhavani","Shivani","Parvani",
    "Sangeetha","Sumathi","Vijayalakshmi","Alamelu","Thilagavathi","Saranya","Nithya","Nandhini","Kiruthiga","Pavithra",
    "Deepa","Deepika","Monika","Sonika","Veronica","Veronika","Ambika","Ahilya","Sailaja","Sharada",
    "Fatima","Ayesha","Zara","Sara","Nadia","Razia","Sana","Hina","Rina","Tina",
    "Ananya","Tananya","Mananya","Vananya","Sananya","Rananya","Pananya","Kananya","Hananya","Dananya",
    "Swathi","Sruthi","Srishti","Shristi","Smrithi","Smriti","Spurthi","Spurti","Sukirtha","Sukirthi",
]

LAST_NAMES = [
    "Sharma","Verma","Gupta","Singh","Kumar","Patel","Shah","Mehta","Joshi","Trivedi",
    "Reddy","Rao","Naidu","Raju","Babu","Murthy","Prasad","Nair","Pillai","Menon",
    "Iyer","Iyengar","Krishnamurthy","Venkataraman","Subramanian","Swaminathan","Raghunathan","Parthasarathy","Thyagarajan","Balakrishnan",
    "Chatterjee","Banerjee","Mukherjee","Ghosh","Bose","Das","Datta","Sen","Roy","Chowdhury",
    "Mishra","Pandey","Tiwari","Dwivedi","Shukla","Tripathi","Upadhyay","Chaturvedi","Srivastava","Agarwal",
    "Desai","Jain","Bhatt","Pandya","Raval","Chauhan","Thakur","Parmar","Solanki","Rathod",
    "Kulkarni","Patil","Jadhav","Shinde","Bhosale","Kadam","Pawar","More","Gaikwad","Deshmukh",
    "Bhat","Kamath","Shetty","Hegde","Gowda","Naik","Amin","Deshpande","Joshi","Kale",
    "Khan","Ansari","Siddiqui","Qureshi","Malik","Sheikh","Mirza","Hussain","Ali","Ahmad",
    "Rajan","Rajendran","Rajasekar","Rajagopalan","Rajagopal","Rajavel","Rajavelu","Rajkumar","Rajamani","Rajanand",
]

# ── Degrees by Specialty ──────────────────────────────────────────────────────
DEGREES = {
    "Cardiologist":         ["MBBS, MD (Medicine), DM (Cardiology)","MBBS, MS, MCh (Cardiothoracic Surgery)","MBBS, MD, DNB (Cardiology)","MBBS, MD, FNB (Cardiology)","MBBS, MRCP (UK), DM (Cardiology)"],
    "Neurologist":          ["MBBS, MD (Medicine), DM (Neurology)","MBBS, MD, DNB (Neurology)","MBBS, MRCP, DM (Neurology)","MBBS, MS, MCh (Neurosurgery)","MBBS, MD, FNB (Neurology)"],
    "Pulmonologist":        ["MBBS, MD (Pulmonary Medicine)","MBBS, MD (Respiratory Medicine)","MBBS, MD, DNB (Pulmonology)","MBBS, MD (Internal Medicine), Fellowship Pulmonology","MBBS, DTCD, MD (Pulmonary)"],
    "Endocrinologist":      ["MBBS, MD (Medicine), DM (Endocrinology)","MBBS, MD, DNB (Endocrinology)","MBBS, MRCP (UK), DM (Endocrinology)","MBBS, MD (Endocrinology & Metabolism)","MBBS, MD, FNB (Endocrinology)"],
    "Nephrologist":         ["MBBS, MD (Medicine), DM (Nephrology)","MBBS, MD, DNB (Nephrology)","MBBS, MRCP, DM (Nephrology)","MBBS, MD, FNB (Nephrology)","MBBS, MS, MCh (Urology & Nephrology)"],
    "Hepatologist":         ["MBBS, MD (Medicine), DM (Hepatology)","MBBS, MD, DNB (Gastroenterology)","MBBS, MD (Gastroenterology & Hepatology)","MBBS, DM (Hepatology)","MBBS, MRCP, DM (Gastroenterology)"],
    "Infectious Disease":   ["MBBS, MD (Internal Medicine), Fellowship ID","MBBS, MD (Microbiology), Fellowship ID","MBBS, MD (Community Medicine), Fellowship ID","MBBS, DTM&H, MD (Infectious Disease)","MBBS, MD, DNB (Infectious Disease)"],
    "Oncologist":           ["MBBS, MD (Radiation Oncology)","MBBS, MD (Medical Oncology)","MBBS, MS, MCh (Surgical Oncology)","MBBS, MD, DM (Medical Oncology)","MBBS, DNB (Oncology), MRCP"],
    "General Physician":    ["MBBS, MD (General Medicine)","MBBS, MD (Internal Medicine)","MBBS, DNB (General Medicine)","MBBS, MRCP (UK)","MBBS, MD, FRCGP"],
    "Psychiatrist":         ["MBBS, MD (Psychiatry)","MBBS, DPM (Psychiatry)","MBBS, MD, DNB (Psychiatry)","MBBS, MRCPsych (UK)","MBBS, MD (Psychiatry & Behavioural Sciences)"],
    "Gastroenterologist":   ["MBBS, MD (Medicine), DM (Gastroenterology)","MBBS, MD, DNB (Gastroenterology)","MBBS, MRCP, DM (Gastroenterology)","MBBS, MD, FNB (Gastroenterology)","MBBS, MS (General Surgery), MCh (GI Surgery)"],
    "Geriatrician":         ["MBBS, MD (Geriatrics)","MBBS, MD (Internal Medicine), Fellowship Geriatrics","MBBS, MRCP, MSc (Geriatrics)","MBBS, MD (Medicine), Diploma Geriatrics","MBBS, DNB (Geriatrics)"],
    "Vascular Surgeon":     ["MBBS, MS (General Surgery), MCh (Vascular Surgery)","MBBS, MS, DNB (Vascular Surgery)","MBBS, MS (Surgery), Fellowship Vascular","MBBS, MCh (Cardiovascular & Thoracic)","MBBS, MS, FRCS (Vascular)"],
    "Haematologist":        ["MBBS, MD (Medicine), DM (Haematology)","MBBS, MD, DNB (Haematology)","MBBS, MD (Pathology), DM (Haematology)","MBBS, MRCP, DM (Haematology)","MBBS, MD, FNB (Haematology)"],
    "Gynaecologist":        ["MBBS, MS (Obstetrics & Gynaecology)","MBBS, MD (Obstetrics & Gynaecology)","MBBS, DGO, MS (Gynaecology)","MBBS, MS, DNB (Gynaecology & Oncology)","MBBS, MS, FRCOG (UK)"],
    "Dermatologist":        ["MBBS, MD (Dermatology)","MBBS, DVD, MD (Dermatology)","MBBS, MD, DNB (Dermatology)","MBBS, MRCP (Derm), MD","MBBS, MD (Dermatology, Venereology & Leprosy)"],
    "Urologist":            ["MBBS, MS (General Surgery), MCh (Urology)","MBBS, MS, DNB (Urology)","MBBS, MS, FRCS (Urology)","MBBS, MCh (Urology), Fellowship Uro-Oncology","MBBS, MS, MCh (Urology & Transplant)"],
    "Orthopaedic Surgeon":  ["MBBS, MS (Orthopaedics)","MBBS, MS, DNB (Orthopaedics)","MBBS, MS, FRCS (Orthopaedics)","MBBS, MS (Ortho), Fellowship Joint Replacement","MBBS, MS, MCh (Spine Surgery)"],
}

# ── Hospitals by City ─────────────────────────────────────────────────────────
HOSPITALS = {
    "Chennai":          ["Apollo Hospitals","MIOT International","Fortis Malar","Sri Ramachandra","Kauvery Hospital","MGM Healthcare","Billroth Hospitals","Global Hospital","Gleneagles Global","Vijaya Hospital"],
    "Coimbatore":       ["PSG Hospitals","KG Hospital","Sri Ramakrishna Hospital","Kovai Medical Center","GKNM Hospital","CHL Hospital","Lotus Eye Hospital","Aravind Eye","Appasamy","Dwaraka Das"],
    "Madurai":          ["Meenakshi Mission","Apollo Spectra","Government Rajaji","Vadamalayan Hospital","SIMS Hospital","Vasan Health","Sooriya Hospital","Ruby Hall","Care Hospitals","Velammal"],
    "Tiruchirappalli":  ["Kauvery Hospital","Ponni Hospitals","Trichy SRM","Government Hospital","Apollo Clinic","Dr.Kamakshi","Srinivasa Hospital","Priyadarshini","Vinayaka Missions","Balaji"],
    "Mumbai":           ["Lilavati Hospital","Hinduja Hospital","Nanavati Hospital","Kokilaben Dhirubhai","Tata Memorial","KEM Hospital","Breach Candy","Wockhardt Hospital","Asian Heart","Bombay Hospital"],
    "Pune":             ["Ruby Hall Clinic","Jehangir Hospital","Sahyadri Hospital","Noble Hospital","Deenanath Mangeshkar","Columbia Asia","KEM Hospital Pune","Inlaks Hospital","Poona Hospital","Aditya Birla"],
    "Nagpur":           ["Wockhardt Nagpur","Orange City Hospital","Alexis Hospital","Care Hospital","Lata Mangeshkar","CIMS Nagpur","Central India Institute","Getwell Hospital","Kingsway Hospital","Tata Hospital Nagpur"],
    "Bangalore":        ["Manipal Hospital","Apollo Bangalore","Fortis Hospital","Narayana Health","BGS Global","Aster CMI","Columbia Asia Bangalore","MS Ramaiah","St.John's Medical","Sri Siddhartha"],
    "Hyderabad":        ["Apollo Jubilee Hills","Yashoda Hospital","KIMS Hospital","Continental Hospital","AIG Hospital","Sunshine Hospital","Care Hospital Hyderabad","Maxcure Hospital","Medicover","Gleneagles Global Hyd"],
    "Kolkata":          ["Fortis Anandapur","Apollo Gleneagles","AMRI Hospital","Peerless Hospital","Medica Superspecialty","Belle Vue Clinic","Woodlands Hospital","RN Tagore","CMRI Kolkata","NRS Medical"],
    "Delhi":            ["AIIMS New Delhi","Max Super Speciality","Apollo Delhi","Fortis Vasant Kunj","BLK-MAX Hospital","Sir Ganga Ram","Safdarjung Hospital","Indraprastha Apollo","Medanta Delhi","Rockland Hospital"],
    "Lucknow":          ["Sahara Hospital","Apollo Medics","Medanta Lucknow","Ram Manohar Lohia","Balrampur Hospital","Charak Hospital","Vivekananda Polyclinic","Healthcity Hospital","Panacea Hospital","Shekhar Hospital"],
    "Ahmedabad":        ["Apollo Ahmedabad","Zydus Hospital","Sterling Hospital","HCG Hospital","UN Mehta Heart","SAL Hospital","KD Hospital","Shalby Hospital","Apollo Spectra Ahmedabad","Karma Hospital"],
    "Jaipur":           ["Fortis Jaipur","Narayana Jaipur","Apollo Jaipur","Eternal Hospital","Mahatma Gandhi Hospital","SMS Hospital","Santokba Durlabhji","Rukmani Birla","CK Birla","Medipulse"],
    "Chandigarh":       ["PGI Chandigarh","Fortis Mohali","Max Hospital Mohali","GMSH Sector 16","IVY Hospital","Alchemist Hospital","Grecian Hospital","Paras Hospital","SPS Hospital","Healing Hospital"],
    "Visakhapatnam":    ["Apollo Vizag","Care Hospital Vizag","Seven Hills Hospital","KIMS Vizag","Andhra Medical College","Sunshine Vizag","Narayana Vizag","Columbia Asia Vizag","Manipal Vizag","Aditya Hospital"],
    "Kochi":            ["Amrita Institute","Aster Medcity","Lakeshore Hospital","KIMS Kochi","PVS Hospital","Baby Memorial","Lisie Hospital","Medical Trust","Renai Medicity","Caritas Hospital"],
    "Bhopal":           ["AIIMS Bhopal","Hamidia Hospital","Bansal Hospital","Chirayu Hospital","Peoples Hospital","NSCB Medical","Care CHL Hospital","Nova Hospital","Sahitya Hospital","City Hospital"],
    "Patna":            ["IGIMS Patna","PMCH Patna","Paras HMRI","Ruban Memorial","Big Apollo","Nalanda Medical","Anugrah Narayan","Kurji Holy Family","Mahavir Cancer","Rajan Hospital"],
    "Guwahati":         ["Gauhati Medical College","Dispur Hospital","Nemcare Hospital","Down Town Hospital","Hayat Hospital","Excelcare Hospital","Sri Sankar Dev","Wintrobe","Gnrc Hospital","Apollo Guwahati"],
}

# Fill missing cities with generic hospitals
DEFAULT_HOSPITALS = [
    "Government District Hospital","District Civil Hospital","Apollo Clinic",
    "Fortis Clinic","Max Clinic","Care Hospital","City Hospital",
    "General Hospital","Medical College Hospital","Primary Health Centre",
    "Community Health Centre","Specialty Hospital","Super Speciality Hospital",
    "Nursing Home & Hospital","Multispeciality Hospital"
]

# ── Diseases → Specialty mapping ──────────────────────────────────────────────
DISEASE_SPECIALTY = {
    "Coronary Artery Disease":    "Cardiologist",
    "Stroke":                     "Neurologist",
    "Hypertension":               "Cardiologist",
    "Heart Failure":              "Cardiologist",
    "Peripheral Artery Disease":  "Vascular Surgeon",
    "COPD":                       "Pulmonologist",
    "Asthma":                     "Pulmonologist",
    "Pneumonia":                  "Pulmonologist",
    "Tuberculosis":               "Pulmonologist",
    "Lung Cancer":                "Oncologist",
    "Diabetes Type 1":            "Endocrinologist",
    "Diabetes Type 2":            "Endocrinologist",
    "Chronic Kidney Disease":     "Nephrologist",
    "Liver Cirrhosis":            "Hepatologist",
    "Obesity":                    "Endocrinologist",
    "Alzheimer's Disease":        "Geriatrician",
    "HIV/AIDS":                   "Infectious Disease",
    "Malaria":                    "Infectious Disease",
    "Dengue Fever":               "Infectious Disease",
    "Hepatitis B":                "Hepatologist",
    "Hepatitis C":                "Hepatologist",
    "Influenza":                  "General Physician",
    "COVID-19":                   "Infectious Disease",
    "Cholera":                    "Infectious Disease",
    "Typhoid Fever":              "Infectious Disease",
    "Meningitis":                 "Neurologist",
    "Breast Cancer":              "Oncologist",
    "Colorectal Cancer":          "Oncologist",
    "Prostate Cancer":            "Urologist",
    "Pancreatic Cancer":          "Oncologist",
    "Cervical Cancer":            "Gynaecologist",
    "Migraine":                   "Neurologist",
    "Anemia":                     "Haematologist",
    "Gastroenteritis":            "Gastroenterologist",
    "UTI":                        "Urologist",
    "Depression":                 "Psychiatrist",
    "Appendicitis":               "Gastroenterologist",
}

# ── Sample Reviews ────────────────────────────────────────────────────────────
POSITIVE_REVIEWS = [
    "Excellent doctor! Very thorough in examination and explained everything clearly. Highly recommended.",
    "Best doctor in the city. He diagnosed my condition accurately when others couldn't. Life saver!",
    "Very caring and patient. Took time to listen to all my concerns. Great bedside manner.",
    "Brilliant doctor with vast experience. Treatment was very effective. Completely recovered.",
    "Outstanding physician. Never rushes during consultation. Always available for follow-up queries.",
    "Highly skilled and knowledgeable. My family has been consulting her for 10+ years. We trust her completely.",
    "Exceptional diagnosis skills. Prescribed the right medication on first visit. Amazing doctor!",
    "Very professional and humble. Explained my condition and treatment in simple language. Grateful!",
    "Top-notch doctor. Modern treatment approach combined with compassionate care. Strongly recommend.",
    "Wonderful experience. Doctor is very attentive and the entire staff is courteous. 5 stars!",
    "She is simply the best. My chronic condition improved dramatically after her treatment.",
    "Excellent doctor who goes beyond the call of duty. Called me personally to check progress.",
    "Very experienced and thorough. Ran all necessary tests before diagnosing. Very reliable.",
    "Great doctor! Waited patiently to understand my symptoms fully before prescribing. Wise approach.",
    "Incredibly knowledgeable. Kept up with latest research. Treatment was modern and effective.",
    "My mother was critically ill and this doctor saved her life. Forever grateful to him.",
    "Very systematic approach to diagnosis. Doesn't over-prescribe. Honest and ethical doctor.",
    "Superb doctor. Available on WhatsApp for queries. Very responsive and caring.",
    "Came from another city specifically to consult this doctor. Worth every rupee spent on travel.",
    "Amazing physician. Caught a condition that three other doctors had missed. Exceptional skills.",
]

NEGATIVE_REVIEWS = [
    "Long wait times but the consultation itself was very good. Doctor is knowledgeable.",
    "Slightly expensive but highly skilled. Worth the cost for the expertise provided.",
    "Waiting time is long due to high patient volume. Doctor is very good though.",
    "Hard to get an appointment. Very popular doctor. But consultation quality is excellent.",
    "Could improve communication but medical expertise is top class.",
]

# ── Generate Doctors ──────────────────────────────────────────────────────────
def get_hospitals_for_city(city):
    return HOSPITALS.get(city, DEFAULT_HOSPITALS)

def generate_doctor(doc_id, disease, district, state, idx):
    specialty = DISEASE_SPECIALTY[disease]
    
    # Gender
    is_female = random.random() < 0.35
    first_name = random.choice(FIRST_NAMES_FEMALE if is_female else FIRST_NAMES_MALE)
    last_name   = random.choice(LAST_NAMES)
    name        = f"Dr. {first_name} {last_name}"
    gender      = "Female" if is_female else "Male"

    # Experience & degree
    experience  = random.randint(5, 42)
    degree      = random.choice(DEGREES.get(specialty, DEGREES["General Physician"]))

    # Hospital
    hospitals   = get_hospitals_for_city(district)
    hospital    = random.choice(hospitals)

    # Rating (weighted toward high)
    rating      = round(random.choices(
        [5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.3, 4.0, 3.8, 3.5],
        weights=[5, 15, 25, 25, 15, 7, 4, 2, 1, 1]
    )[0], 1)

    # Consultation fee
    fee_base    = {"Oncologist":1500,"Cardiologist":1200,"Neurologist":1200,"Nephrologist":1000,
                   "Hepatologist":1000,"Endocrinologist":900,"Haematologist":900,"Vascular Surgeon":1100,
                   "Psychiatrist":800,"Gynaecologist":800,"Urologist":800,"Gastroenterologist":900,
                   "Pulmonologist":800,"Infectious Disease":700,"General Physician":500,
                   "Geriatrician":700,"Dermatologist":600,"Orthopaedic Surgeon":900}.get(specialty, 600)
    fee         = fee_base + random.randint(-100, 500)

    # Reviews
    num_reviews = random.randint(12, 340)
    reviews     = []
    num_pos     = max(1, int(num_reviews * (rating / 5.0)))
    num_neg     = max(0, num_reviews - num_pos)
    reviewer_names = [f"{random.choice(FIRST_NAMES_MALE + FIRST_NAMES_FEMALE)} {random.choice(LAST_NAMES)}" for _ in range(min(5, num_reviews))]
    for i in range(min(5, num_reviews)):
        is_pos = i < max(1, int(5 * (rating/5.0)))
        reviews.append({
            "name":   reviewer_names[i],
            "rating": random.randint(4,5) if is_pos else random.randint(2,4),
            "comment": random.choice(POSITIVE_REVIEWS if is_pos else NEGATIVE_REVIEWS),
            "date":   f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.randint(2022,2024)}"
        })

    # Slots
    all_slots = ["08:00 AM","08:30 AM","09:00 AM","09:30 AM","10:00 AM","10:30 AM",
                 "11:00 AM","11:30 AM","12:00 PM","12:30 PM","02:00 PM","02:30 PM",
                 "03:00 PM","03:30 PM","04:00 PM","04:30 PM","05:00 PM","05:30 PM","06:00 PM"]
    slots = sorted(random.sample(all_slots, 3))

    # Distance
    distance = f"{round(random.uniform(0.3, 15.0), 1)} km"

    # Phone
    phone = f"+91 {random.randint(6,9)}{random.randint(100000000,999999999)}"

    return {
        "doctor_id":    f"DOC{doc_id:05d}",
        "name":         name,
        "gender":       gender,
        "specialty":    specialty,
        "disease":      disease,
        "degree":       degree,
        "experience_years": experience,
        "hospital":     hospital,
        "district":     district,
        "state":        state,
        "distance_km":  distance,
        "rating":       rating,
        "total_reviews": num_reviews,
        "consultation_fee_inr": fee,
        "phone":        phone,
        "slots":        ", ".join(slots),
        "reviews":      json.dumps(reviews),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def generate_all():
    diseases  = list(DISEASE_SPECIALTY.keys())   # 37 diseases
    all_districts = []
    for state, dists in INDIA_DISTRICTS.items():
        for d in dists:
            all_districts.append((d, state))

    records   = []
    doc_id    = 1

    print(f"Generating doctors for {len(diseases)} diseases × 50 doctors = {len(diseases)*50} records")
    print(f"Spread across {len(all_districts)} districts in {len(INDIA_DISTRICTS)} states\n")

    for disease in diseases:
        # Pick 50 districts (with repetition allowed for less-covered diseases)
        chosen = random.choices(all_districts, k=50)
        for idx, (district, state) in enumerate(chosen):
            doc = generate_doctor(doc_id, disease, district, state, idx)
            records.append(doc)
            doc_id += 1
        print(f"  ✓ {disease:35s} — 50 doctors generated")

    df = pd.DataFrame(records)

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    csv_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "india_doctors_dataset.csv")
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "india_doctors_dataset.json")

    # Save without reviews column for CSV readability
    df_csv = df.drop(columns=["reviews"])
    df_csv.to_csv(csv_path, index=False)

    # Save full JSON (includes reviews)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\n{'='*55}")
    print(f"  ✅ Dataset Generated!")
    print(f"{'='*55}")
    print(f"  Total doctors  : {len(df)}")
    print(f"  Diseases       : {df['disease'].nunique()}")
    print(f"  States         : {df['state'].nunique()}")
    print(f"  Districts      : {df['district'].nunique()}")
    print(f"  Specialties    : {df['specialty'].nunique()}")
    print(f"  Avg rating     : {df['rating'].mean():.2f}")
    print(f"  Avg experience : {df['experience_years'].mean():.1f} years")
    print(f"\n  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")
    print(f"{'='*55}\n")

    # Summary
    print("Doctors per disease:")
    print(df.groupby("disease")["doctor_id"].count().to_string())
    print("\nDoctors per state:")
    print(df.groupby("state")["doctor_id"].count().sort_values(ascending=False).to_string())

    return df

if __name__ == "__main__":
    df = generate_all()
