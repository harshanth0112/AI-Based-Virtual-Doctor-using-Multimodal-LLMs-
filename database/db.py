"""
Database module — SQLite with raw sqlite3 (no SQLAlchemy needed)
Handles: users, sessions, consultations
"""
import sqlite3, os, hashlib, secrets, json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "medipredict.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email     TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL,
            dob       TEXT,
            gender    TEXT,
            phone     TEXT,
            blood_group TEXT,
            allergies TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS consultations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            date        TEXT NOT NULL,
            symptom_text TEXT,
            primary_diagnosis TEXT,
            severity    TEXT,
            conditions  TEXT,
            recommendations TEXT,
            age         INTEGER,
            fever       REAL,
            systolic_bp INTEGER,
            diastolic_bp INTEGER,
            heart_rate  INTEGER,
            duration    INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS bookings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            booking_ref TEXT,
            doctor_name TEXT,
            specialty   TEXT,
            hospital    TEXT,
            slot        TEXT,
            date        TEXT,
            status      TEXT DEFAULT 'Confirmed',
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    conn.commit(); conn.close()
    print("✅ Database initialized")

def hash_password(password):
    salt = "medipredict2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

def register_user(full_name, email, password, dob="", gender="", phone="", blood_group="", allergies=""):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (full_name,email,password,dob,gender,phone,blood_group,allergies) VALUES (?,?,?,?,?,?,?,?)",
            (full_name, email.lower(), hash_password(password), dob, gender, phone, blood_group, allergies)
        )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Email already registered. Please login."
    finally:
        conn.close()

def login_user(email, password):
    conn = get_conn()
    user = conn.execute("SELECT * FROM users WHERE email=? AND password=?",
                        (email.lower(), hash_password(password))).fetchone()
    conn.close()
    return dict(user) if user else None

def get_user(user_id):
    conn = get_conn()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None

def save_consultation(user_id, data):
    conn = get_conn()
    conn.execute("""
        INSERT INTO consultations
        (user_id,date,symptom_text,primary_diagnosis,severity,conditions,recommendations,age,fever,systolic_bp,diastolic_bp,heart_rate,duration)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (user_id, datetime.now().strftime("%d %b %Y, %I:%M %p"),
         data.get("symptom_text",""), data.get("primary_diagnosis",""),
         data.get("severity","low"), json.dumps(data.get("conditions",[])),
         json.dumps(data.get("recommendations",[])),
         data.get("age",0), data.get("fever",37.0),
         data.get("systolic_bp",120), data.get("diastolic_bp",80),
         data.get("heart_rate",80), data.get("duration",1))
    )
    conn.commit(); conn.close()

def get_consultations(user_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM consultations WHERE user_id=? ORDER BY id DESC", (user_id,)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["conditions"]      = json.loads(d.get("conditions","[]"))
        d["recommendations"] = json.loads(d.get("recommendations","[]"))
        result.append(d)
    return result

def save_booking(user_id, booking):
    conn = get_conn()
    conn.execute("""
        INSERT INTO bookings (user_id,booking_ref,doctor_name,specialty,hospital,slot,date,status)
        VALUES (?,?,?,?,?,?,?,?)""",
        (user_id, booking["ref"], booking["doctor"], booking["specialty"],
         booking["hospital"], booking["slot"], booking["date"], "Confirmed")
    )
    conn.commit(); conn.close()

def get_bookings(user_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM bookings WHERE user_id=? ORDER BY id DESC", (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_profile(user_id, data):
    conn = get_conn()
    conn.execute("UPDATE users SET full_name=?,phone=?,blood_group=?,allergies=?,dob=?,gender=? WHERE id=?",
                 (data["full_name"],data["phone"],data["blood_group"],data["allergies"],data["dob"],data["gender"],user_id))
    conn.commit(); conn.close()

def init_reviews_table():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id  INTEGER NOT NULL,
            user_id    INTEGER NOT NULL,
            user_name  TEXT NOT NULL,
            rating     INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            comment    TEXT NOT NULL,
            date       TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit(); conn.close()

def add_review(doctor_id, user_id, user_name, rating, comment):
    conn = get_conn()
    # one review per user per doctor
    existing = conn.execute("SELECT id FROM reviews WHERE doctor_id=? AND user_id=?",
                            (doctor_id, user_id)).fetchone()
    if existing:
        conn.execute("UPDATE reviews SET rating=?, comment=?, date=CURRENT_TIMESTAMP WHERE id=?",
                     (rating, comment, existing["id"]))
    else:
        conn.execute("INSERT INTO reviews (doctor_id,user_id,user_name,rating,comment) VALUES (?,?,?,?,?)",
                     (doctor_id, user_id, user_name, rating, comment))
    conn.commit(); conn.close()
    return True

def get_reviews(doctor_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM reviews WHERE doctor_id=? ORDER BY id DESC LIMIT 20", (doctor_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_avg_rating(doctor_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT AVG(rating) as avg, COUNT(*) as cnt FROM reviews WHERE doctor_id=?", (doctor_id,)
    ).fetchone()
    conn.close()
    return round(float(row["avg"] or 0), 1), int(row["cnt"] or 0)
