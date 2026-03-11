"""Comments table extension."""
import sqlite3, os
from .db import get_conn, DB_PATH

def init_comments():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doctor_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            user_name TEXT,
            comment TEXT NOT NULL,
            rating INTEGER NOT NULL,
            date TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit(); conn.close()

def add_comment(doctor_id, user_id, user_name, comment, rating):
    conn = get_conn()
    conn.execute("INSERT INTO doctor_comments (doctor_id,user_id,user_name,comment,rating) VALUES (?,?,?,?,?)",
                 (doctor_id, user_id, user_name, comment, int(rating)))
    conn.commit(); conn.close()

def get_comments(doctor_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM doctor_comments WHERE doctor_id=? ORDER BY id DESC LIMIT 20", (doctor_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def user_already_commented(doctor_id, user_id):
    conn = get_conn()
    r = conn.execute("SELECT id FROM doctor_comments WHERE doctor_id=? AND user_id=?", (doctor_id, user_id)).fetchone()
    conn.close()
    return r is not None
