import sqlite3

DB_NAME = "databases/plant_schedule.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT NOT NULL UNIQUE
    )
    """)
    
    # Schedules table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS schedules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        task TEXT NOT NULL,
        time TEXT NOT NULL,
        frequency TEXT NOT NULL,
        day TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    
    conn.commit()
    conn.close()

def add_user(name, phone):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO users (name, phone) VALUES (?, ?)", (name, phone))
    conn.commit()
    conn.close()

def add_schedule(user_id, task, time, frequency, day=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO schedules (user_id, task, time, frequency, day) VALUES (?, ?, ?, ?, ?)",
        (user_id, task, time, frequency, day)
    )
    conn.commit()
    conn.close()

def get_schedules():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT users.name, users.phone, schedules.task, schedules.time, schedules.frequency, schedules.day
    FROM schedules
    JOIN users ON schedules.user_id = users.id
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows
