import math
import sqlite3
from datetime import datetime, timedelta


class DatabaseManager:
    def __init__(self, db_path="patients.db"):
        self.db_path = db_path
        self._create_tables()

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        with self.connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_code TEXT UNIQUE NOT NULL,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    dob TEXT NOT NULL,
                    gestational_weeks INTEGER NOT NULL,
                    gestational_days INTEGER NOT NULL,
                    age INTEGER,
                    race TEXT,
                    height_inches INTEGER,
                    is_hispanic INTEGER DEFAULT 0
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    risk_factor TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bp_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    systolic INTEGER NOT NULL,
                    diastolic INTEGER NOT NULL,
                    gest_age_at_visit INTEGER NOT NULL,
                    weight REAL NOT NULL DEFAULT 0.0,
                    map_value REAL,
                    new_age INTEGER,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ptt_calibration_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    point_number INTEGER NOT NULL,
                    systolic_ref INTEGER NOT NULL,
                    diastolic_ref INTEGER NOT NULL,
                    ptt_ms REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id),
                    UNIQUE(patient_id, point_number)
                )
            """)

    def get_patient_by_code(self, code):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM patients WHERE patient_code = ?
            """, (code,))
            return cursor.fetchone()

    def add_patient(self, patient, risk_factors):
        with self.connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO patients (
                    patient_code, first_name, last_name, dob,
                    gestational_weeks, gestational_days,
                    age, race, height_inches, is_hispanic
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient["code"],
                patient["first_name"],
                patient["last_name"],
                patient["dob"],
                patient["gestational_weeks"],
                patient["gestational_days"],
                patient["age"],
                patient["race"],
                patient["height_inches"],
                patient["is_hispanic"],
           
            ))

            patient_id = cursor.lastrowid

            for factor in risk_factors:
                cursor.execute("""
                    INSERT INTO risk_factors (patient_id, risk_factor)
                    VALUES (?, ?)
                """, (patient_id, factor))

    def get_patient_by_code_and_dob(self, code, dob):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM patients
                WHERE patient_code = ? AND dob = ?
            """, (code, dob))
            return cursor.fetchone()

    def get_patient_by_id(self, patient_id):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM patients WHERE id = ?
            """, (patient_id,))
            return cursor.fetchone()

    def search_patients_by_name(self, name_query):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM patients
                WHERE lower(first_name || ' ' || last_name) LIKE ?
                ORDER BY last_name, first_name
            """, (f"%{name_query.lower()}%",))
            return cursor.fetchall()

    def add_bp_reading(self, patient_id, systolic, diastolic, gest_age_at_visit, weight, map_value, new_age, timestamp):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bp_readings (
                    patient_id, systolic, diastolic, gest_age_at_visit, weight, map_value, new_age, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, systolic, diastolic, gest_age_at_visit, weight, map_value, new_age, timestamp))

    def get_bp_readings(self, patient_id):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT systolic, diastolic, gest_age_at_visit, weight, map_value, timestamp
                FROM bp_readings
                WHERE patient_id = ?
                ORDER BY timestamp DESC, id DESC
            """, (patient_id,))
            return cursor.fetchall()

    def get_risk_factors(self, patient_id):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT risk_factor
                FROM risk_factors
                WHERE patient_id = ?
            """, (patient_id,))
            return [row["risk_factor"] for row in cursor.fetchall()]
        
        
    def get_first_gest_age(self, patient_id):
        """Retrieves the GA from the very first visit."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT gest_age_at_visit FROM bp_readings 
                WHERE patient_id = ? 
                ORDER BY timestamp ASC LIMIT 1
            """, (patient_id,))
            row = cursor.fetchone()
            return row['gest_age_at_visit'] if row else None

    def update_patient_age(self, patient_id, age_weeks):
        """Updates the age column in the patients table."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE patients SET age = ? WHERE id = ?
            """, (age_weeks, patient_id))

    # ── PTT calibration ───────────────────────────────────────────────────────

    def save_calibration_point(self, patient_id, point_number,
                               systolic_ref, diastolic_ref, ptt_ms, timestamp):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO ptt_calibration_points
                    (patient_id, point_number, systolic_ref, diastolic_ref, ptt_ms, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (patient_id, point_number, systolic_ref, diastolic_ref, ptt_ms, timestamp))

    def get_calibration_points(self, patient_id):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT point_number, systolic_ref, diastolic_ref, ptt_ms
                FROM ptt_calibration_points
                WHERE patient_id = ?
                ORDER BY point_number
            """, (patient_id,))
            return cursor.fetchall()

    def get_calibration(self, patient_id):
        """
        Returns fitted (sbp_a, sbp_b, dbp_a, dbp_b) coefficients for the
        Moens-Korteweg model, or None if fewer than 2 calibration points exist.
        """
        points = self.get_calibration_points(patient_id)
        if len(points) < 2:
            return None
        p1, p2 = dict(points[0]), dict(points[1])
        try:
            ln1 = math.log(p1['ptt_ms'])
            ln2 = math.log(p2['ptt_ms'])
            d_ln = ln2 - ln1
            if abs(d_ln) < 0.01:
                return None
            sbp_b = (p1['systolic_ref']  - p2['systolic_ref'])  / d_ln
            sbp_a = p1['systolic_ref']  + sbp_b * ln1
            dbp_b = (p1['diastolic_ref'] - p2['diastolic_ref']) / d_ln
            dbp_a = p1['diastolic_ref'] + dbp_b * ln1
            return float(sbp_a), float(sbp_b), float(dbp_a), float(dbp_b)
        except (ValueError, ZeroDivisionError):
            return None