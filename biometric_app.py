#!/usr/bin/env python3
"""
============================================================
  BIOMETRIC FACIAL RECOGNITION SYSTEM
  Cross-Platform Python Application
  Uses: OpenCV (LBPH), SQLite, Tkinter/CLI fallback
============================================================
"""

import os
import sys
import cv2
import numpy as np
import sqlite3
import pickle
import datetime
import hashlib
import base64
import io
import logging
from pathlib import Path

# ── Logging Setup ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("BiometricApp")

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "biometric.db"
MODEL_PATH = BASE_DIR / "face_model.yml"
FACES_DIR  = BASE_DIR / "face_images"
FACES_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_SIZE     = (150, 150)
MIN_CONF     = 80   # LBPH confidence threshold (lower = more similar)


# ══════════════════════════════════════════════════════════
#  DATABASE LAYER
# ══════════════════════════════════════════════════════════
class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                age         INTEGER NOT NULL,
                sex         TEXT    NOT NULL CHECK(sex IN ('Male','Female','Other')),
                dob         TEXT    NOT NULL,
                enrolled    INTEGER NOT NULL DEFAULT 1,
                face_label  INTEGER UNIQUE,
                created_at  TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS face_samples (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   INTEGER NOT NULL REFERENCES persons(id),
                face_data   BLOB    NOT NULL,
                captured_at TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS recognition_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id     INTEGER REFERENCES persons(id),
                confidence    REAL,
                result        TEXT,
                recognized_at TEXT    DEFAULT (datetime('now'))
            );
            """)
        log.info("Database initialised at %s", self.db_path)

    # ── Persons ──────────────────────────────────────────
    def add_person(self, name, age, sex, dob, enrolled=1):
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO persons(name, age, sex, dob, enrolled) VALUES(?,?,?,?,?)",
                (name, int(age), sex, dob, int(enrolled))
            )
            pid = cur.lastrowid
            # face_label = person id (used by LBPH)
            conn.execute("UPDATE persons SET face_label=? WHERE id=?", (pid, pid))
            return pid

    def get_person(self, person_id):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM persons WHERE id=?", (person_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_persons(self, enrolled_only=False):
        with self._conn() as conn:
            if enrolled_only:
                rows = conn.execute(
                    "SELECT * FROM persons WHERE enrolled=1 ORDER BY id"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM persons ORDER BY enrolled DESC, id"
                ).fetchall()
            return [dict(r) for r in rows]

    def person_count(self):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, SUM(enrolled) AS e FROM persons"
            ).fetchone()
            return row["n"], row["e"] or 0

    # ── Face Samples ────────────────────────────────────
    def save_face_sample(self, person_id, face_gray_img):
        data = pickle.dumps(face_gray_img)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO face_samples(person_id, face_data) VALUES(?,?)",
                (person_id, data)
            )

    def get_face_samples(self, enrolled_only=True):
        """Returns list of (label, face_array) tuples for training."""
        with self._conn() as conn:
            if enrolled_only:
                query = """
                    SELECT p.face_label, fs.face_data
                    FROM face_samples fs
                    JOIN persons p ON fs.person_id = p.id
                    WHERE p.enrolled = 1
                """
            else:
                query = """
                    SELECT p.face_label, fs.face_data
                    FROM face_samples fs
                    JOIN persons p ON fs.person_id = p.id
                """
            rows = conn.execute(query).fetchall()
        labels, faces = [], []
        for row in rows:
            labels.append(row["face_label"])
            faces.append(pickle.loads(row["face_data"]))
        return labels, faces

    def face_sample_count(self, person_id):
        with self._conn() as conn:
            r = conn.execute(
                "SELECT COUNT(*) AS n FROM face_samples WHERE person_id=?",
                (person_id,)
            ).fetchone()
            return r["n"]

    # ── Recognition Log ─────────────────────────────────
    def log_recognition(self, person_id, confidence, result):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO recognition_log(person_id, confidence, result) VALUES(?,?,?)",
                (person_id, confidence, result)
            )

    def get_recognition_log(self, limit=50):
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT rl.*, p.name
                FROM recognition_log rl
                LEFT JOIN persons p ON rl.person_id = p.id
                ORDER BY rl.id DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════
#  FACE ENGINE (OpenCV LBPH)
# ══════════════════════════════════════════════════════════
class FaceEngine:
    def __init__(self):
        self.cascade    = cv2.CascadeClassifier(CASCADE_PATH)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._trained   = False
        if MODEL_PATH.exists():
            self.recognizer.read(str(MODEL_PATH))
            self._trained = True
            log.info("Loaded existing face model from %s", MODEL_PATH)

    def detect_faces(self, gray_img):
        """Returns list of (x,y,w,h) bounding boxes."""
        faces = self.cascade.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=5,
            minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces if len(faces) else []

    def preprocess(self, gray_img, bbox=None):
        """Crop (optional), resize, equalise histogram."""
        if bbox is not None:
            x, y, w, h = bbox
            gray_img = gray_img[y:y+h, x:x+w]
        resized = cv2.resize(gray_img, IMG_SIZE)
        return cv2.equalizeHist(resized)

    def train(self, labels, faces):
        if not faces:
            log.warning("No faces to train on.")
            return False
        np_faces = [np.array(f, dtype=np.uint8) for f in faces]
        np_labels = np.array(labels, dtype=np.int32)
        self.recognizer.train(np_faces, np_labels)
        self.recognizer.save(str(MODEL_PATH))
        self._trained = True
        log.info("Model trained on %d samples, saved to %s", len(faces), MODEL_PATH)
        return True

    def predict(self, face_img):
        """Returns (label, confidence). Lower confidence = better match."""
        if not self._trained:
            return None, None
        label, conf = self.recognizer.predict(face_img)
        return label, conf

    @staticmethod
    def generate_synthetic_face(seed: int, enrolled: bool = True):
        """
        Generate a deterministic synthetic face image.
        Returns a BGR colour image (for display) and the grayscale (for training).
        """
        rng = np.random.default_rng(seed)
        size = 200

        # Background gradient
        bg = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            val = int(200 + 30 * (i / size))
            bg[i, :] = (val - 20, val - 10, val)

        # Skin tone variety
        skin_base = int(rng.integers(140, 210))
        skin = (
            max(0, skin_base - int(rng.integers(20, 60))),
            max(0, skin_base - int(rng.integers(10, 40))),
            max(0, skin_base - int(rng.integers(0, 20)))
        )

        # Face oval
        cx, cy = size // 2, size // 2 + 10
        face_w = int(rng.integers(58, 72))
        face_h = int(rng.integers(75, 88))
        cv2.ellipse(bg, (cx, cy), (face_w, face_h), 0, 0, 360, skin, -1)

        # Hair
        hair_col = (
            int(rng.integers(10, 80)),
            int(rng.integers(10, 70)),
            int(rng.integers(5, 50))
        )
        cv2.ellipse(bg, (cx, cy - 20), (face_w + 5, face_h - 20), 0, 180, 360, hair_col, -1)
        cv2.rectangle(bg, (cx - face_w - 5, 0), (cx + face_w + 5, cy - 80), hair_col, -1)

        # Eyes
        eye_x_off = int(rng.integers(22, 32))
        eye_y    = cy - int(rng.integers(12, 22))
        eye_col  = (int(rng.integers(20,60)), int(rng.integers(20,60)), int(rng.integers(20,60)))
        for ex in [cx - eye_x_off, cx + eye_x_off]:
            cv2.ellipse(bg, (ex, eye_y), (9, 6), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(bg, (ex, eye_y), 4, eye_col, -1)
            cv2.circle(bg, (ex, eye_y), 2, (0, 0, 0), -1)

        # Eyebrows
        brow_col = hair_col
        for ex in [cx - eye_x_off, cx + eye_x_off]:
            cv2.line(bg, (ex - 9, eye_y - 12), (ex + 9, eye_y - 9), brow_col, 2)

        # Nose
        nose_y = cy + int(rng.integers(5, 15))
        nose_w = int(rng.integers(6, 10))
        cv2.ellipse(bg, (cx, nose_y), (nose_w, 5), 0, 0, 360, skin, -1)
        cv2.circle(bg, (cx - nose_w, nose_y + 2), 3, (
            max(0, skin[0]-30), max(0, skin[1]-30), max(0, skin[2]-30)), -1)
        cv2.circle(bg, (cx + nose_w, nose_y + 2), 3, (
            max(0, skin[0]-30), max(0, skin[1]-30), max(0, skin[2]-30)), -1)

        # Mouth
        mouth_y = cy + int(rng.integers(25, 38))
        lip_col = (
            max(0, skin[0] - int(rng.integers(40, 70))),
            max(0, skin[1] - int(rng.integers(50, 80))),
            max(0, skin[2] - int(rng.integers(50, 80)))
        )
        cv2.ellipse(bg, (cx, mouth_y), (int(rng.integers(14, 20)), 6), 0, 0, 180, lip_col, -1)
        cv2.ellipse(bg, (cx, mouth_y), (int(rng.integers(14, 20)), 4), 0, 180, 360, lip_col, -1)

        # Optional glasses (enrolled persons only, some of them)
        if enrolled and seed % 3 == 0:
            gx = int(rng.integers(18, 24))
            cv2.circle(bg, (cx - gx, eye_y), 11, (80, 80, 80), 2)
            cv2.circle(bg, (cx + gx, eye_y), 11, (80, 80, 80), 2)
            cv2.line(bg, (cx - gx + 11, eye_y), (cx + gx - 11, eye_y), (80, 80, 80), 1)

        # Convert to grayscale for model
        gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        return bg, gray


# ══════════════════════════════════════════════════════════
#  SEED DATA  (10 enrolled + 10 control)
# ══════════════════════════════════════════════════════════
ENROLLED_PERSONS = [
    {"name": "Adaeze Okonkwo",   "age": 28, "sex": "Female", "dob": "1997-03-14"},
    {"name": "Emeka Nwosu",      "age": 34, "sex": "Male",   "dob": "1991-07-22"},
    {"name": "Fatima Al-Hassan", "age": 22, "sex": "Female", "dob": "2003-11-05"},
    {"name": "Chidi Obiora",     "age": 45, "sex": "Male",   "dob": "1980-01-30"},
    {"name": "Ngozi Eze",        "age": 31, "sex": "Female", "dob": "1994-09-18"},
    {"name": "Babatunde Lawal",  "age": 27, "sex": "Male",   "dob": "1998-06-07"},
    {"name": "Ifeoma Uche",      "age": 38, "sex": "Female", "dob": "1987-04-25"},
    {"name": "Olumide Adeyemi",  "age": 52, "sex": "Male",   "dob": "1973-12-10"},
    {"name": "Chisom Nze",       "age": 19, "sex": "Female", "dob": "2006-08-03"},
    {"name": "Taiwo Ogundimu",   "age": 41, "sex": "Male",   "dob": "1984-02-17"},
]

CONTROL_PERSONS = [
    {"name": "Kelechi Mbah",      "age": 30, "sex": "Male",   "dob": "1995-05-09"},
    {"name": "Amina Suleiman",    "age": 24, "sex": "Female", "dob": "2001-10-21"},
    {"name": "Rotimi Akande",     "age": 36, "sex": "Male",   "dob": "1989-03-16"},
    {"name": "Uchenna Obi",       "age": 29, "sex": "Male",   "dob": "1996-07-04"},
    {"name": "Blessing Nwachukwu","age": 43, "sex": "Female", "dob": "1982-11-28"},
    {"name": "Danladi Musa",      "age": 25, "sex": "Male",   "dob": "2000-01-15"},
    {"name": "Orisa Efidi",       "age": 33, "sex": "Female", "dob": "1992-08-30"},
    {"name": "Segun Fasanya",     "age": 48, "sex": "Male",   "dob": "1977-04-11"},
    {"name": "Chiamaka Igwe",     "age": 21, "sex": "Female", "dob": "2004-09-27"},
    {"name": "Nnamdi Okafor",     "age": 55, "sex": "Male",   "dob": "1970-06-02"},
]


def seed_database(db: Database, engine: FaceEngine):
    """Populate DB with 10 enrolled + 10 control persons if empty."""
    total, enrolled_n = db.person_count()
    if total >= 20:
        log.info("Database already seeded (%d persons).", total)
        return

    log.info("Seeding database with 20 persons …")

    # Enrolled persons
    for idx, p in enumerate(ENROLLED_PERSONS):
        pid = db.add_person(p["name"], p["age"], p["sex"], p["dob"], enrolled=1)
        seed = 100 + idx  # deterministic seed per person
        for variant in range(5):  # 5 face samples per person
            _, gray = engine.generate_synthetic_face(seed + variant * 1000, enrolled=True)
            face_proc = engine.preprocess(gray)
            db.save_face_sample(pid, face_proc)
        log.info("  Enrolled: %s (id=%d)", p["name"], pid)

    # Control persons (NOT enrolled – no face samples for training)
    for idx, p in enumerate(CONTROL_PERSONS):
        pid = db.add_person(p["name"], p["age"], p["sex"], p["dob"], enrolled=0)
        log.info("  Control:  %s (id=%d)", p["name"], pid)

    # Train model on enrolled persons only
    labels, faces = db.get_face_samples(enrolled_only=True)
    engine.train(labels, faces)
    log.info("Database seeding complete.")


# ══════════════════════════════════════════════════════════
#  CONSOLE / CLI  APPLICATION
# ══════════════════════════════════════════════════════════
class ConsoleApp:
    def __init__(self):
        self.db     = Database(DB_PATH)
        self.engine = FaceEngine()
        seed_database(self.db, self.engine)

    def run(self):
        print("\n" + "═" * 60)
        print("  BIOMETRIC FACIAL RECOGNITION SYSTEM")
        print("  Python | OpenCV LBPH | SQLite")
        print("═" * 60)

        while True:
            print("\n  MAIN MENU")
            print("  1. Enroll new person")
            print("  2. Recognize face (simulate)")
            print("  3. List all persons (enrolled)")
            print("  4. List control group")
            print("  5. View recognition log")
            print("  6. Database statistics")
            print("  0. Exit")
            choice = input("\n  Enter choice: ").strip()

            if   choice == "1": self.enroll_new()
            elif choice == "2": self.recognize()
            elif choice == "3": self.list_persons(enrolled_only=True)
            elif choice == "4": self.list_control()
            elif choice == "5": self.view_log()
            elif choice == "6": self.show_stats()
            elif choice == "0": print("  Goodbye.\n"); break
            else: print("  Invalid choice.")

    # ── Enroll ───────────────────────────────────────────
    def enroll_new(self):
        print("\n  ── ENROLLMENT MODE ──")
        name = input("  Full name : ").strip()
        if not name:
            print("  Name cannot be empty."); return
        while True:
            try:
                age = int(input("  Age       : "))
                break
            except ValueError:
                print("  Please enter a valid age.")
        sex = input("  Sex (Male/Female/Other): ").strip().capitalize()
        if sex not in ("Male", "Female", "Other"):
            print("  Invalid sex. Defaulting to 'Other'.")
            sex = "Other"
        dob = input("  Date of Birth (YYYY-MM-DD): ").strip()
        try:
            datetime.datetime.strptime(dob, "%Y-%m-%d")
        except ValueError:
            print("  Invalid date format, using today's date.")
            dob = datetime.date.today().isoformat()

        pid = self.db.add_person(name, age, sex, dob, enrolled=1)
        print(f"\n  Person added (ID={pid}). Capturing face samples …")

        seed = pid * 7 + 999
        for i in range(5):
            _, gray = self.engine.generate_synthetic_face(seed + i * 500, enrolled=True)
            face_proc = self.engine.preprocess(gray)
            self.db.save_face_sample(pid, face_proc)

        # Retrain
        labels, faces = self.db.get_face_samples(enrolled_only=True)
        self.engine.train(labels, faces)
        print(f"  ✓ {name} enrolled successfully! Model retrained.")

    # ── Recognise ────────────────────────────────────────
    def recognize(self):
        print("\n  ── RECOGNITION MODE ──")
        print("  Subjects to test:")
        print("  1. Simulate enrolled person match")
        print("  2. Simulate control (unknown) person")
        choice = input("  Choice: ").strip()

        if choice == "1":
            enrolled = self.db.get_all_persons(enrolled_only=True)
            print(f"\n  Enrolled persons ({len(enrolled)}):")
            for p in enrolled:
                print(f"    [{p['id']:3d}] {p['name']}")
            try:
                pid = int(input("  Enter person ID to test: "))
                person = self.db.get_person(pid)
                if not person or not person["enrolled"]:
                    print("  Person not found or not enrolled."); return
                seed = pid * 7 + 999 + 250  # slightly different seed
                _, gray = self.engine.generate_synthetic_face(seed, enrolled=True)
            except ValueError:
                print("  Invalid ID."); return

        elif choice == "2":
            control = self.db.get_all_persons()
            ctrl_list = [p for p in control if not p["enrolled"]]
            print(f"\n  Control persons ({len(ctrl_list)}):")
            for i, p in enumerate(ctrl_list):
                print(f"    [{i+1:2d}] {p['name']}")
            try:
                idx = int(input("  Enter number: ")) - 1
                person = ctrl_list[idx]
                seed = person["id"] * 13 + 500
                _, gray = self.engine.generate_synthetic_face(seed, enrolled=False)
                pid = person["id"]
            except (ValueError, IndexError):
                print("  Invalid selection."); return
        else:
            print("  Invalid choice."); return

        face_proc = self.engine.preprocess(gray)
        label, confidence = self.engine.predict(face_proc)

        print("\n  ── RESULT ──")
        if label is not None and confidence < MIN_CONF:
            matched = self.db.get_person(label)
            if matched:
                result = "MATCH"
                print(f"  ✓ IDENTIFIED: {matched['name']}")
                print(f"    Age: {matched['age']}  |  Sex: {matched['sex']}")
                print(f"    DOB: {matched['dob']}")
                print(f"    Confidence Score: {confidence:.2f} (threshold={MIN_CONF})")
                self.db.log_recognition(label, confidence, result)
            else:
                print(f"  ✗ UNKNOWN (label={label}, conf={confidence:.2f})")
                self.db.log_recognition(None, confidence, "UNKNOWN")
        else:
            print(f"  ✗ PERSON NOT RECOGNISED (conf={confidence:.2f})")
            print(f"    This person is NOT in the enrolled database.")
            self.db.log_recognition(None, confidence, "NO_MATCH")

    # ── List ─────────────────────────────────────────────
    def list_persons(self, enrolled_only=True):
        persons = self.db.get_all_persons(enrolled_only)
        tag = "ENROLLED" if enrolled_only else "ALL"
        print(f"\n  ── {tag} PERSONS ──")
        print(f"  {'ID':>4}  {'Name':<25}  {'Age':>4}  {'Sex':<8}  {'DOB':<12}  {'Samples':>7}")
        print("  " + "─" * 70)
        for p in persons:
            samples = self.db.face_sample_count(p["id"])
            print(f"  {p['id']:>4}  {p['name']:<25}  {p['age']:>4}  "
                  f"{p['sex']:<8}  {p['dob']:<12}  {samples:>7}")

    def list_control(self):
        all_p = self.db.get_all_persons()
        ctrl = [p for p in all_p if not p["enrolled"]]
        print(f"\n  ── CONTROL GROUP (Not Enrolled) ──")
        print(f"  {'ID':>4}  {'Name':<25}  {'Age':>4}  {'Sex':<8}  {'DOB'}")
        print("  " + "─" * 60)
        for p in ctrl:
            print(f"  {p['id']:>4}  {p['name']:<25}  {p['age']:>4}  {p['sex']:<8}  {p['dob']}")

    def view_log(self):
        logs = self.db.get_recognition_log(20)
        print(f"\n  ── RECOGNITION LOG (last {len(logs)}) ──")
        print(f"  {'Time':<20}  {'Name':<25}  {'Confidence':>10}  {'Result'}")
        print("  " + "─" * 70)
        for entry in logs:
            name = entry.get("name") or "Unknown"
            conf = f"{entry['confidence']:.2f}" if entry["confidence"] else "N/A"
            print(f"  {entry['recognized_at']:<20}  {name:<25}  {conf:>10}  {entry['result']}")

    def show_stats(self):
        total, enrolled_n = self.db.person_count()
        control_n = total - enrolled_n
        labels, faces = self.db.get_face_samples(enrolled_only=True)
        logs = self.db.get_recognition_log(1000)
        matches  = sum(1 for l in logs if l["result"] == "MATCH")
        no_match = sum(1 for l in logs if l["result"] in ("NO_MATCH", "UNKNOWN"))

        print("\n  ── DATABASE STATISTICS ──")
        print(f"  Total persons         : {total}")
        print(f"  Enrolled persons      : {enrolled_n}")
        print(f"  Control (unenrolled)  : {control_n}")
        print(f"  Total face samples    : {len(faces)}")
        print(f"  Model trained         : {'Yes' if self.engine._trained else 'No'}")
        print(f"  Total recognitions    : {len(logs)}")
        print(f"    Successful matches  : {matches}")
        print(f"    No match / Unknown  : {no_match}")
        print(f"  Database path         : {DB_PATH}")
        print(f"  Model path            : {MODEL_PATH}")


# ══════════════════════════════════════════════════════════
#  TKINTER GUI  APPLICATION
# ══════════════════════════════════════════════════════════
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, font as tkFont
    from PIL import Image, ImageTk
    HAS_TK = True
except ImportError:
    HAS_TK = False
    log.warning("Tkinter or Pillow not available – using console mode.")


if HAS_TK:
    class GUIApp(tk.Tk):
        # ── Colour Palette ─────────────────────────────
        BG       = "#0d1117"
        PANEL    = "#161b22"
        ACCENT   = "#00ff88"
        ACCENT2  = "#00ccff"
        TEXT     = "#e6edf3"
        MUTED    = "#8b949e"
        DANGER   = "#ff4444"
        WARNING  = "#ffaa00"
        SUCCESS  = "#00ff88"
        BORDER   = "#30363d"

        def __init__(self):
            super().__init__()
            self.db     = Database(DB_PATH)
            self.engine = FaceEngine()
            seed_database(self.db, self.engine)

            self.title("Biometric Facial Recognition System")
            self.configure(bg=self.BG)
            self.geometry("1200x750")
            self.minsize(1000, 650)
            self.resizable(True, True)

            self._setup_styles()
            self._build_ui()
            self._refresh_table()

        # ── Styles ──────────────────────────────────────
        def _setup_styles(self):
            style = ttk.Style(self)
            style.theme_use("clam")
            style.configure(".", background=self.BG, foreground=self.TEXT,
                            fieldbackground=self.PANEL, bordercolor=self.BORDER,
                            font=("Consolas", 10))
            style.configure("Treeview",
                            background=self.PANEL, foreground=self.TEXT,
                            rowheight=28, fieldbackground=self.PANEL,
                            bordercolor=self.BORDER)
            style.configure("Treeview.Heading",
                            background=self.BORDER, foreground=self.ACCENT,
                            font=("Consolas", 10, "bold"))
            style.map("Treeview", background=[("selected", "#21262d")])
            style.configure("TNotebook", background=self.BG)
            style.configure("TNotebook.Tab", background=self.PANEL,
                            foreground=self.MUTED, padding=[12, 6],
                            font=("Consolas", 10))
            style.map("TNotebook.Tab",
                      background=[("selected", self.BG)],
                      foreground=[("selected", self.ACCENT)])

        # ── Layout ──────────────────────────────────────
        def _build_ui(self):
            # Header
            header = tk.Frame(self, bg=self.PANEL, height=60)
            header.pack(fill="x")
            header.pack_propagate(False)

            tk.Label(header, text="◈  BIOMETRIC FACIAL RECOGNITION SYSTEM",
                     bg=self.PANEL, fg=self.ACCENT,
                     font=("Consolas", 16, "bold")).pack(side="left", padx=20, pady=15)

            self._stat_var = tk.StringVar(value="Loading …")
            tk.Label(header, textvariable=self._stat_var,
                     bg=self.PANEL, fg=self.MUTED,
                     font=("Consolas", 10)).pack(side="right", padx=20)

            # Notebook
            self.nb = ttk.Notebook(self)
            self.nb.pack(fill="both", expand=True, padx=10, pady=10)

            self._tab_dashboard()
            self._tab_enroll()
            self._tab_recognize()
            self._tab_log()

        # ── Tab: Dashboard ──────────────────────────────
        def _tab_dashboard(self):
            tab = tk.Frame(self.nb, bg=self.BG)
            self.nb.add(tab, text="  Dashboard  ")

            # Left: table
            left = tk.Frame(tab, bg=self.BG)
            left.pack(side="left", fill="both", expand=True, padx=(10,5), pady=10)

            filter_bar = tk.Frame(left, bg=self.BG)
            filter_bar.pack(fill="x", pady=(0,5))

            self._filter_var = tk.StringVar(value="enrolled")
            for txt, val in [("Enrolled", "enrolled"), ("Control", "control"), ("All", "all")]:
                tk.Radiobutton(filter_bar, text=txt, variable=self._filter_var,
                               value=val, bg=self.BG, fg=self.TEXT,
                               selectcolor=self.PANEL, activebackground=self.BG,
                               command=self._refresh_table,
                               font=("Consolas", 10)).pack(side="left", padx=6)

            cols = ("ID", "Name", "Age", "Sex", "DOB", "Enrolled", "Samples")
            self.tree = ttk.Treeview(left, columns=cols, show="headings", height=20)
            widths = [40, 200, 50, 70, 100, 70, 70]
            for col, w in zip(cols, widths):
                self.tree.heading(col, text=col)
                self.tree.column(col, width=w, anchor="center")
            self.tree.column("Name", anchor="w")

            vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=vsb.set)
            self.tree.pack(side="left", fill="both", expand=True)
            vsb.pack(side="left", fill="y")

            self.tree.bind("<<TreeviewSelect>>", self._on_select)

            # Right: preview panel
            right = tk.Frame(tab, bg=self.PANEL, width=280)
            right.pack(side="right", fill="y", padx=(5,10), pady=10)
            right.pack_propagate(False)

            tk.Label(right, text="PERSON PREVIEW", bg=self.PANEL, fg=self.ACCENT,
                     font=("Consolas", 11, "bold")).pack(pady=12)

            self._face_label = tk.Label(right, bg=self.PANEL)
            self._face_label.pack(pady=5)

            self._preview_info = tk.Text(right, bg=self.PANEL, fg=self.TEXT,
                                         font=("Consolas", 10), height=12,
                                         state="disabled", relief="flat",
                                         borderwidth=0)
            self._preview_info.pack(fill="x", padx=12, pady=8)

        # ── Tab: Enroll ─────────────────────────────────
        def _tab_enroll(self):
            tab = tk.Frame(self.nb, bg=self.BG)
            self.nb.add(tab, text="  Enroll  ")

            card = tk.Frame(tab, bg=self.PANEL, bd=0)
            card.place(relx=0.5, rely=0.5, anchor="center", width=500, height=520)

            tk.Label(card, text="NEW ENROLLMENT", bg=self.PANEL, fg=self.ACCENT,
                     font=("Consolas", 14, "bold")).pack(pady=(24, 20))

            fields = [
                ("Full Name", "e_name"),
                ("Age",       "e_age"),
                ("Date of Birth (YYYY-MM-DD)", "e_dob"),
            ]
            self._evars = {}
            for label, key in fields:
                row = tk.Frame(card, bg=self.PANEL)
                row.pack(fill="x", padx=30, pady=6)
                tk.Label(row, text=label, bg=self.PANEL, fg=self.MUTED,
                         font=("Consolas", 9), width=28, anchor="w").pack(side="left")
                var = tk.StringVar()
                self._evars[key] = var
                tk.Entry(row, textvariable=var, bg="#21262d", fg=self.TEXT,
                         insertbackground=self.ACCENT, relief="flat",
                         font=("Consolas", 11)).pack(side="right", fill="x", expand=True)

            # Sex selector
            row = tk.Frame(card, bg=self.PANEL)
            row.pack(fill="x", padx=30, pady=6)
            tk.Label(row, text="Sex", bg=self.PANEL, fg=self.MUTED,
                     font=("Consolas", 9), width=28, anchor="w").pack(side="left")
            self._sex_var = tk.StringVar(value="Male")
            sex_cb = ttk.Combobox(row, textvariable=self._sex_var,
                                  values=["Male", "Female", "Other"],
                                  state="readonly", font=("Consolas", 11))
            sex_cb.pack(side="right", fill="x", expand=True)

            # Face preview
            self._enroll_face_lbl = tk.Label(card, bg=self.PANEL, text="[ Face preview ]",
                                             fg=self.MUTED, font=("Consolas", 9))
            self._enroll_face_lbl.pack(pady=10)

            btn_frame = tk.Frame(card, bg=self.PANEL)
            btn_frame.pack(pady=10)

            tk.Button(btn_frame, text="  Preview Face  ",
                      bg="#21262d", fg=self.ACCENT2,
                      font=("Consolas", 10), relief="flat",
                      cursor="hand2", padx=12, pady=6,
                      command=self._preview_face).pack(side="left", padx=8)

            tk.Button(btn_frame, text="  Enroll Now  ",
                      bg=self.ACCENT, fg=self.BG,
                      font=("Consolas", 11, "bold"), relief="flat",
                      cursor="hand2", padx=14, pady=8,
                      command=self._do_enroll).pack(side="left", padx=8)

            self._enroll_status = tk.Label(card, text="", bg=self.PANEL,
                                           fg=self.SUCCESS, font=("Consolas", 10))
            self._enroll_status.pack(pady=6)

        # ── Tab: Recognize ──────────────────────────────
        def _tab_recognize(self):
            tab = tk.Frame(self.nb, bg=self.BG)
            self.nb.add(tab, text="  Recognize  ")

            left = tk.Frame(tab, bg=self.BG)
            left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

            tk.Label(left, text="SELECT PERSON TO TEST",
                     bg=self.BG, fg=self.ACCENT,
                     font=("Consolas", 11, "bold")).pack(anchor="w", pady=(0,6))

            self._rec_mode = tk.StringVar(value="enrolled")
            mf = tk.Frame(left, bg=self.BG)
            mf.pack(fill="x", pady=(0,6))
            for txt, val in [("Test Enrolled Person", "enrolled"),
                             ("Test Control Person (expect rejection)", "control")]:
                tk.Radiobutton(mf, text=txt, variable=self._rec_mode,
                               value=val, bg=self.BG, fg=self.TEXT,
                               selectcolor=self.PANEL, activebackground=self.BG,
                               command=self._rec_load_list,
                               font=("Consolas", 10)).pack(anchor="w")

            cols = ("ID", "Name", "Age", "Sex", "DOB")
            self.rec_tree = ttk.Treeview(left, columns=cols, show="headings", height=15)
            widths = [40, 220, 50, 70, 100]
            for col, w in zip(cols, widths):
                self.rec_tree.heading(col, text=col)
                self.rec_tree.column(col, width=w, anchor="center")
            self.rec_tree.column("Name", anchor="w")
            self.rec_tree.pack(fill="both", expand=True)

            tk.Button(left, text="  ▶  RUN RECOGNITION  ",
                      bg=self.ACCENT2, fg=self.BG,
                      font=("Consolas", 12, "bold"), relief="flat",
                      cursor="hand2", padx=16, pady=10,
                      command=self._do_recognize).pack(pady=10)

            # Right: result panel
            right = tk.Frame(tab, bg=self.PANEL, width=320)
            right.pack(side="right", fill="y", padx=(5,10), pady=10)
            right.pack_propagate(False)

            tk.Label(right, text="RECOGNITION RESULT", bg=self.PANEL, fg=self.ACCENT,
                     font=("Consolas", 11, "bold")).pack(pady=12)

            self._rec_face_lbl = tk.Label(right, bg=self.PANEL)
            self._rec_face_lbl.pack(pady=5)

            self._rec_result_var = tk.StringVar(value="─── Awaiting scan ───")
            self._rec_result_lbl = tk.Label(right, textvariable=self._rec_result_var,
                                            bg=self.PANEL, fg=self.MUTED,
                                            font=("Consolas", 12, "bold"),
                                            wraplength=280, justify="center")
            self._rec_result_lbl.pack(pady=8)

            self._rec_detail = tk.Text(right, bg=self.PANEL, fg=self.TEXT,
                                       font=("Consolas", 10), height=12,
                                       state="disabled", relief="flat", borderwidth=0)
            self._rec_detail.pack(fill="x", padx=12, pady=8)

            self._rec_load_list()

        # ── Tab: Log ────────────────────────────────────
        def _tab_log(self):
            tab = tk.Frame(self.nb, bg=self.BG)
            self.nb.add(tab, text="  Log  ")

            tk.Label(tab, text="RECOGNITION AUDIT LOG",
                     bg=self.BG, fg=self.ACCENT,
                     font=("Consolas", 12, "bold")).pack(anchor="w", padx=14, pady=(12,4))

            cols = ("Timestamp", "Name", "Confidence", "Result")
            self.log_tree = ttk.Treeview(tab, columns=cols, show="headings", height=22)
            widths = [180, 220, 100, 100]
            for col, w in zip(cols, widths):
                self.log_tree.heading(col, text=col)
                self.log_tree.column(col, width=w, anchor="center")
            self.log_tree.column("Name", anchor="w")

            self.log_tree.tag_configure("match",    foreground=self.SUCCESS)
            self.log_tree.tag_configure("no_match", foreground=self.DANGER)
            self.log_tree.tag_configure("unknown",  foreground=self.WARNING)

            vsb = ttk.Scrollbar(tab, orient="vertical", command=self.log_tree.yview)
            self.log_tree.configure(yscrollcommand=vsb.set)
            self.log_tree.pack(side="left", fill="both", expand=True, padx=(14,0), pady=8)
            vsb.pack(side="right", fill="y", pady=8, padx=(0,14))

            tk.Button(tab, text="Refresh Log",
                      bg="#21262d", fg=self.ACCENT,
                      font=("Consolas", 10), relief="flat",
                      cursor="hand2", padx=10, pady=5,
                      command=self._refresh_log).pack(pady=6)

        # ── Actions ─────────────────────────────────────
        def _refresh_table(self):
            mode = self._filter_var.get() if hasattr(self, "_filter_var") else "enrolled"
            all_p = self.db.get_all_persons()
            if mode == "enrolled":
                persons = [p for p in all_p if p["enrolled"]]
            elif mode == "control":
                persons = [p for p in all_p if not p["enrolled"]]
            else:
                persons = all_p

            self.tree.delete(*self.tree.get_children())
            for p in persons:
                samples = self.db.face_sample_count(p["id"])
                enr_tag = "✓ YES" if p["enrolled"] else "✗ NO"
                tag = "enrolled" if p["enrolled"] else "control"
                self.tree.insert("", "end", iid=str(p["id"]),
                                  values=(p["id"], p["name"], p["age"], p["sex"],
                                          p["dob"], enr_tag, samples),
                                  tags=(tag,))
            self.tree.tag_configure("enrolled", foreground=self.TEXT)
            self.tree.tag_configure("control",  foreground=self.WARNING)

            total, enrolled_n = self.db.person_count()
            self._stat_var.set(
                f"DB: {total} persons | Enrolled: {enrolled_n} | "
                f"Control: {total - enrolled_n}"
            )

        def _on_select(self, event):
            sel = self.tree.selection()
            if not sel: return
            pid = int(sel[0])
            p = self.db.get_person(pid)
            if not p: return
            # Generate face image preview
            seed = pid * 7 + 999
            enrolled = bool(p["enrolled"])
            bgr, _ = self.engine.generate_synthetic_face(seed, enrolled=enrolled)
            self._set_face_label(bgr, self._face_label, size=(170, 170))

            info = (
                f"ID       : {p['id']}\n"
                f"Name     : {p['name']}\n"
                f"Age      : {p['age']}\n"
                f"Sex      : {p['sex']}\n"
                f"DOB      : {p['dob']}\n"
                f"Enrolled : {'YES' if p['enrolled'] else 'NO'}\n"
                f"Samples  : {self.db.face_sample_count(pid)}\n"
                f"Enrolled : {p['created_at']}"
            )
            self._preview_info.configure(state="normal")
            self._preview_info.delete("1.0", "end")
            self._preview_info.insert("end", info)
            self._preview_info.configure(state="disabled")

        def _preview_face(self):
            name = self._evars["e_name"].get().strip()
            if not name:
                self._enroll_status.config(text="Enter name first.", fg=self.WARNING)
                return
            seed = abs(hash(name)) % 100000
            bgr, _ = self.engine.generate_synthetic_face(seed, enrolled=True)
            self._set_face_label(bgr, self._enroll_face_lbl, size=(120, 120))

        def _do_enroll(self):
            name = self._evars["e_name"].get().strip()
            age_s = self._evars["e_age"].get().strip()
            dob   = self._evars["e_dob"].get().strip()
            sex   = self._sex_var.get()

            if not name:
                self._enroll_status.config(text="Name required.", fg=self.DANGER); return
            try:
                age = int(age_s)
                if age < 0 or age > 130: raise ValueError
            except ValueError:
                self._enroll_status.config(text="Valid age required.", fg=self.DANGER); return
            try:
                datetime.datetime.strptime(dob, "%Y-%m-%d")
            except ValueError:
                self._enroll_status.config(text="DOB format: YYYY-MM-DD", fg=self.DANGER); return

            pid = self.db.add_person(name, age, sex, dob, enrolled=1)
            seed = abs(hash(name)) % 100000
            for i in range(5):
                _, gray = self.engine.generate_synthetic_face(seed + i * 500, enrolled=True)
                face_proc = self.engine.preprocess(gray)
                self.db.save_face_sample(pid, face_proc)

            labels, faces = self.db.get_face_samples(enrolled_only=True)
            self.engine.train(labels, faces)

            bgr, _ = self.engine.generate_synthetic_face(seed, enrolled=True)
            self._set_face_label(bgr, self._enroll_face_lbl, size=(120, 120))

            self._enroll_status.config(
                text=f"✓ {name} enrolled (ID={pid})!", fg=self.SUCCESS)
            self._refresh_table()
            for v in self._evars.values():
                v.set("")

        def _rec_load_list(self):
            mode = self._rec_mode.get()
            all_p = self.db.get_all_persons()
            if mode == "enrolled":
                persons = [p for p in all_p if p["enrolled"]]
            else:
                persons = [p for p in all_p if not p["enrolled"]]

            self.rec_tree.delete(*self.rec_tree.get_children())
            for p in persons:
                self.rec_tree.insert("", "end", iid=str(p["id"]),
                                      values=(p["id"], p["name"], p["age"],
                                              p["sex"], p["dob"]))

        def _do_recognize(self):
            sel = self.rec_tree.selection()
            if not sel:
                messagebox.showwarning("No Selection", "Select a person first.")
                return
            pid  = int(sel[0])
            mode = self._rec_mode.get()
            p    = self.db.get_person(pid)

            enrolled = bool(p["enrolled"])
            seed = pid * 13 + (250 if enrolled else 500)
            bgr, gray = self.engine.generate_synthetic_face(seed, enrolled=enrolled)
            self._set_face_label(bgr, self._rec_face_lbl, size=(180, 180))

            face_proc = self.engine.preprocess(gray)
            label, confidence = self.engine.predict(face_proc)

            self._rec_detail.configure(state="normal")
            self._rec_detail.delete("1.0", "end")

            if label is not None and confidence < MIN_CONF:
                matched = self.db.get_person(label)
                if matched and matched["enrolled"]:
                    self._rec_result_var.set("✓  MATCH FOUND")
                    self._rec_result_lbl.configure(fg=self.SUCCESS)
                    detail = (
                        f"Identified as:\n"
                        f"  {matched['name']}\n\n"
                        f"Age       : {matched['age']}\n"
                        f"Sex       : {matched['sex']}\n"
                        f"DOB       : {matched['dob']}\n"
                        f"Confidence: {confidence:.2f}\n"
                        f"Threshold : {MIN_CONF}\n\n"
                        f"Status    : ENROLLED PERSON"
                    )
                    self.db.log_recognition(label, confidence, "MATCH")
                else:
                    self._rec_result_var.set("⚠  LOW CONFIDENCE")
                    self._rec_result_lbl.configure(fg=self.WARNING)
                    detail = f"Ambiguous match.\nConf: {confidence:.2f}"
                    self.db.log_recognition(None, confidence, "UNKNOWN")
            else:
                self._rec_result_var.set("✗  NOT RECOGNISED")
                self._rec_result_lbl.configure(fg=self.DANGER)
                conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
                detail = (
                    f"Person: {p['name']}\n\n"
                    f"Confidence: {conf_str}\n"
                    f"Threshold : {MIN_CONF}\n\n"
                    f"This person is NOT in the\nenrolled database.\n\n"
                    f"Status    : {'CONTROL GROUP' if not enrolled else 'VARIANT NOT MATCHED'}"
                )
                self.db.log_recognition(None, confidence, "NO_MATCH")

            self._rec_detail.insert("end", detail)
            self._rec_detail.configure(state="disabled")
            self._refresh_log()

        def _refresh_log(self):
            if not hasattr(self, "log_tree"): return
            logs = self.db.get_recognition_log(100)
            self.log_tree.delete(*self.log_tree.get_children())
            for entry in logs:
                name = entry.get("name") or "—Unknown—"
                conf = f"{entry['confidence']:.2f}" if entry["confidence"] else "N/A"
                result = entry["result"]
                tag = {"MATCH": "match", "NO_MATCH": "no_match"}.get(result, "unknown")
                self.log_tree.insert("", "end",
                                      values=(entry["recognized_at"], name, conf, result),
                                      tags=(tag,))

        # ── Utilities ────────────────────────────────────
        def _set_face_label(self, bgr_img, label_widget, size=(150, 150)):
            try:
                rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize(size, Image.LANCZOS)
                # Add dark border
                bordered = Image.new("RGB", (size[0]+4, size[1]+4), (48, 54, 61))
                bordered.paste(pil, (2, 2))
                imgtk = ImageTk.PhotoImage(bordered)
                label_widget.configure(image=imgtk, text="")
                label_widget.image = imgtk  # keep reference
            except Exception as e:
                label_widget.configure(text="[Face]")
                log.warning("Image display error: %s", e)


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════
def main():
    if HAS_TK:
        try:
            app = GUIApp()
            app.mainloop()
        except Exception as e:
            log.error("GUI failed: %s — falling back to console.", e)
            ConsoleApp().run()
    else:
        ConsoleApp().run()


if __name__ == "__main__":
    main()
