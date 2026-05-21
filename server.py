#!/usr/bin/env python3
"""
============================================================
  BIOMETRIC FACIAL RECOGNITION SYSTEM
  Real Webcam Capture via Browser getUserMedia API
  Python HTTP Server + OpenCV LBPH + SQLite
  Run:  python3 server.py
============================================================
"""

import json, sqlite3, pickle, datetime, base64, io
import threading, webbrowser, http.server, urllib.parse
from pathlib import Path
import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "biometric.db"
MODEL_PATH = BASE_DIR / "face_model.yml"
HTML_PATH  = BASE_DIR / "index.html"
PORT       = 8787
MIN_CONF   = 75
IMG_SIZE   = (150, 150)
SAMPLES_NEEDED = 5


# ══════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════
class Database:
    def __init__(self):
        self._init()

    def _conn(self):
        c = sqlite3.connect(DB_PATH)
        c.row_factory = sqlite3.Row
        return c

    def _init(self):
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                age         INTEGER NOT NULL,
                sex         TEXT    NOT NULL,
                dob         TEXT    NOT NULL,
                enrolled    INTEGER DEFAULT 1,
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
                recognized_at TEXT DEFAULT (datetime('now'))
            );
            """)

    def add_person(self, name, age, sex, dob, enrolled=1):
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO persons(name,age,sex,dob,enrolled) VALUES(?,?,?,?,?)",
                (name, int(age), sex, dob, int(enrolled))
            )
            pid = cur.lastrowid
            c.execute("UPDATE persons SET face_label=? WHERE id=?", (pid, pid))
            return pid

    def delete_person(self, pid):
        with self._conn() as c:
            c.execute("DELETE FROM face_samples WHERE person_id=?", (pid,))
            c.execute("DELETE FROM persons WHERE id=?", (pid,))

    def get_person(self, pid):
        with self._conn() as c:
            r = c.execute("SELECT * FROM persons WHERE id=?", (pid,)).fetchone()
            return dict(r) if r else None

    def get_all(self, mode="all"):
        with self._conn() as c:
            if mode == "enrolled":
                rows = c.execute("SELECT * FROM persons WHERE enrolled=1 ORDER BY id").fetchall()
            elif mode == "control":
                rows = c.execute("SELECT * FROM persons WHERE enrolled=0 ORDER BY id").fetchall()
            else:
                rows = c.execute("SELECT * FROM persons ORDER BY enrolled DESC, id").fetchall()
            return [dict(r) for r in rows]

    def stats(self):
        with self._conn() as c:
            r = c.execute("SELECT COUNT(*) n, SUM(enrolled) e FROM persons").fetchone()
            s = c.execute("SELECT COUNT(*) n FROM face_samples").fetchone()
            l = c.execute("SELECT COUNT(*) n FROM recognition_log").fetchone()
            m = c.execute("SELECT COUNT(*) n FROM recognition_log WHERE result='MATCH'").fetchone()
            return {
                "total":   r["n"],
                "enrolled": r["e"] or 0,
                "control":  r["n"] - (r["e"] or 0),
                "samples":  s["n"],
                "logs":     l["n"],
                "matches":  m["n"]
            }

    def save_face(self, person_id, face_arr):
        with self._conn() as c:
            c.execute(
                "INSERT INTO face_samples(person_id,face_data) VALUES(?,?)",
                (person_id, pickle.dumps(face_arr))
            )

    def get_face_samples(self, enrolled_only=True):
        with self._conn() as c:
            q = """SELECT p.face_label, fs.face_data
                   FROM face_samples fs
                   JOIN persons p ON fs.person_id=p.id"""
            if enrolled_only:
                q += " WHERE p.enrolled=1"
            rows = c.execute(q).fetchall()
        labels, faces = [], []
        for r in rows:
            labels.append(r["face_label"])
            faces.append(pickle.loads(r["face_data"]))
        return labels, faces

    def sample_count(self, pid):
        with self._conn() as c:
            return c.execute(
                "SELECT COUNT(*) n FROM face_samples WHERE person_id=?", (pid,)
            ).fetchone()["n"]

    def log_rec(self, pid, conf, result):
        with self._conn() as c:
            c.execute(
                "INSERT INTO recognition_log(person_id,confidence,result) VALUES(?,?,?)",
                (pid, conf, result)
            )

    def get_log(self, limit=50):
        with self._conn() as c:
            rows = c.execute("""
                SELECT rl.*, p.name FROM recognition_log rl
                LEFT JOIN persons p ON rl.person_id=p.id
                ORDER BY rl.id DESC LIMIT ?""", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stored_face(self, pid):
        with self._conn() as c:
            r = c.execute(
                "SELECT face_data FROM face_samples WHERE person_id=? LIMIT 1", (pid,)
            ).fetchone()
        if not r:
            return None
        arr = pickle.loads(r["face_data"])
        return arr_to_b64(arr)


# ══════════════════════════════════════════════════════════
#  FACE ENGINE
# ══════════════════════════════════════════════════════════
class FaceEngine:
    CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(self):
        self.cascade    = cv2.CascadeClassifier(self.CASCADE)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.trained    = False
        if MODEL_PATH.exists():
            self.recognizer.read(str(MODEL_PATH))
            self.trained = True
            print("✓ Loaded existing face model.")

    def detect(self, gray):
        """Try multiple parameter sets for robust real-world detection."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        for sf, mn, ms in [(1.1,4,(40,40)), (1.05,3,(30,30)), (1.1,2,(30,30)), (1.3,2,(20,20))]:
            faces = self.cascade.detectMultiScale(
                enhanced, scaleFactor=sf, minNeighbors=mn,
                minSize=ms, flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces):
                return list(faces)
        return []

    def preprocess(self, gray, bbox=None):
        if bbox is not None:
            x, y, w, h = bbox
            pad_x = int(w * 0.1); pad_y = int(h * 0.1)
            x1 = max(0, x-pad_x); y1 = max(0, y-pad_y)
            x2 = min(gray.shape[1], x+w+pad_x)
            y2 = min(gray.shape[0], y+h+pad_y)
            gray = gray[y1:y2, x1:x2]
        resized = cv2.resize(gray, IMG_SIZE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(resized)

    def train(self, labels, faces):
        if not faces:
            return False
        self.recognizer.train(
            [np.array(f, dtype=np.uint8) for f in faces],
            np.array(labels, dtype=np.int32)
        )
        self.recognizer.save(str(MODEL_PATH))
        self.trained = True
        print(f"✓ Model retrained on {len(faces)} samples.")
        return True

    def predict(self, face):
        if not self.trained:
            return None, None
        return self.recognizer.predict(face)


# ══════════════════════════════════════════════════════════
#  IMAGE HELPERS
# ══════════════════════════════════════════════════════════
def b64_to_bgr(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def arr_to_b64(gray_arr, size=(160, 160)):
    resized = cv2.resize(gray_arr, size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    _, buf = cv2.imencode(".png", rgb)
    return "data:image/png;base64," + base64.b64encode(buf).decode()

def bgr_to_b64(bgr_arr, size=(320, 240)):
    resized = cv2.resize(bgr_arr, size)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ══════════════════════════════════════════════════════════
#  DATE HELPER  — accepts YYYY-MM-DD or MM/DD/YYYY
# ══════════════════════════════════════════════════════════
def parse_dob(raw):
    """Parse date from either YYYY-MM-DD or MM/DD/YYYY, return YYYY-MM-DD string."""
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {raw}")


# ══════════════════════════════════════════════════════════
#  SEED DATA
# ══════════════════════════════════════════════════════════
ENROLLED_SEED = [
    {"name":"Adaeze Okonkwo",   "age":28,"sex":"Female","dob":"1997-03-14"},
    {"name":"Emeka Nwosu",      "age":34,"sex":"Male",  "dob":"1991-07-22"},
    {"name":"Fatima Al-Hassan", "age":22,"sex":"Female","dob":"2003-11-05"},
    {"name":"Chidi Obiora",     "age":45,"sex":"Male",  "dob":"1980-01-30"},
    {"name":"Ngozi Eze",        "age":31,"sex":"Female","dob":"1994-09-18"},
    {"name":"Babatunde Lawal",  "age":27,"sex":"Male",  "dob":"1998-06-07"},
    {"name":"Ifeoma Uche",      "age":38,"sex":"Female","dob":"1987-04-25"},
    {"name":"Olumide Adeyemi",  "age":52,"sex":"Male",  "dob":"1973-12-10"},
    {"name":"Chisom Nze",       "age":19,"sex":"Female","dob":"2006-08-03"},
    {"name":"Taiwo Ogundimu",   "age":41,"sex":"Male",  "dob":"1984-02-17"},
]
CONTROL_SEED = [
    {"name":"Kelechi Mbah",       "age":30,"sex":"Male",  "dob":"1995-05-09"},
    {"name":"Amina Suleiman",     "age":24,"sex":"Female","dob":"2001-10-21"},
    {"name":"Rotimi Akande",      "age":36,"sex":"Male",  "dob":"1989-03-16"},
    {"name":"Uchenna Obi",        "age":29,"sex":"Male",  "dob":"1996-07-04"},
    {"name":"Blessing Nwachukwu", "age":43,"sex":"Female","dob":"1982-11-28"},
    {"name":"Danladi Musa",       "age":25,"sex":"Male",  "dob":"2000-01-15"},
    {"name":"Orisa Efidi",        "age":33,"sex":"Female","dob":"1992-08-30"},
    {"name":"Segun Fasanya",      "age":48,"sex":"Male",  "dob":"1977-04-11"},
    {"name":"Chiamaka Igwe",      "age":21,"sex":"Female","dob":"2004-09-27"},
    {"name":"Nnamdi Okafor",      "age":55,"sex":"Male",  "dob":"1970-06-02"},
]

def make_synthetic_face(seed_val):
    rng = np.random.default_rng(seed_val)
    sz  = 200
    img = np.zeros((sz, sz, 3), dtype=np.uint8)
    for i in range(sz):
        v = int(180 + 40*(i/sz))
        img[i] = (v-30, v-15, v)
    sb = int(rng.integers(140, 215))
    sk = (max(0,sb-int(rng.integers(20,60))),
          max(0,sb-int(rng.integers(10,40))),
          max(0,sb-int(rng.integers(0,25))))
    cx, cy = sz//2, sz//2+10
    fw = int(rng.integers(56,72)); fh = int(rng.integers(74,88))
    cv2.ellipse(img,(cx,cy),(fw,fh),0,0,360,sk,-1)
    hc = (int(rng.integers(10,80)),int(rng.integers(8,60)),int(rng.integers(5,45)))
    cv2.ellipse(img,(cx,cy-18),(fw+6,fh-22),0,180,360,hc,-1)
    cv2.rectangle(img,(cx-fw-6,0),(cx+fw+6,cy-76),hc,-1)
    ex_off = int(rng.integers(22,32)); ey = cy-int(rng.integers(12,22))
    ec = (int(rng.integers(15,55)),)*3
    for ex in [cx-ex_off, cx+ex_off]:
        cv2.ellipse(img,(ex,ey),(9,6),0,0,360,(255,255,255),-1)
        cv2.circle(img,(ex,ey),4,ec,-1)
        cv2.circle(img,(ex,ey),2,(0,0,0),-1)
    cv2.ellipse(img,(cx,cy+10),(8,5),0,0,360,sk,-1)
    lc = tuple(max(0,s-55) for s in sk)
    cv2.ellipse(img,(cx,cy+32),(16,6),0,0,180,lc,-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def seed(db_inst, eng):
    if db_inst.stats()["total"] >= 20:
        print("✓ Database already seeded.")
        return
    print("Seeding database …")
    for i, p in enumerate(ENROLLED_SEED):
        pid = db_inst.add_person(p["name"],p["age"],p["sex"],p["dob"],1)
        for v in range(SAMPLES_NEEDED):
            gray = make_synthetic_face(100 + i + v * 1000)
            db_inst.save_face(pid, eng.preprocess(gray))
        print(f"  Enrolled: {p['name']}")
    for p in CONTROL_SEED:
        db_inst.add_person(p["name"],p["age"],p["sex"],p["dob"],0)
        print(f"  Control:  {p['name']}")
    labels, faces = db_inst.get_face_samples(True)
    eng.train(labels, faces)
    print(f"✓ Seeding complete. Model trained on {len(faces)} samples.")


# ══════════════════════════════════════════════════════════
#  GLOBALS
# ══════════════════════════════════════════════════════════
db     = Database()
engine = FaceEngine()
seed(db, engine)

enroll_session = {}
enroll_lock    = threading.Lock()


# ══════════════════════════════════════════════════════════
#  HTTP HANDLER
# ══════════════════════════════════════════════════════════
class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress console noise

    # ── Shared CORS headers ──────────────────────────────
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, data, code=200):
        try:
            body = json.dumps(data).encode("utf-8")
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode("utf-8")
            code = 500
        self.send_response(code)
        self.send_header("Content-Type",   "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, content: bytes):
        self.send_response(200)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self._cors()
        self.end_headers()
        self.wfile.write(content)

    # ── OPTIONS preflight ────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ── GET ──────────────────────────────────────────────
    def do_GET(self):
        try:
            self._handle_get()
        except Exception as e:
            print(f"[GET ERROR] {e}")
            self.send_json({"error": str(e)}, 500)

    def _handle_get(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path
        params = dict(urllib.parse.parse_qsl(parsed.query))

        if path in ("/", "/index.html"):
            self.send_html(HTML_PATH.read_bytes())

        elif path == "/api/stats":
            self.send_json(db.stats())

        elif path == "/api/persons":
            mode    = params.get("mode", "all")
            persons = db.get_all(mode)
            for p in persons:
                p["samples"] = db.sample_count(p["id"])
            self.send_json(persons)

        elif path == "/api/face":
            pid = int(params.get("id", 1))
            img = db.get_stored_face(pid)
            self.send_json({"img": img or ""})

        elif path == "/api/log":
            self.send_json(db.get_log(50))

        else:
            self.send_json({"error": "Not found"}, 404)

    # ── POST ─────────────────────────────────────────────
    def do_POST(self):
        try:
            self._handle_post()
        except Exception as e:
            print(f"[POST ERROR] {self.path} — {e}")
            import traceback; traceback.print_exc()
            self.send_json({"error": str(e)}, 500)

    def _handle_post(self):
        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            self.send_json({"error": f"Bad JSON: {e}"}, 400)
            return

        path = self.path

        # ── /api/enroll/start ────────────────────────────
        if path == "/api/enroll/start":
            name = str(body.get("name", "")).strip()
            age  = body.get("age")
            sex  = str(body.get("sex", "Male"))
            dob_raw = str(body.get("dob", "")).strip()

            if not name:
                self.send_json({"error": "Name is required"}, 400); return
            if not age:
                self.send_json({"error": "Age is required"}, 400); return
            if not dob_raw:
                self.send_json({"error": "Date of birth is required"}, 400); return

            try:
                age = int(age)
                if age < 1 or age > 130:
                    raise ValueError("Age out of range")
            except ValueError as e:
                self.send_json({"error": f"Invalid age: {e}"}, 400); return

            try:
                dob = parse_dob(dob_raw)
            except ValueError as e:
                self.send_json({"error": str(e)}, 400); return

            pid = db.add_person(name, age, sex, dob, enrolled=1)
            with enroll_lock:
                enroll_session.clear()
                enroll_session.update({"pid": pid, "name": name, "samples": []})

            print(f"  → Enrollment started: {name} (ID={pid}, DOB={dob})")
            self.send_json({"ok": True, "pid": pid, "name": name, "needed": SAMPLES_NEEDED})

        # ── /api/enroll/frame ────────────────────────────
        elif path == "/api/enroll/frame":
            frame_b64 = body.get("frame", "")
            if not frame_b64:
                self.send_json({"error": "No frame data"}, 400); return

            with enroll_lock:
                if not enroll_session:
                    self.send_json({"error": "No active enrollment session. Start enrollment first."}, 400); return
                pid      = enroll_session["pid"]
                collected = list(enroll_session["samples"])

            bgr = b64_to_bgr(frame_b64)
            if bgr is None:
                self.send_json({"error": "Cannot decode image frame"}, 400); return

            gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            boxes = engine.detect(gray)

            # Annotate frame
            out = bgr.copy()

            if not boxes:
                cv2.putText(out, "No face detected", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
                self.send_json({
                    "status":    "no_face",
                    "collected": len(collected),
                    "needed":    SAMPLES_NEEDED,
                    "preview":   bgr_to_b64(out)
                })
                return

            # Largest face
            box = max(boxes, key=lambda b: b[2]*b[3])
            x, y, w, h = box
            face_proc  = engine.preprocess(gray, bbox=box)

            with enroll_lock:
                enroll_session["samples"].append(face_proc)
                count = len(enroll_session["samples"])

            color = (0, 255, 100)
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
            cv2.putText(out, f"Sample {count}/{SAMPLES_NEEDED}",
                        (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if count >= SAMPLES_NEEDED:
                with enroll_lock:
                    all_samples = list(enroll_session["samples"])
                    enroll_session.clear()

                for s in all_samples:
                    db.save_face(pid, s)

                labels, faces = db.get_face_samples(enrolled_only=True)
                engine.train(labels, faces)

                face_b64 = arr_to_b64(all_samples[0])
                print(f"  ✓ Enrollment complete for ID={pid}")
                self.send_json({
                    "status":    "complete",
                    "pid":       pid,
                    "collected": count,
                    "needed":    SAMPLES_NEEDED,
                    "preview":   bgr_to_b64(out),
                    "face_img":  face_b64
                })
            else:
                self.send_json({
                    "status":    "collecting",
                    "collected": count,
                    "needed":    SAMPLES_NEEDED,
                    "preview":   bgr_to_b64(out)
                })

        # ── /api/enroll/cancel ───────────────────────────
        elif path == "/api/enroll/cancel":
            with enroll_lock:
                pid = enroll_session.get("pid")
                enroll_session.clear()
            if pid and db.sample_count(pid) == 0:
                db.delete_person(pid)
                print(f"  → Enrollment cancelled, removed person ID={pid}")
            self.send_json({"ok": True})

        # ── /api/recognize/frame ─────────────────────────
        elif path == "/api/recognize/frame":
            frame_b64 = body.get("frame", "")
            if not frame_b64:
                self.send_json({"error": "No frame data"}, 400); return

            bgr = b64_to_bgr(frame_b64)
            if bgr is None:
                self.send_json({"error": "Cannot decode image frame"}, 400); return

            gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            boxes = engine.detect(gray)
            out   = bgr.copy()

            if not boxes:
                cv2.putText(out, "No face detected", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
                self.send_json({
                    "status":  "no_face",
                    "results": [],
                    "preview": bgr_to_b64(out)
                })
                return

            results = []
            for box in boxes:
                face_proc   = engine.preprocess(gray, bbox=box)
                label, conf = engine.predict(face_proc)
                x, y, w, h  = [int(v) for v in box]

                if label is not None and conf is not None and conf < MIN_CONF:
                    matched = db.get_person(label)
                    if matched and matched["enrolled"]:
                        db.log_rec(label, round(conf, 2), "MATCH")
                        color = (0, 255, 100)
                        tag   = matched["name"]
                        results.append({
                            "result":     "MATCH",
                            "confidence": round(conf, 2),
                            "person":     matched,
                            "box":        [x, y, w, h]
                        })
                    else:
                        db.log_rec(None, round(conf, 2), "NO_MATCH")
                        color = (0, 60, 255)
                        tag   = "Unknown"
                        results.append({
                            "result":     "NO_MATCH",
                            "confidence": round(conf, 2),
                            "box":        [x, y, w, h]
                        })
                else:
                    conf_val = round(conf, 2) if conf is not None else 0
                    db.log_rec(None, conf_val, "NO_MATCH")
                    color = (0, 60, 255)
                    tag   = "Unknown"
                    results.append({
                        "result":     "NO_MATCH",
                        "confidence": conf_val,
                        "box":        [x, y, w, h]
                    })

                cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
                cv2.putText(out, tag, (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            self.send_json({
                "status":  "ok",
                "results": results,
                "preview": bgr_to_b64(out)
            })

        else:
            self.send_json({"error": f"Unknown endpoint: {path}"}, 404)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print(f"\n{'='*54}")
    print(f"  BIOMETRIC FACIAL RECOGNITION SYSTEM")
    print(f"  Real Webcam Edition")
    print(f"  Server: {url}")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*54}\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    server = http.server.HTTPServer(("localhost", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")