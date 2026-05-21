# Biometric Facial Recognition System
**Python | OpenCV LBPH | SQLite3 | Tkinter GUI**

---

## Overview

A cross-platform biometric facial recognition application that supports:

| Feature | Details |
|---|---|
| **Enrollment Mode** | Register new persons with name, age, sex, DOB, and face |
| **Face Recognition** | LBPH (Local Binary Pattern Histogram) via OpenCV |
| **Database** | SQLite3 — zero-configuration, embedded, portable |
| **GUI** | Tkinter (auto-detected; falls back to console if unavailable) |
| **Platform** | Windows, macOS, Linux — any Python 3.8+ environment |

---

## Requirements

```
Python 3.8+
opencv-python
opencv-contrib-python
numpy
Pillow             (for GUI image rendering — optional)
```

Install dependencies:
```bash
pip3 install -r requirements.txt
python3 server.py to run
```

---

## Running the Application

```bash
python biometric_app.py
```

- If Tkinter and Pillow are available → **GUI launches automatically**
- Otherwise → **Console (CLI) menu**

---

## Database Design

Three SQLite tables are created automatically in `biometric.db`:

### `persons`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-incremented unique ID |
| `name` | TEXT | Full name |
| `age` | INTEGER | Age in years |
| `sex` | TEXT | Male / Female / Other |
| `dob` | TEXT | Date of birth (YYYY-MM-DD) |
| `enrolled` | INTEGER | 1 = enrolled, 0 = control group |
| `face_label` | INTEGER | LBPH label (= person id) |
| `created_at` | TEXT | Enrollment timestamp |

### `face_samples`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Sample ID |
| `person_id` | INTEGER FK | Links to persons.id |
| `face_data` | BLOB | Serialised (pickled) 150×150 grayscale face array |
| `captured_at` | TEXT | Capture timestamp |

### `recognition_log`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Log entry ID |
| `person_id` | INTEGER FK | Matched person (NULL if unknown) |
| `confidence` | REAL | LBPH confidence score (lower = better) |
| `result` | TEXT | MATCH / NO_MATCH / UNKNOWN |
| `recognized_at` | TEXT | Timestamp |

---

## Pre-Seeded Data

### 10 Enrolled Persons
| # | Name | Age | Sex | DOB |
|---|---|---|---|---|
| 1 | Adaeze Okonkwo | 28 | Female | 1997-03-14 |
| 2 | Emeka Nwosu | 34 | Male | 1991-07-22 |
| 3 | Fatima Al-Hassan | 22 | Female | 2003-11-05 |
| 4 | Chidi Obiora | 45 | Male | 1980-01-30 |
| 5 | Ngozi Eze | 31 | Female | 1994-09-18 |
| 6 | Babatunde Lawal | 27 | Male | 1998-06-07 |
| 7 | Ifeoma Uche | 38 | Female | 1987-04-25 |
| 8 | Olumide Adeyemi | 52 | Male | 1973-12-10 |
| 9 | Chisom Nze | 19 | Female | 2006-08-03 |
| 10 | Taiwo Ogundimu | 41 | Male | 1984-02-17 |

### 10 Control Persons (NOT enrolled)
| # | Name | Age | Sex | DOB |
|---|---|---|---|---|
| 11 | Kelechi Mbah | 30 | Male | 1995-05-09 |
| 12 | Amina Suleiman | 24 | Female | 2001-10-21 |
| 13 | Rotimi Akande | 36 | Male | 1989-03-16 |
| 14 | Uchenna Obi | 29 | Male | 1996-07-04 |
| 15 | Blessing Nwachukwu | 43 | Female | 1982-11-28 |
| 16 | Danladi Musa | 25 | Male | 2000-01-15 |
| 17 | Orisa Efidi | 33 | Female | 1992-08-30 |
| 18 | Segun Fasanya | 48 | Male | 1977-04-11 |
| 19 | Chiamaka Igwe | 21 | Female | 2004-09-27 |
| 20 | Nnamdi Okafor | 55 | Male | 1970-06-02 |

---

## Algorithm: LBPH Face Recognition

**Local Binary Pattern Histogram (LBPH)** works by:
1. Dividing the face image into small grid cells
2. For each pixel, comparing its value to its 8 neighbours
3. Encoding the result as an 8-bit binary number → decimal label
4. Building a histogram of these labels for each cell
5. Concatenating all histograms into a single feature vector
6. At recognition time, computing **Chi-square distance** between histograms

A **confidence score < 80** means a match is accepted. Lower = more similar.

---

## Architecture

```
biometric_app.py
│
├── Database          — SQLite CRUD layer
│   ├── add_person()
│   ├── save_face_sample()
│   ├── get_face_samples()   ← returns labels + arrays for LBPH training
│   └── log_recognition()
│
├── FaceEngine        — OpenCV wrapper
│   ├── detect_faces()       ← Haar Cascade
│   ├── preprocess()         ← resize + histogram equalisation
│   ├── train()              ← LBPH fit, saves face_model.yml
│   ├── predict()            ← returns (label, confidence)
│   └── generate_synthetic_face()   ← deterministic NumPy face art
│
├── GUIApp (Tkinter)  — 4-tab interface
│   ├── Dashboard    — searchable persons table + face preview
│   ├── Enroll       — form to register new person
│   ├── Recognize    — pick & test enrolled vs control
│   └── Log          — colour-coded audit trail
│
└── ConsoleApp        — full CLI fallback, same features
```

---

## Notes for Students

1. **Synthetic faces** are used here since real webcam capture requires a physical camera. In a production system, replace `generate_synthetic_face()` calls with `cv2.VideoCapture(0)` to grab live frames.

2. **LBPH** is chosen because it:
   - Ships with `opencv-contrib-python` (no extra install)
   - Runs on CPU — no GPU required
   - Works on any OS
   - Handles lighting variation reasonably well

3. **Control group** persons are intentionally NOT enrolled — testing them demonstrates the system correctly **rejects** unknown individuals (False Acceptance Rate evaluation).

4. **Extending to real camera**:
   ```python
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = engine.detect_faces(gray)
   for (x, y, w, h) in faces:
       face = engine.preprocess(gray, bbox=(x, y, w, h))
       label, conf = engine.predict(face)
   cap.release()
   ```
