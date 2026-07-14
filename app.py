import csv
import io
import json
import os
import re
import smtplib
import sqlite3
import urllib.request
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from functools import lru_cache, wraps
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ATTENDANCE_DIR = BASE_DIR / "Attendance"
FACES_DIR = STATIC_DIR / "faces"
MODEL_PATH = STATIC_DIR / "face_recognition_model.pkl"
CASCADE_PATH = STATIC_DIR / "haarcascade_frontalface_default.xml"
DATABASE_PATH = BASE_DIR / "attendance.db"
DEFAULT_SECRET = os.getenv("ATTENDANCE_SECRET_KEY", "attendance-dev-secret")
app.secret_key = DEFAULT_SECRET
EMBEDDING_SIZE = (64, 64)

DEFAULT_SETTINGS = {
    "admin_username": "admin",
    "admin_password_hash": generate_password_hash(os.getenv("ATTENDANCE_ADMIN_PASSWORD", "admin123")),
    "unknown_threshold": "0.47",
    "cooldown_minutes": "10",
    "camera_index": "0",
    "frame_width": "960",
    "frame_height": "720",
    "scan_frames": "240",
    "capture_target_images": "40",
    "capture_stride": "8",
    "enable_liveness": "1",
    "liveness_motion_threshold": "12.0",
    "late_after": "09:30",
    "auto_send_alerts": "0",
    "alert_email_enabled": "0",
    "alert_email_host": "",
    "alert_email_port": "587",
    "alert_email_user": "",
    "alert_email_password": "",
    "alert_email_from": "",
    "alert_email_to": "",
    "whatsapp_webhook_enabled": "0",
    "whatsapp_webhook_url": "",
}

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
def load_face_detector():
    detector_factory = getattr(cv2, "CascadeClassifier", None)
    if detector_factory is None:
        return None
    detector = detector_factory(str(CASCADE_PATH))
    if hasattr(detector, "empty") and detector.empty():
        return None
    return detector


face_detector = load_face_detector()


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_directories():
    ATTENDANCE_DIR.mkdir(exist_ok=True)
    FACES_DIR.mkdir(parents=True, exist_ok=True)


def init_db():
    ensure_directories()
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                user_code TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );

            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                recognized_name TEXT NOT NULL,
                user_code TEXT,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                liveness_score REAL NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                camera_name TEXT NOT NULL DEFAULT 'Default Camera',
                created_at TEXT NOT NULL,
                session_month TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );

            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                model_dirty INTEGER NOT NULL DEFAULT 1,
                last_trained_at TEXT
            );
            """
        )
        connection.execute("INSERT OR IGNORE INTO training_state (id, model_dirty, last_trained_at) VALUES (1, 1, NULL)")
        for key, value in DEFAULT_SETTINGS.items():
            connection.execute(
                "INSERT OR IGNORE INTO app_settings (key, value) VALUES (?, ?)",
                (key, value),
            )
        connection.commit()
    sync_users_from_folders()


def query_all(query, params=()):
    with get_connection() as connection:
        return connection.execute(query, params).fetchall()


def query_one(query, params=()):
    with get_connection() as connection:
        return connection.execute(query, params).fetchone()


def execute_query(query, params=()):
    with get_connection() as connection:
        cursor = connection.execute(query, params)
        connection.commit()
        return cursor


def load_settings():
    rows = query_all("SELECT key, value FROM app_settings")
    settings = {row["key"]: row["value"] for row in rows}
    for key, value in DEFAULT_SETTINGS.items():
        settings.setdefault(key, value)
    return settings


def cast_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_float_setting(settings, key):
    return float(settings.get(key, DEFAULT_SETTINGS[key]))


def parse_int_setting(settings, key):
    return int(settings.get(key, DEFAULT_SETTINGS[key]))


def mark_model_dirty():
    execute_query("UPDATE training_state SET model_dirty = 1 WHERE id = 1")
    load_trained_model.cache_clear()


def mark_model_clean():
    execute_query(
        "UPDATE training_state SET model_dirty = 0, last_trained_at = ? WHERE id = 1",
        (datetime.now().isoformat(timespec="seconds"),),
    )
    load_trained_model.cache_clear()


def get_training_state():
    row = query_one("SELECT model_dirty, last_trained_at FROM training_state WHERE id = 1")
    return {
        "model_dirty": bool(row["model_dirty"]) if row else True,
        "last_trained_at": row["last_trained_at"] if row else None,
    }


def normalize_username(raw_name):
    cleaned = re.sub(r"\s+", " ", raw_name.strip())
    if not cleaned:
        raise ValueError("User name is required.")
    if not re.fullmatch(r"[A-Za-z0-9 ]{3,40}", cleaned):
        raise ValueError("Use 3-40 letters, numbers, or spaces for the user name.")
    return cleaned


def slug_name(name):
    return re.sub(r"\s+", "_", name.strip())


def user_folder_name(name, user_code):
    return f"{slug_name(name)}_{user_code}"


def user_folder_path(name, user_code):
    return FACES_DIR / user_folder_name(name, user_code)


def totalreg():
    row = query_one("SELECT COUNT(*) AS total FROM users")
    return int(row["total"]) if row else 0


def extract_faces(img):
    if img is None or img.size == 0 or face_detector is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))


def encode_face_image(image):
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, EMBEDDING_SIZE)
    descriptor = cv2.HOGDescriptor(
        _winSize=EMBEDDING_SIZE,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    embedding = descriptor.compute(normalized)
    if embedding is None:
        return None
    return embedding.flatten().astype(np.float32)


@lru_cache(maxsize=1)
def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model file was not found.")

    model_bundle = joblib.load(MODEL_PATH)
    encodings = np.asarray(model_bundle.get("encodings", []), dtype=np.float32)
    labels = model_bundle.get("labels", [])
    metadata = model_bundle.get("metadata", [])

    if encodings.size == 0 or not labels:
        raise ValueError("The trained face embedding bundle is empty. Add users and retrain the model.")

    return {
            "encodings": encodings,
            "labels": labels,
            "metadata": metadata,
            "embedding_model": model_bundle.get("embedding_model", "opencv_hog_descriptor"),
        }


def identify_face(face_image, threshold):
    model_bundle = load_trained_model()
    face_encoding = encode_face_image(face_image)
    if face_encoding is None:
        return {
            "label": None,
            "user_id": None,
            "user_code": None,
            "distance": 1.0,
            "confidence": 0.0,
            "status": "no_embedding",
        }

    known_encodings = model_bundle["encodings"]
    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    confidence = max(0.0, 1.0 - min(best_distance / max(threshold, 0.01), 1.0))

    if best_distance > threshold:
        return {
            "label": "Unknown",
            "user_id": None,
            "user_code": None,
            "distance": best_distance,
            "confidence": confidence,
            "status": "unknown",
        }

    metadata = model_bundle["metadata"][best_index] if model_bundle["metadata"] else {}
    return {
        "label": model_bundle["labels"][best_index],
        "user_id": metadata.get("user_id"),
        "user_code": metadata.get("user_code"),
        "distance": best_distance,
        "confidence": confidence,
        "status": "recognized",
    }


def sync_users_from_folders():
    for user_dir in FACES_DIR.iterdir():
        if not user_dir.is_dir() or "_" not in user_dir.name:
            continue
        name_part, user_code = user_dir.name.rsplit("_", 1)
        display_name = name_part.replace("_", " ")
        user_row = query_one("SELECT id FROM users WHERE user_code = ?", (user_code,))
        if user_row is None:
            execute_query(
                "INSERT INTO users (name, user_code, role, created_at) VALUES (?, ?, 'user', ?)",
                (display_name, user_code, datetime.now().isoformat(timespec="seconds")),
            )
            user_row = query_one("SELECT id FROM users WHERE user_code = ?", (user_code,))

        known_photo_rows = query_all("SELECT file_path FROM user_photos WHERE user_id = ?", (user_row["id"],))
        known_paths = {row["file_path"] for row in known_photo_rows}
        for image_path in sorted(user_dir.glob("*")):
            if image_path.is_file():
                relative_path = str(image_path.relative_to(STATIC_DIR)).replace("\\", "/")
                if relative_path not in known_paths:
                    execute_query(
                        "INSERT INTO user_photos (user_id, file_path, created_at) VALUES (?, ?, ?)",
                        (user_row["id"], relative_path, datetime.now().isoformat(timespec="seconds")),
                    )


def registered_users():
    rows = query_all(
        """
        SELECT u.id, u.name, u.user_code, u.role, u.created_at, COUNT(p.id) AS image_count,
               MIN(p.file_path) AS preview_path
        FROM users u
        LEFT JOIN user_photos p ON p.user_id = u.id
        GROUP BY u.id
        ORDER BY u.name COLLATE NOCASE
        """
    )
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "user_code": row["user_code"],
            "role": row["role"],
            "created_at": row["created_at"],
            "image_count": int(row["image_count"] or 0),
            "preview_path": row["preview_path"],
        }
        for row in rows
    ]


def train_model():
    encodings = []
    labels = []
    metadata = []

    users = query_all("SELECT id, name, user_code FROM users ORDER BY name COLLATE NOCASE")
    for user in users:
        folder = user_folder_path(user["name"], user["user_code"])
        if not folder.exists():
            continue
        for image_path in folder.iterdir():
            if not image_path.is_file():
                continue
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            embedding = encode_face_image(img)
            if embedding is None:
                continue
            encodings.append(embedding)
            labels.append(f"{slug_name(user['name'])}_{user['user_code']}")
            metadata.append(
                {
                    "user_id": user["id"],
                    "user_code": user["user_code"],
                    "name": user["name"],
                }
            )

    if not encodings:
        raise ValueError("No valid face embeddings could be created from the stored images.")

    joblib.dump(
        {
            "encodings": np.vstack(encodings),
            "labels": labels,
            "metadata": metadata,
            "embedding_model": "opencv_hog_descriptor",
        },
        MODEL_PATH,
    )
    mark_model_clean()


def ensure_model_ready():
    training_state = get_training_state()
    if not MODEL_PATH.exists() or training_state["model_dirty"]:
        train_model()
        return "Model retrained automatically before attendance."
    return None


def can_mark_attendance(user_id, cooldown_minutes):
    row = query_one(
        "SELECT created_at FROM attendance_logs WHERE user_id = ? AND status = 'recorded' ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    )
    if row is None:
        return True
    last_time = datetime.fromisoformat(row["created_at"])
    return datetime.now() - last_time >= timedelta(minutes=cooldown_minutes)


def record_attendance_event(user_id, recognized_name, user_code, status, confidence, liveness_score, notes=""):
    created_at = datetime.now().isoformat(timespec="seconds")
    execute_query(
        """
        INSERT INTO attendance_logs (
            user_id, recognized_name, user_code, status, confidence, liveness_score, notes, created_at, session_month
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            recognized_name,
            user_code,
            status,
            confidence,
            liveness_score,
            notes,
            created_at,
            created_at[:7],
        ),
    )


def mark_attendance(match_result, cooldown_minutes, liveness_score):
    label = match_result["label"]
    if label in {None, "Unknown"}:
        record_attendance_event(
            None,
            label or "Unknown",
            None,
            "unknown",
            match_result["confidence"],
            liveness_score,
            "Distance threshold rejected this face." if label == "Unknown" else "No face embedding generated.",
        )
        return "unknown"

    if not can_mark_attendance(match_result["user_id"], cooldown_minutes):
        record_attendance_event(
            match_result["user_id"],
            label,
            match_result["user_code"],
            "cooldown_skipped",
            match_result["confidence"],
            liveness_score,
            "Duplicate attendance blocked by cooldown window.",
        )
        return "cooldown"

    record_attendance_event(
        match_result["user_id"],
        label,
        match_result["user_code"],
        "recorded",
        match_result["confidence"],
        liveness_score,
    )
    return "recorded"


def compute_liveness_score(previous_crop, current_crop):
    if previous_crop is None or current_crop is None:
        return 100.0
    prev_gray = cv2.cvtColor(cv2.resize(previous_crop, (96, 96)), cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(cv2.resize(current_crop, (96, 96)), cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(prev_gray, curr_gray)))


def attendance_summary():
    today_prefix = datetime.now().date().isoformat()
    rows = query_all(
        """
        SELECT status, COUNT(*) AS total
        FROM attendance_logs
        WHERE created_at LIKE ?
        GROUP BY status
        """,
        (f"{today_prefix}%",),
    )
    counts = {row["status"]: int(row["total"]) for row in rows}
    last_seen_row = query_one(
        "SELECT recognized_name, created_at FROM attendance_logs ORDER BY created_at DESC LIMIT 1"
    )
    return {
        "total_records": counts.get("recorded", 0),
        "unique_people": unique_people_today(),
        "last_seen": last_seen_row["created_at"].replace("T", " ") if last_seen_row else "No attendance yet",
        "unknown_events": counts.get("unknown", 0),
        "cooldown_skips": counts.get("cooldown_skipped", 0),
    }


def unique_people_today():
    today_prefix = datetime.now().date().isoformat()
    row = query_one(
        """
        SELECT COUNT(DISTINCT user_id) AS total
        FROM attendance_logs
        WHERE created_at LIKE ? AND status = 'recorded' AND user_id IS NOT NULL
        """,
        (f"{today_prefix}%",),
    )
    return int(row["total"] or 0) if row else 0


def attendance_records(limit=50, filters=None):
    filters = filters or {}
    clauses = []
    params = []

    if filters.get("start_date"):
        clauses.append("date(created_at) >= date(?)")
        params.append(filters["start_date"])
    if filters.get("end_date"):
        clauses.append("date(created_at) <= date(?)")
        params.append(filters["end_date"])
    if filters.get("user_code"):
        clauses.append("user_code = ?")
        params.append(filters["user_code"])
    if filters.get("month"):
        clauses.append("session_month = ?")
        params.append(filters["month"])
    if filters.get("search"):
        clauses.append("recognized_name LIKE ?")
        params.append(f"%{filters['search']}%")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = (
        f"SELECT * FROM attendance_logs {where} ORDER BY created_at DESC LIMIT ?"
    )
    params.append(limit)
    rows = query_all(query, params)
    return [dict(row) for row in rows]


def analytics_snapshot():
    month_key = datetime.now().strftime("%Y-%m")
    daily = query_all(
        """
        SELECT date(created_at) AS day, COUNT(*) AS total
        FROM attendance_logs
        WHERE status = 'recorded'
        GROUP BY date(created_at)
        ORDER BY day DESC
        LIMIT 7
        """
    )
    top_users = query_all(
        """
        SELECT recognized_name, COUNT(*) AS total
        FROM attendance_logs
        WHERE status = 'recorded' AND user_id IS NOT NULL
        GROUP BY recognized_name
        ORDER BY total DESC, recognized_name ASC
        LIMIT 5
        """
    )
    late_after = load_settings().get("late_after", "09:30")
    late_count_row = query_one(
        """
        SELECT COUNT(*) AS total
        FROM attendance_logs
        WHERE status = 'recorded' AND time(created_at) > time(?)
        """,
        (late_after,),
    )
    month_count_row = query_one(
        "SELECT COUNT(*) AS total FROM attendance_logs WHERE session_month = ? AND status = 'recorded'",
        (month_key,),
    )
    return {
        "daily_totals": [dict(row) for row in daily],
        "top_users": [dict(row) for row in top_users],
        "late_count": int(late_count_row["total"] or 0) if late_count_row else 0,
        "month_total": int(month_count_row["total"] or 0) if month_count_row else 0,
    }


def admin_required(handler):
    @wraps(handler)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("home", message="Admin login required."))
        return handler(*args, **kwargs)

    return wrapper


def send_summary_notifications(summary_text):
    settings = load_settings()
    results = []

    if cast_bool(settings.get("alert_email_enabled")):
        try:
            send_email_summary(settings, summary_text)
            results.append("Email summary sent.")
        except Exception as exc:  # pragma: no cover - runtime integration
            results.append(f"Email failed: {exc}")

    if cast_bool(settings.get("whatsapp_webhook_enabled")):
        try:
            send_whatsapp_summary(settings, summary_text)
            results.append("WhatsApp webhook summary sent.")
        except Exception as exc:  # pragma: no cover - runtime integration
            results.append(f"WhatsApp webhook failed: {exc}")

    if not results:
        results.append("No alert channels are enabled.")
    return results


def send_email_summary(settings, summary_text):
    message = EmailMessage()
    message["Subject"] = "Attendance Summary"
    message["From"] = settings["alert_email_from"] or settings["alert_email_user"]
    message["To"] = settings["alert_email_to"]
    message.set_content(summary_text)

    with smtplib.SMTP(settings["alert_email_host"], int(settings["alert_email_port"])) as smtp:
        smtp.starttls()
        if settings["alert_email_user"]:
            smtp.login(settings["alert_email_user"], settings["alert_email_password"])
        smtp.send_message(message)


def send_whatsapp_summary(settings, summary_text):
    payload = json.dumps({"message": summary_text}).encode("utf-8")
    request_obj = urllib.request.Request(
        settings["whatsapp_webhook_url"],
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request_obj, timeout=10):
        return


def attendance_export_rows(filters):
    rows = attendance_records(limit=5000, filters=filters)
    return [
        {
            "Name": row["recognized_name"],
            "User ID": row["user_code"] or "",
            "Status": row["status"],
            "Confidence": f"{row['confidence']:.2f}",
            "Liveness Score": f"{row['liveness_score']:.2f}",
            "Time": row["created_at"].replace("T", " "),
            "Notes": row["notes"],
        }
        for row in rows
    ]


def csv_download(filename, rows, fieldnames):
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    return send_file(
        io.BytesIO(csv_bytes),
        as_attachment=True,
        download_name=filename,
        mimetype="text/csv",
    )


def render_home(message=None):
    settings = load_settings()
    records = attendance_records(
        limit=100,
        filters={
            "start_date": request.args.get("start_date", ""),
            "end_date": request.args.get("end_date", ""),
            "user_code": request.args.get("user_code", ""),
            "month": request.args.get("month", ""),
            "search": request.args.get("search", "").strip(),
        },
    )
    return render_template(
        "home.html",
        datetoday2=datetoday2,
        mess=message or request.args.get("message", ""),
        summary=attendance_summary(),
        registered_users=registered_users(),
        model_ready=MODEL_PATH.exists() and not get_training_state()["model_dirty"],
        settings=settings,
        totalreg=totalreg(),
        records=records,
        analytics=analytics_snapshot(),
        training_state=get_training_state(),
        is_admin=session.get("is_admin", False),
        admin_username=settings.get("admin_username", "admin"),
        filter_values={
            "start_date": request.args.get("start_date", ""),
            "end_date": request.args.get("end_date", ""),
            "user_code": request.args.get("user_code", ""),
            "month": request.args.get("month", ""),
            "search": request.args.get("search", ""),
        },
    )


@app.route("/")
def home():
    return render_home()


@app.route("/login", methods=["POST"])
def login():
    settings = load_settings()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    valid_username = username == settings.get("admin_username", "admin")
    valid_password = check_password_hash(settings["admin_password_hash"], password)
    if not (valid_username and valid_password):
        return render_home("Invalid admin credentials.")

    session["is_admin"] = True
    return redirect(url_for("home", message="Admin session started."))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home", message="Admin session closed."))


@app.route("/start", methods=["GET"])
def start():
    if face_detector is None:
        return render_home("Face detection model is missing. Please verify the cascade XML file in the static folder.")

    settings = load_settings()
    retrain_message = None
    try:
        retrain_message = ensure_model_ready()
    except ValueError as exc:
        return render_home(str(exc))

    cap = cv2.VideoCapture(parse_int_setting(settings, "camera_index"))
    if not cap.isOpened():
        return render_home("Could not access the webcam. Check your camera permissions and try again.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, parse_int_setting(settings, "frame_width"))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, parse_int_setting(settings, "frame_height"))

    max_frames = parse_int_setting(settings, "scan_frames")
    threshold = parse_float_setting(settings, "unknown_threshold")
    cooldown_minutes = parse_int_setting(settings, "cooldown_minutes")
    liveness_threshold = parse_float_setting(settings, "liveness_motion_threshold")
    enable_liveness = cast_bool(settings.get("enable_liveness"))
    previous_face_crops = {}
    session_counts = {"recorded": 0, "cooldown": 0, "unknown": 0, "liveness": 0}

    try:
        for frame_index in range(max_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                return render_home("Webcam frames could not be read. Please reconnect the camera and try again.")

            faces = sorted(extract_faces(frame), key=lambda face: face[0])
            for idx, (x, y, w, h) in enumerate(faces):
                crop = frame[y : y + h, x : x + w]
                liveness_score = compute_liveness_score(previous_face_crops.get(idx), crop)
                previous_face_crops[idx] = crop.copy()

                if enable_liveness and liveness_score < liveness_threshold:
                    record_attendance_event(
                        None,
                        "Liveness blocked",
                        None,
                        "liveness_blocked",
                        0.0,
                        liveness_score,
                        "Low motion score blocked attendance.",
                    )
                    session_counts["liveness"] += 1
                    label = f"Spoof? {liveness_score:.1f}"
                    color = (56, 84, 255)
                else:
                    match_result = identify_face(crop, threshold)
                    outcome = mark_attendance(match_result, cooldown_minutes, liveness_score)
                    session_counts[outcome if outcome in session_counts else "unknown"] += 1
                    label = f"{match_result['label']} ({match_result['confidence']:.2f})"
                    color = (105, 210, 166) if outcome == "recorded" else (255, 138, 76)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x, max(24, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"Recorded {session_counts['recorded']} | Unknown {session_counts['unknown']} | Cooldown {session_counts['cooldown']}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) == 27:
                break
            if frame_index + 1 >= max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    summary_text = (
        f"Attendance finished. Recorded: {session_counts['recorded']}, Unknown: {session_counts['unknown']}, "
        f"Cooldown: {session_counts['cooldown']}, Liveness blocked: {session_counts['liveness']}."
    )
    if cast_bool(settings.get("auto_send_alerts")):
        summary_text += " " + " ".join(send_summary_notifications(summary_text))
    if retrain_message:
        summary_text = f"{retrain_message} {summary_text}"
    return render_home(summary_text)


@app.route("/add", methods=["POST"])
@admin_required
def add():
    raw_username = request.form.get("newusername", "")
    user_code = request.form.get("newuserid", "").strip()
    settings = load_settings()

    if not user_code.isdigit():
        return render_home("User ID must be numeric.")

    try:
        normalized_username = normalize_username(raw_username)
    except ValueError as exc:
        return render_home(str(exc))

    if face_detector is None:
        return render_home("Face detection model is missing. Please verify the cascade XML file in the static folder.")

    existing_user = query_one("SELECT id FROM users WHERE user_code = ?", (user_code,))
    if existing_user is not None:
        return render_home("A user with that ID already exists.")

    execute_query(
        "INSERT INTO users (name, user_code, role, created_at) VALUES (?, ?, 'user', ?)",
        (normalized_username, user_code, datetime.now().isoformat(timespec="seconds")),
    )
    user = query_one("SELECT id FROM users WHERE user_code = ?", (user_code,))
    user_folder = user_folder_path(normalized_username, user_code)
    user_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(parse_int_setting(settings, "camera_index"))
    if not cap.isOpened():
        return render_home("Could not access the webcam. Check your camera permissions and try again.")

    capture_target_images = parse_int_setting(settings, "capture_target_images")
    capture_stride = parse_int_setting(settings, "capture_stride")
    captured_images = 0
    frame_counter = 0

    try:
        while captured_images < capture_target_images:
            ret, frame = cap.read()
            if not ret or frame is None:
                return render_home("Webcam frames could not be read while capturing the new user.")

            faces = extract_faces(frame)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 138, 76), 2)
                cv2.putText(
                    frame,
                    f"Images {captured_images}/{capture_target_images}",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 138, 76),
                    2,
                    cv2.LINE_AA,
                )
                if frame_counter % capture_stride == 0:
                    image_name = f"{slug_name(normalized_username)}_{captured_images}.jpg"
                    image_path = user_folder / image_name
                    crop = frame[y : y + h, x : x + w]
                    cv2.imwrite(str(image_path), crop)
                    execute_query(
                        "INSERT INTO user_photos (user_id, file_path, created_at) VALUES (?, ?, ?)",
                        (
                            user["id"],
                            str(image_path.relative_to(STATIC_DIR)).replace("\\", "/"),
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                    )
                    captured_images += 1
                    if captured_images >= capture_target_images:
                        break
                frame_counter += 1

            cv2.imshow("Adding New User", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured_images == 0:
        execute_query("DELETE FROM users WHERE id = ?", (user["id"],))
        return render_home("No face images were captured. Please keep the face visible to the webcam and try again.")

    mark_model_dirty()
    return render_home(f"User {normalized_username} was added. {captured_images} images captured. Model marked for retraining.")


@app.route("/users/<int:user_id>/upload", methods=["POST"])
@admin_required
def upload_user_photos(user_id):
    user = query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if user is None:
        return render_home("User not found.")

    files = request.files.getlist("photos")
    if not files or not any(file.filename for file in files):
        return render_home("Choose at least one image file.")

    user_folder = user_folder_path(user["name"], user["user_code"])
    user_folder.mkdir(parents=True, exist_ok=True)

    saved = 0
    for file in files:
        if not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower() or ".jpg"
        file_name = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}_{saved}{suffix}"
        image_path = user_folder / file_name
        file.save(image_path)
        execute_query(
            "INSERT INTO user_photos (user_id, file_path, created_at) VALUES (?, ?, ?)",
            (
                user_id,
                str(image_path.relative_to(STATIC_DIR)).replace("\\", "/"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        saved += 1

    if saved == 0:
        return render_home("No files were uploaded.")

    mark_model_dirty()
    return render_home(f"Uploaded {saved} photo(s) for {user['name']}. Model marked for retraining.")


@app.route("/users/<int:user_id>/reset-photos", methods=["POST"])
@admin_required
def reset_user_photos(user_id):
    user = query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if user is None:
        return render_home("User not found.")

    folder = user_folder_path(user["name"], user["user_code"])
    if folder.exists():
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()
    execute_query("DELETE FROM user_photos WHERE user_id = ?", (user_id,))
    mark_model_dirty()
    return render_home(f"Photos cleared for {user['name']}. Add or upload new images and retrain.")


@app.route("/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def delete_user(user_id):
    user = query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if user is None:
        return render_home("User not found.")

    folder = user_folder_path(user["name"], user["user_code"])
    if folder.exists():
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()
        folder.rmdir()
    execute_query("DELETE FROM user_photos WHERE user_id = ?", (user_id,))
    execute_query("DELETE FROM attendance_logs WHERE user_id = ?", (user_id,))
    execute_query("DELETE FROM users WHERE id = ?", (user_id,))
    mark_model_dirty()
    return render_home(f"User {user['name']} was deleted.")


@app.route("/retrain", methods=["POST"])
@admin_required
def retrain():
    try:
        train_model()
    except ValueError as exc:
        return render_home(str(exc))
    return render_home("Model retrained successfully.")


@app.route("/settings", methods=["POST"])
@admin_required
def save_settings():
    settings = load_settings()
    updates = {
        "admin_username": request.form.get("admin_username", settings["admin_username"]).strip() or settings["admin_username"],
        "unknown_threshold": request.form.get("unknown_threshold", settings["unknown_threshold"]).strip(),
        "cooldown_minutes": request.form.get("cooldown_minutes", settings["cooldown_minutes"]).strip(),
        "camera_index": request.form.get("camera_index", settings["camera_index"]).strip(),
        "frame_width": request.form.get("frame_width", settings["frame_width"]).strip(),
        "frame_height": request.form.get("frame_height", settings["frame_height"]).strip(),
        "scan_frames": request.form.get("scan_frames", settings["scan_frames"]).strip(),
        "capture_target_images": request.form.get("capture_target_images", settings["capture_target_images"]).strip(),
        "capture_stride": request.form.get("capture_stride", settings["capture_stride"]).strip(),
        "enable_liveness": "1" if request.form.get("enable_liveness") else "0",
        "liveness_motion_threshold": request.form.get("liveness_motion_threshold", settings["liveness_motion_threshold"]).strip(),
        "late_after": request.form.get("late_after", settings["late_after"]).strip(),
        "auto_send_alerts": "1" if request.form.get("auto_send_alerts") else "0",
        "alert_email_enabled": "1" if request.form.get("alert_email_enabled") else "0",
        "alert_email_host": request.form.get("alert_email_host", settings["alert_email_host"]).strip(),
        "alert_email_port": request.form.get("alert_email_port", settings["alert_email_port"]).strip(),
        "alert_email_user": request.form.get("alert_email_user", settings["alert_email_user"]).strip(),
        "alert_email_password": request.form.get("alert_email_password", settings["alert_email_password"]).strip(),
        "alert_email_from": request.form.get("alert_email_from", settings["alert_email_from"]).strip(),
        "alert_email_to": request.form.get("alert_email_to", settings["alert_email_to"]).strip(),
        "whatsapp_webhook_enabled": "1" if request.form.get("whatsapp_webhook_enabled") else "0",
        "whatsapp_webhook_url": request.form.get("whatsapp_webhook_url", settings["whatsapp_webhook_url"]).strip(),
    }

    new_password = request.form.get("admin_password", "").strip()
    if new_password:
        updates["admin_password_hash"] = generate_password_hash(new_password)

    for key, value in updates.items():
        execute_query("REPLACE INTO app_settings (key, value) VALUES (?, ?)", (key, value))
    return render_home("Settings updated successfully.")


@app.route("/attendance/download")
def download_attendance():
    filters = {
        "start_date": request.args.get("start_date", ""),
        "end_date": request.args.get("end_date", ""),
        "user_code": request.args.get("user_code", ""),
        "month": request.args.get("month", ""),
        "search": request.args.get("search", "").strip(),
    }
    rows = attendance_export_rows(filters)
    return csv_download("attendance_report.csv", rows, list(rows[0].keys()) if rows else ["Name", "User ID", "Status", "Confidence", "Liveness Score", "Time", "Notes"])


@app.route("/users/download")
def download_users():
    rows = [
        {
            "Name": user["name"],
            "User ID": user["user_code"],
            "Role": user["role"],
            "Images": user["image_count"],
            "Created At": user["created_at"],
        }
        for user in registered_users()
    ]
    return csv_download("registered_users.csv", rows, ["Name", "User ID", "Role", "Images", "Created At"])


@app.route("/attendance/<int:record_id>/update", methods=["POST"])
@admin_required
def update_record(record_id):
    recognized_name = request.form.get("recognized_name", "").strip() or "Unknown"
    status = request.form.get("status", "recorded").strip()
    notes = request.form.get("notes", "").strip()
    execute_query(
        "UPDATE attendance_logs SET recognized_name = ?, status = ?, notes = ? WHERE id = ?",
        (recognized_name, status, notes, record_id),
    )
    return render_home("Attendance record updated.")


@app.route("/attendance/<int:record_id>/delete", methods=["POST"])
@admin_required
def delete_record(record_id):
    execute_query("DELETE FROM attendance_logs WHERE id = ?", (record_id,))
    return render_home("Attendance record deleted.")


@app.route("/alerts/send", methods=["POST"])
@admin_required
def send_alerts():
    summary = attendance_summary()
    summary_text = (
        f"Attendance summary for {datetoday2}: recorded={summary['total_records']}, unique={summary['unique_people']}, "
        f"unknown={summary['unknown_events']}, cooldown={summary['cooldown_skips']}."
    )
    results = send_summary_notifications(summary_text)
    return render_home(" ".join(results))


init_db()


if __name__ == "__main__":
    app.run(debug=True)
