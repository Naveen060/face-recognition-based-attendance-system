import re
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
from sklearn.neighbors import KNeighborsClassifier


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ATTENDANCE_DIR = BASE_DIR / "Attendance"
FACES_DIR = STATIC_DIR / "faces"
MODEL_PATH = STATIC_DIR / "face_recognition_model.pkl"
CASCADE_PATH = STATIC_DIR / "haarcascade_frontalface_default.xml"
MIN_IMAGES_PER_USER = 50

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


def attendance_file_path():
    return ATTENDANCE_DIR / f"Attendance-{datetoday}.csv"


def ensure_directories():
    ATTENDANCE_DIR.mkdir(exist_ok=True)
    FACES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_attendance_file():
    attendance_path = attendance_file_path()
    if not attendance_path.exists():
        attendance_path.write_text("Name,Roll,Time\n", encoding="utf-8")
    return attendance_path


ensure_directories()
ensure_attendance_file()

face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))


def totalreg():
    if not FACES_DIR.exists():
        return 0
    return len([entry for entry in FACES_DIR.iterdir() if entry.is_dir()])


def extract_faces(img):
    if img is None or img.size == 0 or face_detector.empty():
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.3, 5)


def identify_face(facearray):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model file was not found.")
    model = joblib.load(MODEL_PATH)
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []

    for user_dir in FACES_DIR.iterdir():
        if not user_dir.is_dir():
            continue
        for image_path in user_dir.iterdir():
            if not image_path.is_file():
                continue
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user_dir.name)

    if not faces:
        raise ValueError("No face images are available to train the model.")

    n_neighbors = min(5, len(faces))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(np.array(faces), labels)
    joblib.dump(knn, MODEL_PATH)


def attendance_dataframe():
    attendance_path = ensure_attendance_file()
    try:
        df = pd.read_csv(attendance_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["Name", "Roll", "Time"])
    return df.fillna("")


def extract_attendance():
    df = attendance_dataframe()
    if df.empty:
        return [], [], [], 0
    return df["Name"].tolist(), df["Roll"].tolist(), df["Time"].tolist(), len(df)


def attendance_summary():
    df = attendance_dataframe()
    total_records = len(df)
    unique_people = df["Roll"].nunique() if not df.empty else 0
    last_seen = df["Time"].iloc[-1] if not df.empty else "No attendance yet"
    return {
        "total_records": total_records,
        "unique_people": int(unique_people),
        "last_seen": last_seen,
    }


def registered_users():
    users = []
    for user_dir in sorted(FACES_DIR.iterdir()):
        if not user_dir.is_dir() or "_" not in user_dir.name:
            continue
        name, user_id = user_dir.name.rsplit("_", 1)
        image_count = len([file for file in user_dir.iterdir() if file.is_file()])
        users.append(
            {
                "name": name.replace("_", " "),
                "user_id": user_id,
                "image_count": image_count,
            }
        )
    return users


def normalize_username(raw_name):
    cleaned = re.sub(r"\s+", " ", raw_name.strip())
    if not cleaned:
        raise ValueError("User name is required.")
    if not re.fullmatch(r"[A-Za-z0-9 ]{3,40}", cleaned):
        raise ValueError("Use 3-40 letters, numbers, or spaces for the user name.")
    return cleaned.replace(" ", "_")


def add_attendance(name):
    parts = name.split("_", 1)
    if len(parts) != 2:
        return

    username, userid = parts
    current_time = datetime.now().strftime("%H:%M:%S")
    attendance_path = ensure_attendance_file()
    df = attendance_dataframe()

    try:
        userid_int = int(userid)
    except ValueError:
        return

    if userid_int not in pd.to_numeric(df["Roll"], errors="coerce").dropna().astype(int).tolist():
        with attendance_path.open("a", encoding="utf-8") as attendance_file:
            attendance_file.write(f"{username},{userid_int},{current_time}\n")


def render_home(message=None):
    names, rolls, times, total_rows = extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        times=times,
        l=total_rows,
        totalreg=totalreg(),
        datetoday2=datetoday2,
        mess=message,
        summary=attendance_summary(),
        registered_users=registered_users(),
        model_ready=MODEL_PATH.exists(),
        min_images_per_user=MIN_IMAGES_PER_USER,
    )


@app.route("/")
def home():
    return render_home()


@app.route("/attendance/download")
def download_attendance():
    csv_bytes = attendance_file_path().read_bytes()
    return send_file(
        BytesIO(csv_bytes),
        as_attachment=True,
        download_name=attendance_file_path().name,
        mimetype="text/csv",
    )


@app.route("/users/download")
def download_users():
    df = pd.DataFrame(registered_users())
    if df.empty:
        df = pd.DataFrame(columns=["name", "user_id", "image_count"])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return send_file(
        BytesIO(csv_bytes),
        as_attachment=True,
        download_name="registered_users.csv",
        mimetype="text/csv",
    )


@app.route("/start", methods=["GET"])
def start():
    if face_detector.empty():
        return render_home("Face detection model is missing. Please verify the cascade XML file in the static folder.")

    if not MODEL_PATH.exists():
        return render_home("There is no trained model yet. Add at least one user before taking attendance.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_home("Could not access the webcam. Check your camera permissions and try again.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                return render_home("Webcam frames could not be read. Please reconnect the camera and try again.")

            faces = extract_faces(frame)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 138, 76), 2)
                face = cv2.resize(frame[y : y + h, x : x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(
                    frame,
                    identified_person,
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 138, 76),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return render_home("Attendance session finished successfully.")


@app.route("/add", methods=["POST"])
def add():
    raw_username = request.form.get("newusername", "")
    newuserid = request.form.get("newuserid", "").strip()

    if not newuserid.isdigit():
        return render_home("User ID must be numeric.")

    try:
        normalized_username = normalize_username(raw_username)
    except ValueError as exc:
        return render_home(str(exc))

    if face_detector.empty():
        return render_home("Face detection model is missing. Please verify the cascade XML file in the static folder.")

    userimagefolder = FACES_DIR / f"{normalized_username}_{newuserid}"
    userimagefolder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_home("Could not access the webcam. Check your camera permissions and try again.")

    captured_images = 0
    frame_counter = 0

    try:
        while captured_images < MIN_IMAGES_PER_USER:
            ret, frame = cap.read()
            if not ret or frame is None:
                return render_home("Webcam frames could not be read while capturing the new user.")

            faces = extract_faces(frame)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 138, 76), 2)
                cv2.putText(
                    frame,
                    f"Images Captured: {captured_images}/{MIN_IMAGES_PER_USER}",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 138, 76),
                    2,
                    cv2.LINE_AA,
                )
                if frame_counter % 10 == 0:
                    image_name = f"{normalized_username}_{captured_images}.jpg"
                    cv2.imwrite(str(userimagefolder / image_name), frame[y : y + h, x : x + w])
                    captured_images += 1
                    if captured_images >= MIN_IMAGES_PER_USER:
                        break
                frame_counter += 1

            cv2.imshow("Adding New User", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured_images == 0:
        return render_home("No face images were captured. Please keep the face visible to the webcam and try again.")

    try:
        train_model()
    except ValueError as exc:
        return render_home(str(exc))

    return render_home(
        f"User {raw_username.strip()} was added successfully with {captured_images} captured images."
    )


if __name__ == "__main__":
    app.run(debug=True)
