from datetime import date, datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
ATTENDANCE_DIR = BASE_DIR / "Attendance"
FACES_DIR = STATIC_DIR / "faces"
MODEL_PATH = STATIC_DIR / "face_recognition_model.pkl"
CASCADE_PATH = STATIC_DIR / "haarcascade_frontalface_default.xml"

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


def extract_attendance():
    attendance_path = ensure_attendance_file()
    try:
        df = pd.read_csv(attendance_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["Name", "Roll", "Time"])

    if df.empty:
        return [], [], [], 0

    names = df["Name"].fillna("").tolist()
    rolls = df["Roll"].fillna("").tolist()
    times = df["Time"].fillna("").tolist()
    return names, rolls, times, len(df)


def add_attendance(name):
    parts = name.split("_", 1)
    if len(parts) != 2:
        return

    username, userid = parts
    current_time = datetime.now().strftime("%H:%M:%S")
    attendance_path = ensure_attendance_file()
    df = pd.read_csv(attendance_path)

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
    )


@app.route("/")
def home():
    return render_home()


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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                face = cv2.resize(frame[y : y + h, x : x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(
                    frame,
                    identified_person,
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 20),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return render_home()


@app.route("/add", methods=["GET", "POST"])
def add():
    newusername = request.form.get("newusername", "").strip()
    newuserid = request.form.get("newuserid", "").strip()

    if not newusername or not newuserid:
        return render_home("Both user name and user ID are required.")

    if face_detector.empty():
        return render_home("Face detection model is missing. Please verify the cascade XML file in the static folder.")

    userimagefolder = FACES_DIR / f"{newusername}_{newuserid}"
    userimagefolder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_home("Could not access the webcam. Check your camera permissions and try again.")

    captured_images = 0
    frame_counter = 0

    try:
        while captured_images < 50:
            ret, frame = cap.read()
            if not ret or frame is None:
                return render_home("Webcam frames could not be read while capturing the new user.")

            faces = extract_faces(frame)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(
                    frame,
                    f"Images Captured: {captured_images}/50",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 20),
                    2,
                    cv2.LINE_AA,
                )
                if frame_counter % 10 == 0:
                    image_name = f"{newusername}_{captured_images}.jpg"
                    cv2.imwrite(str(userimagefolder / image_name), frame[y : y + h, x : x + w])
                    captured_images += 1
                    if captured_images >= 50:
                        break
                frame_counter += 1

            cv2.imshow("Adding new User", frame)
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

    return render_home(f"User {newusername} was added successfully with {captured_images} captured images.")


if __name__ == "__main__":
    app.run(debug=True)
