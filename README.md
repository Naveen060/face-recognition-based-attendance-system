# Face Recognition Based Attendance System

This repository contains a Flask-based attendance application that captures webcam images, detects faces with OpenCV, identifies registered users with a KNN classifier, and stores daily attendance records in CSV format.

## Current Project Structure

```text
face-recognition-based-attendance-system/
|-- app.py
|-- requirements.txt
|-- README.md
|-- templates/
|   `-- home.html
`-- static/
    `-- haarcascade_frontalface_default.xml
```

## Features

- Register a new user by capturing face images from the webcam.
- Train a K-nearest neighbors face recognition model from stored images.
- Take attendance in real time using webcam face detection.
- Save attendance for each day in `Attendance/Attendance-<date>.csv`.
- Display attendance records in the Flask web interface.

## Changes Made In This Recovery Pass

The original repository had a working prototype, but it also had several issues that made it fragile on a fresh machine. The following changes were applied:

1. Removed webcam initialization at import time.
   The original code opened `cv2.VideoCapture(0)` as soon as `app.py` was imported, which can lock the camera before a route is even used.

2. Added path-safe file handling.
   The app now uses `pathlib.Path` so file operations are based on the repository location instead of the current shell directory.

3. Added automatic directory and attendance-file creation.
   The `Attendance/` and `static/faces/` folders are created when needed, and the daily CSV file is initialized safely.

4. Improved empty and invalid state handling.
   The app now checks for:
   - missing cascade XML file
   - missing trained model
   - unavailable webcam
   - unreadable webcam frames
   - empty training datasets
   - malformed predicted labels

5. Fixed face detection checks.
   The old implementation compared the result of `detectMultiScale` against `()`, which is brittle. The app now uses a normal length check.

6. Made model training more robust.
   The KNN neighbor count now adapts to the available number of training samples instead of always forcing `n_neighbors=5`.

7. Added reusable `render_home()` flow.
   Rendering is now centralized so success and error messages are handled consistently.

8. Added a dependency manifest.
   A `requirements.txt` file is now included for environment setup.

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Run the Flask application.

```powershell
python app.py
```

4. Open the local URL shown by Flask in your browser.

## Usage

1. Open the home page.
2. Add a new user with a name and numeric ID.
3. Allow the webcam to capture face images.
4. After training completes, click `Take Attendance`.
5. Press `Esc` in the OpenCV camera window to stop capture.

## Notes

- This project requires a working webcam.
- Face recognition accuracy depends on image quality and lighting.
- Attendance is stored locally in CSV format and is not connected to a database.
- The UI template is still mostly original; this pass focused on code reliability and setup recovery.

## Verification Performed

- Repository contents were restored locally from GitHub.
- `app.py` was reviewed and updated for safer runtime behavior.
- A syntax-level validation pass was attempted. Standard `py_compile` was blocked by a local filesystem write issue under OneDrive, so full runtime verification still depends on installing dependencies and testing the webcam flow on this machine.
