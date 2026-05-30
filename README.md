# Face Recognition Based Attendance System

This repository contains a Flask-based attendance application that captures webcam images, detects faces with OpenCV, identifies registered users with a KNN classifier, and stores daily attendance records in CSV format.

## Stack

- Flask
- OpenCV
- scikit-learn
- pandas
- joblib

## Project Structure

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

- Register a new user from live webcam capture
- Train a local KNN face recognition model from stored images
- Mark attendance in real time from the webcam
- Export the current day's attendance as CSV
- Show summary metrics for registered users and attendance activity
- Run from a simple browser-based dashboard

## Modernization Pass

This version is more current than the original prototype in both runtime behavior and presentation:

1. safer filesystem handling with `pathlib`
2. webcam access only when routes actually need it
3. validation for user names and numeric user IDs
4. automatic attendance file creation
5. CSV download route for the current day's records
6. summary metrics for attendance activity
7. a cleaner, more polished dashboard layout
8. more defensive model and cascade checks

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open the local Flask URL in your browser.

## Usage

1. Open the dashboard.
2. Add a new user with a name and numeric ID.
3. Let the webcam capture the training images.
4. After model training completes, click `Take Attendance`.
5. Press `Esc` in the OpenCV window to stop the live scan.
6. Download the daily CSV if needed.

## Notes

- This project requires a working webcam.
- The recognition model is trained locally from captured face images.
- Attendance is stored as daily CSV files in `Attendance/`.
- The interface is still intentionally lightweight and local-first.
