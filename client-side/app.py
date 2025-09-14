import requests
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/results")
def results_page():
    return render_template("results.html")

@app.route("/report")
def report_page():
    return render_template("report.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    files = {}
    for key in ["photo1", "photo2", "photo3"]:
        file = request.files[key]
        # Use the original filename if it exists
        filename = file.filename if file.filename else f"{key}.jpg"
        files[key] = (filename, file.stream, file.mimetype)

    ai_url = "http://192.168.10.3:5000/analyze"  # friend's IP
    response = requests.post(ai_url, files=files)

    return response.json(), response.status_code



if __name__ == "__main__":
    app.run(debug=True)
