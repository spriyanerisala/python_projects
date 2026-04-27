from flask import Flask
from flask_cors import CORS
from routes.predict import predict_bp
from routes.student import student_bp
from config.db import check_connection
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
check_connection()

frontend_url = os.getenv("FRONTEND_URL", "https://python-projects-e6vh.vercel.app")
print(f"Allowing CORS for: {frontend_url}")  # helpful for debugging in Render logs

CORS(app, origins=[frontend_url])

app.register_blueprint(predict_bp)
app.register_blueprint(student_bp)

@app.route('/')
def home():
    return {"message": "Flask API is running"}

if __name__ == '__main__':
    app.run(debug=True)