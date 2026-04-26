from flask import Blueprint, request, jsonify
import pickle
import numpy as np
from config.db import get_connection

predict_bp = Blueprint('predict', __name__)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        name = data.get("name")
        hours = float(data.get("hours"))
        attendance = float(data.get("attendance"))

        prediction = model.predict([[hours, attendance]])[0]

        # Save to DB
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO students (name, hours_studied, attendance, predicted_marks)
            VALUES (%s, %s, %s, %s)
        """, (name, hours, attendance, float(prediction)))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "predicted_marks": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})