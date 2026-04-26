from flask import Flask
from flask_cors import CORS
from routes.predict import predict_bp
from routes.student import student_bp
from config.db import check_connection
app = Flask(__name__)
check_connection()

CORS(app)

app.register_blueprint(predict_bp)
app.register_blueprint(student_bp)

@app.route('/')
def home():
    return {"message":"Flask API is running"}
if __name__ == '__main__':
    app.run(debug=True)