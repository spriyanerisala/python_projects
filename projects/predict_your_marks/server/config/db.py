import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        ssl_disabled=True
    )

def check_connection():
    try:
        conn = get_connection()
        if conn.is_connected():
            print("Connected to MySQL Database")
            conn.close()
    except Exception as e:
        print("MySQL connection error:", str(e))

if __name__ == "__main__":
    check_connection()