import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='mysql',
        database='python'
        
    )
    
    
    
def check_connection():
    try:
        conn = get_connection()
        if conn.is_connected():
            print("Connected to MYSQL Database")
            conn.close()
    except Exception as e:
        print("MYSQL connection error :",str(e)) 