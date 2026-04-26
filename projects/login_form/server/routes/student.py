from flask import Blueprint, jsonify
from config.db import get_connection

student_bp = Blueprint('student', __name__)
@student_bp.route('/students',methods=['GET'])
def get_students():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("select * from students")
    students=cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(students)


@student_bp.route('/students/<int:id>',methods=["DELETE"])
def delete_student(id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("delete from students where id=%s",(id,))
    conn.commit()
    cursor.close()
    return jsonify({"message":"Student deleted successfully"})

    