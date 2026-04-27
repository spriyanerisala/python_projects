/* eslint-disable react-hooks/immutability */
import { useEffect, useState } from "react";
import API from "../services/api";

export default function StudentTable() {
  const [students, setStudents] = useState([]);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const res = await API.get("/students");
      setStudents(res.data);
    } catch (err) {
      console.log(err);
    }
  };

  const handleDelete = async (id) => {
    try {
      await API.delete(`/students/${id}`);
      fetchData(); // ✅ table refresh
    } catch (err) {
      console.log(err);
    }
  };

  return (
    <div className="max-w-4xl mx-auto mt-10 p-4">
      <h2 className="text-2xl font-bold text-center mb-4 text-blue-600">
        Students Data
      </h2>

      <div className="overflow-x-auto">
        <table className="w-full border border-gray-300 shadow-md rounded-lg overflow-hidden">
          <thead className="bg-blue-600 text-white">
            <tr>
              <th className="p-3">Name</th>
              <th className="p-3">Hours</th>
              <th className="p-3">Attendance</th>
              <th className="p-3">Predicted Marks</th>
              <th className="p-3">Action</th>
            </tr>
          </thead>

          <tbody>
            {students.map((s, i) => (
              <tr
                key={i}
                className="text-center border-b hover:bg-gray-100 transition"
              >
                <td className="p-3">{s.name}</td>
                <td className="p-3">{s.hours_studied}</td>
                <td className="p-3">{s.attendance}</td>
                <td className="p-3 font-semibold text-green-600">
                  {s.predicted_marks}
                </td>
                <td className="p-3">
                  <button
                    onClick={() => handleDelete(s.id)}
                    className="bg-red-500 text-white px-3 py-1 rounded hover:bg-red-600 transition"
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}