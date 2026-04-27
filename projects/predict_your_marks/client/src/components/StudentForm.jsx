import { useState } from "react";
import API from "../services/api";

export default function StudentForm({ setResult }) {
  const [form, setForm] = useState({
    name: "",
    hours: "",
    attendance: "",
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await API.post("/predict", form);
      setResult(res.data);
      alert("Prediction Done ✅");
      setForm({ name: "", hours: "", attendance: "" }); // ✅ form clear
    } catch (err) {
      console.log(err);
    }
  };

  return (
    <div className="max-w-md mx-auto mt-6 p-6 bg-white shadow-lg rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-4 text-blue-500">
        Student Prediction Form
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          name="name"
          value={form.name}
          placeholder="Name"
          onChange={handleChange}
          className="w-full p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
        />

        <input
          name="hours"
          value={form.hours}
          placeholder="Hours Studied"
          onChange={handleChange}
          className="w-full p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
        />

        <input
          name="attendance"
          value={form.attendance}
          placeholder="Attendance"
          onChange={handleChange}
          className="w-full p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
        />

        <button
          type="submit"
          className="w-full cursor-pointer bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
        >
          Predict
        </button>
      </form>
    </div>
  );
}