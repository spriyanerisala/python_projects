/* eslint-disable no-unused-vars */
import React, { useState } from 'react'
import StudentForm from './components/StudentForm'
import StudentTable from './components/StudentTable';
const App = () => {

  const [result,setResult] = useState(null);
  return (
  <div style={{ textAlign: "center" }}>
      <h1 className='text-blue-800 font-bold text-2xl'>Student Performance Predictor</h1>

      <StudentForm setResult={setResult} />
      <StudentTable/>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h2 className='text-blue-700'>Prediction Result</h2>
          <h3>Marks: {result.predicted_marks}</h3>
        </div>
      )}
    </div>
  )
}

export default App