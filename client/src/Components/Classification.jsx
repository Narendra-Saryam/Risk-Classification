import React, { useState } from 'react';
import axios from 'axios';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Pie } from 'react-chartjs-2';

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend, ArcElement);

const Classification = () => {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    weight: '',
    height: '',
    alcohol_consumption: '1',
    tobacco_chewing: '1',
    smoking: '1',
    duration: '1',
    liver_function: 'Mild',
    kidney_function: 'Mild',
    lung_function: 'Mild',
    addiction_dependence: 'No',
    cancer: 'No',
    diabetes: 'No',
    hypertension: 'No'
  });

  const [result, setResult] = useState(null);
  const [probabilities, setProbabilities] = useState([0.33, 0.33, 0.33]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = {
      ...formData,
      gender: parseInt(formData.gender),
      alcohol_consumption: parseInt(formData.alcohol_consumption),
      tobacco_chewing: parseInt(formData.tobacco_chewing),
      smoking: parseInt(formData.smoking),
      duration: parseInt(formData.duration),
      cancer: formData.cancer === 'Yes' ? 1 : 0,
      diabetes: formData.diabetes === 'Yes' ? 1 : 0,
      hypertension: formData.hypertension === 'Yes' ? 1 : 0
    };

    try {
      const API_URL = import.meta.env.DEV
        ? 'http://localhost:5000/predict'
        : 'https://risk-classification.onrender.com/predict';

        const res = await axios.post(API_URL, payload);

      setResult(res.data.risk_prediction);

      // Fake probabilities as placeholders
      const fakeProbs = [0, 0, 0];
      fakeProbs[res.data.risk_prediction] = 0.7;
      const remaining = 0.3 / 2;
      fakeProbs.forEach((v, i) => { if (v === 0) fakeProbs[i] = remaining });
      setProbabilities(fakeProbs);
    } catch (error) {
      const errorMessage = error.response?.data?.error || error.message || 'An unknown error occurred';
      alert('Prediction error: ' + errorMessage);
      console.error('Prediction error details:', error);
    }
  };

  const riskLabels = ['Normal Risk', 'Low Risk', 'High Risk'];
  const riskColors = ['#4caf50', '#ff9800', '#f44336'];

  const pieData = {
    labels: riskLabels,
    datasets: [{
      data: probabilities,
      backgroundColor: riskColors
    }]
  };

  const barData = {
    labels: riskLabels,
    datasets: [{
      label: 'Probability',
      data: probabilities,
      backgroundColor: riskColors
    }]
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Clinical Risk Classifier</h2>
      <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">
        <input type="number" name="age" value={formData.age} onChange={handleChange} placeholder="Age" required className="border p-2 rounded" />
        <select name="gender" value={formData.gender} onChange={handleChange} className="border p-2 rounded">
          <option value="">Select Gender</option>
          <option value="0">Female</option>
          <option value="1">Male</option>
        </select>
        <input type="number" name="weight" value={formData.weight} onChange={handleChange} placeholder="Weight (kg)" required className="border p-2 rounded" />
        <input type="number" name="height" value={formData.height} onChange={handleChange} placeholder="Height (cm)" required className="border p-2 rounded" />

        {['alcohol_consumption', 'tobacco_chewing', 'smoking'].map(field => (
          <select key={field} name={field} value={formData[field]} onChange={handleChange} className="border p-2 rounded">
            {[...Array(10)].map((_, i) => (
              <option key={i+1} value={i+1}>{field.replace('_', ' ')} Level {i+1}</option>
            ))}
          </select>
        ))}

        <select name="duration" value={formData.duration} onChange={handleChange} className="border p-2 rounded">
          {[...Array(20)].map((_, i) => (
            <option key={i+1} value={i+1}>Duration: {i+1} year(s)</option>
          ))}
        </select>

        {['liver_function', 'kidney_function', 'lung_function'].map(field => (
          <select key={field} name={field} value={formData[field]} onChange={handleChange} className="border p-2 rounded">
            <option value="Mild">{field.replace('_', ' ')}: Mild</option>
            <option value="Moderate">{field.replace('_', ' ')}: Moderate</option>
            <option value="Severe">{field.replace('_', ' ')}: Severe</option>
          </select>
        ))}

        {['addiction_dependence', 'cancer', 'diabetes', 'hypertension'].map(field => (
          <select key={field} name={field} value={formData[field]} onChange={handleChange} className="border p-2 rounded">
            <option value="No">{field.replace('_', ' ')}: No</option>
            <option value="Yes">{field.replace('_', ' ')}: Yes</option>
          </select>
        ))}

        <button type="submit" className="col-span-2 bg-blue-500 text-white px-4 py-2 rounded">Predict Risk</button>
      </form>

      {result !== null && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold text-center">Prediction Result: <span className="font-bold" style={{ color: riskColors[result] }}>{riskLabels[result]}</span></h3>
          <div className="flex flex-col md:flex-row justify-center gap-10 mt-6">
            <div className="w-full md:w-1/2">
              <Bar data={barData} />
            </div>
            <div className="w-full md:w-1/2">
              <Pie data={pieData} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Classification;