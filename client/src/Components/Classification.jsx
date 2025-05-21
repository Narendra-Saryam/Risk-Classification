import React, { useState } from 'react';
import axios from 'axios';
import { Bar, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend, ArcElement);

const Classification = () => {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    weight: '',
    height: '',
    alcohol_consumption: '1',
    alcohol_duration: '1',
    tobacco_chewing: '1',
    tobacco_duration: '1',
    smoking: '1',
    smoking_duration: '1',
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
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const payload = {
      ...formData,
      gender: parseInt(formData.gender),
      age: parseFloat(formData.age),
      weight: parseFloat(formData.weight),
      height: parseFloat(formData.height),
      alcohol_consumption: parseFloat(formData.alcohol_consumption),
      alcohol_duration: parseFloat(formData.alcohol_duration),
      tobacco_chewing: parseFloat(formData.tobacco_chewing),
      tobacco_duration: parseFloat(formData.tobacco_duration),
      smoking: parseFloat(formData.smoking),
      smoking_duration: parseFloat(formData.smoking_duration),
      cancer: formData.cancer === 'Yes' ? 1 : 0,
      diabetes: formData.diabetes === 'Yes' ? 1 : 0,
      hypertension: formData.hypertension === 'Yes' ? 1 : 0
    };

    try {
      const API_URL = import.meta.env.DEV
        ? 'http://localhost:5000/predict'
        : 'https://risk-classification.onrender.com/predict';

      const res = await axios.post(API_URL, payload);

      const probs = res.data.probabilities;
      const riskLabels = ['Normal', 'Low', 'High'];
      const predictedIndex = riskLabels.indexOf(res.data.risk_level);

      setResult(predictedIndex);
      setProbabilities([
        probs['Normal'] / 100,
        probs['Low'] / 100,
        probs['High'] / 100
      ]);
    } catch (error) {
      const msg = error.response?.data?.error || error.message || 'Unknown error';
      alert('Prediction error: ' + msg);
      console.error('Error details:', error);
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
        <select name="gender" value={formData.gender} onChange={handleChange} className="border p-2 rounded" required>
          <option value="">Select Gender</option>
          <option value="0">Female</option>
          <option value="1">Male</option>
        </select>
        <input type="number" name="weight" value={formData.weight} onChange={handleChange} placeholder="Weight (kg)" required className="border p-2 rounded" />
        <input type="number" name="height" value={formData.height} onChange={handleChange} placeholder="Height (cm)" required className="border p-2 rounded" />

        {/* Consumption Levels */}
        {['alcohol', 'tobacco', 'smoking'].map(sub => (
          <React.Fragment key={sub}>
            <select name={`${sub}_consumption`} value={formData[`${sub}_consumption`]} onChange={handleChange} className="border p-2 rounded" required>
              {[...Array(10)].map((_, i) => (
                <option key={i+1} value={i+1}>{sub} level {i+1}</option>
              ))}
            </select>
            <select name={`${sub}_duration`} value={formData[`${sub}_duration`]} onChange={handleChange} className="border p-2 rounded" required>
              {[...Array(20)].map((_, i) => (
                <option key={i+1} value={i+1}>{sub} duration {i+1} year(s)</option>
              ))}
            </select>
          </React.Fragment>
        ))}

        {/* Organ Functions */}
        {['liver_function', 'kidney_function', 'lung_function'].map(field => (
          <select key={field} name={field} value={formData[field]} onChange={handleChange} className="border p-2 rounded" required>
            <option value="Mild">{field.replace('_', ' ')}: Mild</option>
            <option value="Moderate">{field.replace('_', ' ')}: Moderate</option>
            <option value="Severe">{field.replace('_', ' ')}: Severe</option>
          </select>
        ))}

        {/* Binary fields */}
        {['addiction_dependence', 'cancer', 'diabetes', 'hypertension'].map(field => (
          <select key={field} name={field} value={formData[field]} onChange={handleChange} className="border p-2 rounded" required>
            <option value="No">{field.replace('_', ' ')}: No</option>
            <option value="Yes">{field.replace('_', ' ')}: Yes</option>
          </select>
        ))}

        <button type="submit" className="col-span-2 bg-blue-500 text-white px-4 py-2 rounded">Predict Risk</button>
      </form>

      {result !== null && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold text-center">
            Prediction Result: <span className="font-bold" style={{ color: riskColors[result] }}>{riskLabels[result]}</span>
          </h3>
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
