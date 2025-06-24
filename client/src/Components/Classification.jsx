import React, { useState } from 'react';
import axios from 'axios';

const initialForm = {
  age: '',
  gender: '',
  height: '',
  weight: '',
  alcohol_consumption_per_day_in_liter: '',
  alcohol_duration: '',
  tobacco_chewing_per_day_in_gram: '',
  tobacco_duration: '',
  smoking_per_day: '',
  smoking_duration: '',
  addiction_dependence: '',
  liver_function: '',
  kidney_function: '',
  lung_function: '',
  cancer: '',
  diabetes: '',
  hypertension: ''
};

const range = (start, end) => {
  const step = start < end ? 1 : -1;
  return Array.from({ length: Math.abs(end - start) + 1 }, (_, i) => start + i * step);
};

const selectOptions = {
  age: range(18, 91),
  gender: [
    { label: "Male", value: 0 },
    { label: "Female", value: 1 }
  ],
  height: range(150, 200),
  weight: range(40, 117),
  alcohol_consumption_per_day_in_liter: range(0, 15),
  alcohol_duration: range(1, 29),
  tobacco_chewing_per_day_in_gram: range(0, 8),
  tobacco_duration: range(1, 24),
  smoking_per_day: range(0, 40),
  smoking_duration: range(1, 34),
  addiction_dependence: [
    { label: "None", value: 0 },
    { label: "Mild", value: 1 },
    { label: "Severe", value: 2 }
  ],
  liver_function: [
    { label: "Normal", value: 0 },
    { label: "Abnormal", value: 1 }
  ],
  kidney_function: [
    { label: "Normal", value: 0 },
    { label: "Abnormal", value: 1 }
  ],
  lung_function: [
    { label: "Normal", value: 0 },
    { label: "Abnormal", value: 1 }
  ],
  cancer: [
    { label: "No", value: 0 },
    { label: "Yes", value: 1 }
  ],
  diabetes: [
    { label: "No", value: 0 },
    { label: "Yes", value: 1 }
  ],
  hypertension: [
    { label: "No", value: 0 },
    { label: "Yes", value: 1 }
  ]
};

const Classification = () => {
  const [formData, setFormData] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setError('');

    const payload = {};
    for (let key in formData) {
      const val = formData[key];
      payload[key] = isNaN(val) ? val : Number(val);
    }

    try {
      const API_URL = import.meta.env.DEV
        ? 'http://localhost:5000/predict'
        : 'https://risk-classification.onrender.com/predict';

      const res = await axios.post(API_URL, payload);
      const pred = res.data.risk_prediction;
      const label = ['Normal', 'Low Risk', 'High Risk'][pred];
      setResult(label);
    } catch (err) {
      setError('Prediction failed: ' + (err.response?.data?.error || err.message));
    }
  };

  const renderOptions = (key) => {
    const values = selectOptions[key];
    return typeof values[0] === 'object'
      ? values.map(({ label, value }) => (
          <option key={value} value={value}>{label}</option>
        ))
      : values.map(val => (
          <option key={val} value={val}>{val}</option>
        ));
  };

  return (
    <>
      <div className='left-0 right-0 z-50 p-1 md:p-2 flex flex-row gap-4 sm:gap-0 w-full text-white bg-black bg-opacity-30 items-center justify-between'>
        <h1 className='text-lg sm:text-xl text-center sm:text-left text-white hover:text-purple-500 transition-colors duration-300'>Risk Classification Model</h1>
        <div className='flex flex-row gap-4 text-lg sm:text-xl text-center sm:text-left'>
          <a className='text-white hover:text-purple-500 transition-colors duration-300'  href="">About</a>
          <a className='text-white hover:text-purple-500 transition-colors duration-300' href="">Code</a>
        </div>
      </div>
      <div className="max-w-2xl mx-auto p-6 bg-white shadow-md rounded-xl space-y-4">
      <h2 className="text-2xl font-semibold mb-4">Clinical Risk Prediction</h2>
      <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">
        {Object.keys(initialForm).map((key) => (
          <div key={key} className="flex flex-col">
            <label className="capitalize text-sm font-medium text-gray-700">
              {key.replaceAll('_', ' ')}
            </label>
            <select
              name={key}
              value={formData[key]}
              onChange={handleChange}
              required
              className="border p-2 rounded"
            >
              <option value="" disabled>Select {key.replaceAll('_', ' ')}</option>
              {renderOptions(key)}
            </select>
          </div>
        ))}
        <div className="col-span-2 text-center">
          <button type="submit" className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700">
            Predict Risk
          </button>
        </div>
      </form>
      {result && (
        <div className="text-center text-lg font-semibold text-green-600">
          Predicted Risk: {result}
        </div>
      )}
      {error && (
        <div className="text-center text-red-600">
          {error}
        </div>
      )}
    </div>
    </>
    
  );
};

export default Classification;
