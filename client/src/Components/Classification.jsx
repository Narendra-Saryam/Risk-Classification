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
    <div className='max-h-screen min-h-screen flex flex-col jusbtify-between'>
      <div className='left-0 right-0 z-50 p-1 md:p-2 flex flex-row gap-4 sm:gap-0 w-full border-b-[0.5px] border-purple-300 shadow-black text-white bg-black bg-opacity-30 items-center justify-between'>
        <h1 className='text-lg sm:text-xl text-center sm:text-left text-white hover:text-purple-500 transition-colors duration-300 ml-1 md:pl-4'>Risk Classification Model</h1>
        <div className='flex flex-row gap-4 text-lg sm:text-xl text-center sm:text-left'>
          <a className='text-white hover:text-purple-500 transition-colors duration-300 md:pr-3'  href="https://narendra-saryam.netlify.app/">About</a>
          <a className='text-white hover:text-purple-500 transition-colors duration-300 md:pr-6' href="https://github.com/Narendra-Saryam/Risk-Classification">Github</a>
        </div>
      </div>

      <div className="flex flex-col items-center justify-center  max-w-7xl mx-auto p-6 mt-8 bg-[#161c25] text-white shadow-md rounded-xl space-y-4 border-2 border-purple-500">
      <h2 className="text-2xl font-semibold mb-4 bg-purple-500 p-1 px-4 rounded-lg text-black">Risk Prediction</h2>
      <form onSubmit={handleSubmit} className="grid grid-cols-2 md:grid-cols-3 gap-2">
        {Object.keys(initialForm).map((key) => (
          <div key={key} className="flex flex-col">
            <label className="capitalize text-sm font-medium text-white p-1">
              {key.replaceAll('_', ' ')}
            </label>
            <select
              name={key}
              value={formData[key]}
              onChange={handleChange}
              required
              className="border-b p-2 rounded bg-transparent text-purple-700 hover:bg-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="" disabled>Select {key.replaceAll('_', ' ')}</option>
              {renderOptions(key)}
            </select>
          </div>
        ))}
        <div className=" w-full text-center pt-7">
          <button type="submit" className="w-full bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700">
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

      <div className='md:absolute bottom-0 mt-2 p-2 w-full text-center bg-purple-500 bg-opacity-50 text-white text-sm md:text-base'>
      The Project is made by: <strong>Narendra Saryam</strong><br />
      <span className="text-xs italic">(Note: Model may give wrong prediction)</span>
      </div>
    </div>
    
  );
};

export default Classification;
