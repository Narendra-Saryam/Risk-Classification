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
      setResult(res.data.prediction);
    } catch (err) {
      setError('Error making prediction. Please try again.');
      console.error(err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-teal-900 py-12 px-4">
      <div className="text-center mb-10">
        <div className="text-7xl mb-4">🏥</div>
        <h1 className="text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-teal-400 mb-4">Risk Classification</h1>
        <p className="text-cyan-200 text-xl">Advanced Health Risk Assessment System</p>
      </div>
      <div className="max-w-6xl mx-auto bg-gradient-to-br from-slate-800 via-blue-900 to-slate-900 rounded-xl shadow-2xl border-2 border-teal-600 p-12 space-y-10">
        <form onSubmit={handleSubmit} className="space-y-6">
          <section>
            <h2 className="text-3xl font-bold text-cyan-200 mb-8 border-b-2 border-teal-500 pb-3 flex items-center gap-3">
              <span className="text-4xl">👤</span>
              Personal Information
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {/* Age */}
              <div>
                <label htmlFor="age" className="block text-sm font-semibold text-cyan-200 mb-2">Age</label>
                <select
                  id="age"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Age</option>
                  {selectOptions.age.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              {/* Gender */}
              <div>
                <label htmlFor="gender" className="block text-sm font-semibold text-cyan-200 mb-2">Gender</label>
                <select
                  id="gender"
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Gender</option>
                  {selectOptions.gender.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>

              {/* Height */}
              <div>
                <label htmlFor="height" className="block text-sm font-semibold text-cyan-200 mb-2">Height (cm)</label>
                <select
                  id="height"
                  name="height"
                  value={formData.height}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Height</option>
                  {selectOptions.height.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              {/* Weight */}
              <div>
                <label htmlFor="weight" className="block text-sm font-semibold text-cyan-200 mb-2">Weight (kg)</label>
                <select
                  id="weight"
                  name="weight"
                  value={formData.weight}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Weight</option>
                  {selectOptions.weight.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>
            </div>
          </section>
          <section>
            <h2 className="text-3xl font-bold text-cyan-200 mb-8 border-b-2 border-teal-500 pb-3 flex items-center gap-3">
              <span className="text-4xl">🚬</span>
              Lifestyle Information
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {/* Alcohol Consumption per Day in Liter */}
              <div>
                <label htmlFor="alcohol_consumption_per_day_in_liter" className="block text-sm font-semibold text-cyan-200 mb-2">Alcohol Consumption (L/day)</label>
                <select
                  id="alcohol_consumption_per_day_in_liter"
                  name="alcohol_consumption_per_day_in_liter"
                  value={formData.alcohol_consumption_per_day_in_liter}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Amount</option>
                  {selectOptions.alcohol_consumption_per_day_in_liter.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              {/* Alcohol Duration */}
              <div>
                <label htmlFor="alcohol_duration" className="block text-sm font-semibold text-cyan-200 mb-2">Alcohol Duration (Years)</label>
                <select
                  id="alcohol_duration"
                  name="alcohol_duration"
                  value={formData.alcohol_duration}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Duration</option>
                  {selectOptions.alcohol_duration.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              {/* Tobacco Chewing per Day in Gram */}
              <div>
                <label htmlFor="tobacco_chewing_per_day_in_gram" className="block text-sm font-semibold text-cyan-200 mb-2">Tobacco Chewing (g/day)</label>
                <select
                  id="tobacco_chewing_per_day_in_gram"
                  name="tobacco_chewing_per_day_in_gram"
                  value={formData.tobacco_chewing_per_day_in_gram}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Amount</option>
                  {selectOptions.tobacco_chewing_per_day_in_gram.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              {/* Tobacco Duration */}
              <div>
                <label htmlFor="tobacco_duration" className="block text-sm font-semibold text-cyan-200 mb-2">Tobacco Duration (Years)</label>
                <select
                  id="tobacco_duration"
                  name="tobacco_duration"
                  value={formData.tobacco_duration}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Duration</option>
                  {selectOptions.tobacco_duration.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              {/* Smoking per Day */}
              <div>
                <label htmlFor="smoking_per_day" className="block text-sm font-semibold text-cyan-200 mb-2">Smoking (Cigarettes/day)</label>
                <select
                  id="smoking_per_day"
                  name="smoking_per_day"
                  value={formData.smoking_per_day}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Amount</option>
                  {selectOptions.smoking_per_day.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              {/* Smoking Duration */}
              <div>
                <label htmlFor="smoking_duration" className="block text-sm font-semibold text-cyan-200 mb-2">Smoking Duration (Years)</label>
                <select
                  id="smoking_duration"
                  name="smoking_duration"
                  value={formData.smoking_duration}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Duration</option>
                  {selectOptions.smoking_duration.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>
            </div>
          </section>
          <section>
            <h2 className="text-3xl font-bold text-cyan-200 mb-8 border-b-2 border-teal-500 pb-3 flex items-center gap-3">
              <span className="text-4xl">💊</span>
              Health Information
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {/* Addiction Dependence */}
              <div>
                <label htmlFor="addiction_dependence" className="block text-sm font-semibold text-cyan-200 mb-2">Addiction Dependence</label>
                <select
                  id="addiction_dependence"
                  name="addiction_dependence"
                  value={formData.addiction_dependence}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Level</option>
                  {selectOptions.addiction_dependence.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>

              {/* Liver Function */}
              <div>
                <label htmlFor="liver_function" className="block text-sm font-semibold text-cyan-200 mb-2">Liver Function</label>
                <select
                  id="liver_function"
                  name="liver_function"
                  value={formData.liver_function}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Status</option>
                  {selectOptions.liver_function.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>

              {/* Kidney Function */}
              <div>
                <label htmlFor="kidney_function" className="block text-sm font-semibold text-cyan-200 mb-2">Kidney Function</label>
                <select
                  id="kidney_function"
                  name="kidney_function"
                  value={formData.kidney_function}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Status</option>
                  {selectOptions.kidney_function.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>

              {/* Lung Function */}
              <div>
                <label htmlFor="lung_function" className="block text-sm font-semibold text-cyan-200 mb-2">Lung Function</label>
                <select
                  id="lung_function"
                  name="lung_function"
                  value={formData.lung_function}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Status</option>
                  {selectOptions.lung_function.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>

              {/* Cancer */}
              <div>
                <label htmlFor="cancer" className="block text-sm font-semibold text-cyan-200 mb-2">Cancer</label>
                <select
                  id="cancer"
                  name="cancer"
                  value={formData.cancer}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Status</option>
                  {selectOptions.cancer.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>

              {/* Diabetes */}
              <div>
                <label htmlFor="diabetes" className="block text-sm font-semibold text-cyan-200 mb-2">Diabetes</label>
                <select
                  id="diabetes"
                  name="diabetes"
                  value={formData.diabetes}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Status</option>
                  {selectOptions.diabetes.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>

              {/* Hypertension */}
              <div>
                <label htmlFor="hypertension" className="block text-sm font-semibold text-cyan-200 mb-2">Hypertension</label>
                <select
                  id="hypertension"
                  name="hypertension"
                  value={formData.hypertension}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-3 text-base text-black bg-white border-2 border-teal-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 rounded-md shadow-sm"
                  required
                >
                  <option value="">Select Status</option>
                  {selectOptions.hypertension.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>
            </div>
          </section>
          <div className="flex justify-center mt-12">
            <button
              type="submit"
              className="px-12 py-5 bg-gradient-to-r from-teal-600 to-cyan-600 text-white font-bold text-xl rounded-lg shadow-2xl hover:from-teal-700 hover:to-cyan-700 focus:outline-none focus:ring-4 focus:ring-teal-500 focus:ring-opacity-50 transform hover:scale-105 transition ease-in-out duration-300 flex items-center gap-3"
            >
              <span className="text-2xl">🔍</span>
              Predict Risk
            </button>
          </div>
        </form>

        {result && (
          <div className="mt-8 p-6 bg-gradient-to-r from-emerald-900 to-teal-900 border-2 border-emerald-500 text-emerald-200 rounded-lg shadow-2xl">
            <h3 className="text-2xl font-bold mb-3 flex items-center gap-2">
              <span className="text-3xl">✅</span>
              Prediction Result:
            </h3>
            <p className="text-xl">The predicted risk classification is: <span className="font-bold text-emerald-300">{result}</span></p>
          </div>
        )}

        {error && (
          <div className="mt-8 p-6 bg-gradient-to-r from-red-900 to-rose-900 border-2 border-red-500 text-red-200 rounded-lg shadow-2xl">
            <h3 className="text-2xl font-bold mb-3 flex items-center gap-2">
              <span className="text-3xl">❌</span>
              Error:
            </h3>
            <p className="text-xl">{error}</p>
          </div>
        )}
      </div>

      <footer className="mt-12 py-8 border-t-2 border-teal-700">
        <div className="max-w-6xl mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-6">
            <div>
              <h3 className="text-cyan-400 font-bold text-lg mb-3">About Risk Classification</h3>
              <p className="text-cyan-200 text-sm">Advanced AI-powered health risk assessment system designed to provide accurate predictions based on comprehensive patient data.</p>
            </div>
            <div>
              <h3 className="text-cyan-400 font-bold text-lg mb-3">Quick Links</h3>
              <ul className="text-cyan-200 text-sm space-y-2">
                <li className="hover:text-teal-400 cursor-pointer transition">Privacy Policy</li>
                <li className="hover:text-teal-400 cursor-pointer transition">Terms of Service</li>
                <li className="hover:text-teal-400 cursor-pointer transition">Contact Us</li>
                <li className="hover:text-teal-400 cursor-pointer transition">Help & Support</li>
              </ul>
            </div>
            <div>
              <h3 className="text-cyan-400 font-bold text-lg mb-3">Contact Information</h3>
              <p className="text-cyan-200 text-sm mb-2">📧 support@riskclassification.com</p>
              <p className="text-cyan-200 text-sm mb-2">📞 +1 (555) 123-4567</p>
              <p className="text-cyan-200 text-sm">🏢 123 Healthcare Ave, Medical City</p>
            </div>
          </div>
          <div className="text-center pt-6 border-t border-teal-700">
            <p className="text-cyan-300 text-sm mb-2">The Project is made by: <strong className="text-teal-400">Narendra Saryam</strong></p>
            <p className="text-cyan-400 text-xs italic mb-2">(Note: Model may give wrong prediction)</p>
            <p className="text-cyan-300 text-sm">© 2026 Risk Classification System. All rights reserved.</p>
            <p className="text-cyan-400 text-xs mt-2">Powered by Advanced Machine Learning Technology</p>
          </div>
        </div>
      </footer>
    </div>
    
  );
};

export default Classification;
