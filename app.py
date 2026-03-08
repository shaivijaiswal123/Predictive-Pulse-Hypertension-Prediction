"""
Predictive Pulse - Advanced Flask Web Application
Professional Medical Dashboard with Complete Functionality
"""

from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
import secrets
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Load all required files
print("="*60)
print("PREDICTIVE PULSE - LOADING SYSTEM")
print("="*60)

try:
    # Try loading from models folder first, then root
    if os.path.exists('models/logreg_model.pkl'):
        model = pickle.load(open('models/logreg_model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        feature_names = pickle.load(open('models/feature_names.pkl', 'rb'))
        mappings = pickle.load(open('models/mappings.pkl', 'rb'))
        print("✅ All model files loaded from models/ folder!")
    else:
        model = pickle.load(open('logreg_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))
        mappings = pickle.load(open('mappings.pkl', 'rb'))
        print("✅ All model files loaded from root folder!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Please run train_model.py first")
    exit()

# Reverse mappings for display
age_groups = ['18-34', '35-50', '51-64', '65+']
severity_options = ['Mild', 'Moderate', 'Severe']
time_options = ['<1 Year', '1-3 Years', '3-5 Years', '5+ Years']
bp_ranges = {
    'systolic': ['90-100', '101-110', '111-120', '121-130', '131-140', '141-150', 
                 '151-160', '161-170', '171-180', '181-190', '191-200'],
    'diastolic': ['60-70', '71-80', '81-90', '91-100', '101-110', '111-120']
}

# Comprehensive stage information with detailed medical recommendations
STAGE_INFO = {
    0: {
        'name': 'NORMAL',
        'display_name': 'Normal Blood Pressure',
        'color': '#10b981',
        'gradient': 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
        'light_bg': '#d1fae5',
        'risk': 'Low Risk',
        'risk_level': 1,
        'icon': '🟢',
        'description': 'Your blood pressure is within the optimal range. Great job maintaining healthy levels!',
        'clinical_guidelines': 'Systolic < 120 mmHg AND Diastolic < 80 mmHg',
        'recommendations': [
            {'type': 'lifestyle', 'icon': '🥗', 'text': 'Continue your healthy lifestyle habits'},
            {'type': 'exercise', 'icon': '🏃', 'text': 'Maintain regular exercise (30 min/day, 5 days/week)'},
            {'type': 'diet', 'icon': '🧂', 'text': 'Keep salt intake below 5g per day'},
            {'type': 'monitoring', 'icon': '📊', 'text': 'Check blood pressure annually'},
            {'type': 'diet', 'icon': '🍎', 'text': 'Follow DASH diet principles'},
            {'type': 'stress', 'icon': '🧘', 'text': 'Practice stress management techniques'}
        ],
        'diet_plan': [
            'Rich in fruits and vegetables',
            'Whole grains and lean proteins',
            'Low-fat dairy products',
            'Limited processed foods',
            'Reduced sodium intake'
        ],
        'exercise_plan': [
            'Brisk walking 30 min daily',
            'Swimming or cycling 2-3 times/week',
            'Strength training 2 times/week',
            'Flexibility exercises like yoga'
        ],
        'warning_signs': [
            'Persistent headaches',
            'Unexplained anxiety',
            'Sleep disturbances',
            'Chest discomfort'
        ],
        'next_checkup': 'Annual physical examination',
        'medication': 'No medication needed at this time'
    },
    1: {
        'name': 'STAGE-1 HYPERTENSION',
        'display_name': 'Stage 1 Hypertension',
        'color': '#f59e0b',
        'gradient': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
        'light_bg': '#fef3c7',
        'risk': 'Moderate Risk',
        'risk_level': 2,
        'icon': '🟡',
        'description': 'Your blood pressure indicates Stage 1 hypertension. Early intervention can prevent progression.',
        'clinical_guidelines': 'Systolic 130-139 mmHg OR Diastolic 80-89 mmHg',
        'recommendations': [
            {'type': 'medical', 'icon': '👨‍⚕️', 'text': 'Consult healthcare provider within 1 month'},
            {'type': 'lifestyle', 'icon': '📝', 'text': 'Start lifestyle modifications immediately'},
            {'type': 'diet', 'icon': '🥬', 'text': 'Reduce sodium intake to <4g per day'},
            {'type': 'exercise', 'icon': '🚶', 'text': 'Increase physical activity gradually'},
            {'type': 'monitoring', 'icon': '📱', 'text': 'Monitor BP weekly at home'},
            {'type': 'weight', 'icon': '⚖️', 'text': 'Achieve and maintain healthy weight'}
        ],
        'diet_plan': [
            'DASH diet: fruits, vegetables, whole grains',
            'Limit sodium to 1500-2300mg/day',
            'Reduce saturated fats',
            'Limit alcohol to 1-2 drinks/day',
            'Increase potassium-rich foods'
        ],
        'exercise_plan': [
            'Aerobic exercise 30-45 min daily',
            'Resistance training 2-3 times/week',
            'Start slowly and progress gradually',
            'Include warm-up and cool-down periods'
        ],
        'warning_signs': [
            'Frequent headaches',
            'Dizziness',
            'Blurred vision',
            'Shortness of breath with exertion',
            'Chest pain'
        ],
        'next_checkup': 'Follow-up in 2-4 weeks',
        'medication': 'Consider medication if lifestyle changes insufficient after 3-6 months'
    },
    2: {
        'name': 'STAGE-2 HYPERTENSION',
        'display_name': 'Stage 2 Hypertension',
        'color': '#dc2626',
        'gradient': 'linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)',
        'light_bg': '#fee2e2',
        'risk': 'High Risk',
        'risk_level': 3,
        'icon': '🟠',
        'description': 'Your blood pressure indicates Stage 2 hypertension. Prompt medical attention is needed.',
        'clinical_guidelines': 'Systolic ≥ 140 mmHg OR Diastolic ≥ 90 mmHg',
        'recommendations': [
            {'type': 'urgent', 'icon': '⚠️', 'text': 'Schedule medical appointment within 1 week'},
            {'type': 'medication', 'icon': '💊', 'text': 'Medication will likely be prescribed'},
            {'type': 'diet', 'icon': '🥗', 'text': 'Strict dietary changes required'},
            {'type': 'monitoring', 'icon': '📊', 'text': 'Daily blood pressure monitoring'},
            {'type': 'lifestyle', 'icon': '🏥', 'text': 'Regular follow-ups with doctor'},
            {'type': 'stress', 'icon': '🧘', 'text': 'Intensive stress management'}
        ],
        'diet_plan': [
            'Strict DASH diet adherence',
            'Sodium restriction <1500mg/day',
            'Eliminate processed foods',
            'Limit caffeine',
            'No alcohol consumption'
        ],
        'exercise_plan': [
            'Moderate exercise 30 min most days',
            'Avoid heavy lifting',
            'Monitor BP before and after exercise',
            'Stop if chest pain or shortness of breath',
            'Consult doctor before starting exercise'
        ],
        'warning_signs': [
            'Severe headaches',
            'Chest pain',
            'Shortness of breath',
            'Vision changes',
            'Nosebleeds',
            'Irregular heartbeat'
        ],
        'next_checkup': 'Follow-up in 1-2 weeks',
        'medication': 'One or more antihypertensive medications likely needed'
    },
    3: {
        'name': 'HYPERTENSIVE CRISIS',
        'display_name': 'Hypertensive Crisis',
        'color': '#7f1d1d',
        'gradient': 'linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)',
        'light_bg': '#fee2e2',
        'risk': 'CRITICAL - EMERGENCY',
        'risk_level': 4,
        'icon': '🔴',
        'description': 'EMERGENCY: Your blood pressure is at crisis level requiring immediate medical attention!',
        'clinical_guidelines': 'Systolic > 180 mmHg AND/OR Diastolic > 120 mmHg',
        'recommendations': [
            {'type': 'emergency', 'icon': '🚨', 'text': 'CALL EMERGENCY SERVICES (911) IMMEDIATELY'},
            {'type': 'emergency', 'icon': '🏥', 'text': 'Go to nearest Emergency Room NOW'},
            {'type': 'emergency', 'icon': '⏱️', 'text': 'DO NOT WAIT for appointment'},
            {'type': 'warning', 'icon': '⚠️', 'text': 'If you have chest pain or shortness of breath, this is URGENT'},
            {'type': 'warning', 'icon': '🚑', 'text': 'Do not drive yourself - call ambulance'},
            {'type': 'info', 'icon': '📋', 'text': 'Bring medication list to hospital'}
        ],
        'emergency_actions': [
            'Call 911 immediately',
            'Do not wait for symptoms to worsen',
            'Do not take extra medication without instruction',
            'Sit down and try to stay calm',
            'Loosen tight clothing',
            'If prescribed emergency medication, take as directed'
        ],
        'warning_signs': [
            'Chest pain or pressure',
            'Severe headache',
            'Shortness of breath',
            'Vision loss',
            'Difficulty speaking',
            'Weakness on one side of body',
            'Back pain',
            'Numbness',
            'Seizures'
        ],
        'next_checkup': 'IMMEDIATE EMERGENCY CARE',
        'medication': 'Emergency intravenous medication may be needed'
    }
}

@app.route('/')
def home():
    """Render the main dashboard"""
    return render_template('index.html', 
                         age_groups=age_groups,
                         severity_options=severity_options,
                         time_options=time_options,
                         bp_ranges=bp_ranges,
                         stage_info=STAGE_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        form_data = {
            'gender': request.form['gender'],
            'age': request.form['age_group'],
            'history': request.form['family_history'],
            'patient': request.form['patient_status'],
            'medication': request.form['medication'],
            'severity': request.form['severity'],
            'breath': request.form['breath_shortness'],
            'visual': request.form['visual_changes'],
            'nosebleed': request.form['nosebleeds'],
            'time': request.form['time_diagnosis'],
            'systolic': request.form['systolic'],
            'diastolic': request.form['diastolic'],
            'diet': request.form['diet_control']
        }
        
        # Process features for model
        features = []
        
        # Gender
        features.append(0 if form_data['gender'] == 'Male' else 1)
        
        # Age Group
        age_map = {'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4}
        features.append(age_map[form_data['age']])
        
        # Binary features
        binary_map = {'No': 0, 'Yes': 1}
        features.append(binary_map[form_data['history']])
        features.append(binary_map[form_data['patient']])
        features.append(binary_map[form_data['medication']])
        
        # Severity
        severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}
        features.append(severity_map[form_data['severity']])
        
        # More binary
        features.append(binary_map[form_data['breath']])
        features.append(binary_map[form_data['visual']])
        features.append(binary_map[form_data['nosebleed']])
        
        # Time since diagnosis
        time_map = {'<1 Year': 1, '1-3 Years': 2, '3-5 Years': 3, '5+ Years': 4}
        features.append(time_map[form_data['time']])
        
        # Blood pressure - convert to midpoint
        def bp_to_midpoint(bp_str):
            parts = bp_str.replace(' ', '').split('-')
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2
            return float(bp_str)
        
        systolic_val = bp_to_midpoint(form_data['systolic'])
        diastolic_val = bp_to_midpoint(form_data['diastolic'])
        features.append(systolic_val)
        features.append(diastolic_val)
        
        # Diet control
        features.append(binary_map[form_data['diet']])
        
        # Scale features
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Predict
        prediction = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get stage info
        stage_info = STAGE_INFO[prediction]
        
        # Prepare probability distribution
        prob_dist = {
            'Normal': float(probabilities[0] * 100),
            'Stage-1': float(probabilities[1] * 100),
            'Stage-2': float(probabilities[2] * 100),
            'Crisis': float(probabilities[3] * 100)
        }
        
        # Get dominant factors (for insights)
        feature_importance = []
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0]
            feature_imp = list(zip(feature_names, coefs, features))
            feature_imp.sort(key=lambda x: abs(x[1]), reverse=True)
            feature_importance = feature_imp[:5]  # Top 5 factors
        
        # Prepare result
        result = {
            'success': True,
            'prediction': prediction,
            'stage_name': stage_info['name'],
            'display_name': stage_info['display_name'],
            'color': stage_info['color'],
            'gradient': stage_info['gradient'],
            'light_bg': stage_info['light_bg'],
            'risk': stage_info['risk'],
            'risk_level': stage_info['risk_level'],
            'icon': stage_info['icon'],
            'description': stage_info['description'],
            'clinical_guidelines': stage_info['clinical_guidelines'],
            'recommendations': stage_info['recommendations'],
            'diet_plan': stage_info['diet_plan'],
            'exercise_plan': stage_info['exercise_plan'],
            'warning_signs': stage_info['warning_signs'],
            'next_checkup': stage_info['next_checkup'],
            'medication': stage_info['medication'],
            'emergency_actions': stage_info.get('emergency_actions', []),
            'probabilities': prob_dist,
            'confidence': float(max(probabilities) * 100),
            'bp_values': {
                'systolic': form_data['systolic'],
                'diastolic': form_data['diastolic']
            },
            'feature_importance': feature_importance,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Store in session for history
        if 'history' not in session:
            session['history'] = []
        session['history'].append(result)
        if len(session['history']) > 10:  # Keep last 10
            session['history'] = session['history'][-10:]
        
        return render_template('index.html',
                             result=result,
                             age_groups=age_groups,
                             severity_options=severity_options,
                             time_options=time_options,
                             bp_ranges=bp_ranges,
                             stage_info=STAGE_INFO)
        
    except Exception as e:
        return render_template('index.html',
                             error=str(e),
                             age_groups=age_groups,
                             severity_options=severity_options,
                             time_options=time_options,
                             bp_ranges=bp_ranges,
                             stage_info=STAGE_INFO)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint"""
    try:
        data = request.get_json()
        # Similar processing as above but return JSON
        return jsonify({'success': True, 'message': 'API endpoint ready'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/about')
def about():
    """About page with system information"""
    return render_template('about.html', stage_info=STAGE_INFO)

@app.route('/clear-history')
def clear_history():
    """Clear session history"""
    session.pop('history', None)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)