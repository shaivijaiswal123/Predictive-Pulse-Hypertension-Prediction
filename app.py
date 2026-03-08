"""
Predictive Pulse - Fixed Flask Web Application
"""

from flask import Flask, render_template, request
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)

# Load model
try:
    if os.path.exists('models/logreg_model.pkl'):
        model = pickle.load(open('models/logreg_model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        feature_names = pickle.load(open('models/feature_names.pkl', 'rb'))
        print("✅ All model files loaded from models/ folder!")
    else:
        model = pickle.load(open('logreg_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))
        print("✅ All model files loaded from root folder!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Stage information (MUST be defined BEFORE routes)
STAGE_INFO = {
    0: {
        'name': 'NORMAL',
        'display_name': 'Normal Blood Pressure',
        'color': '#10b981',
        'gradient': 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
        'light_bg': '#d1fae5',
        'risk': 'Low Risk',
        'icon': '🟢',
        'description': 'Your blood pressure is within the optimal range.',
        'clinical_guidelines': 'Systolic < 120 mmHg AND Diastolic < 80 mmHg',
        'recommendations': [
            {'text': 'Continue healthy lifestyle habits'},
            {'text': 'Exercise 30 min/day, 5 days/week'},
            {'text': 'Keep salt intake below 5g per day'},
            {'text': 'Check blood pressure annually'}
        ],
        'next_checkup': 'Annual physical examination',
        'medication': 'No medication needed'
    },
    1: {
        'name': 'STAGE-1 HYPERTENSION',
        'display_name': 'Stage 1 Hypertension',
        'color': '#f59e0b',
        'gradient': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
        'light_bg': '#fef3c7',
        'risk': 'Moderate Risk',
        'icon': '🟡',
        'description': 'Your blood pressure indicates Stage 1 hypertension.',
        'clinical_guidelines': 'Systolic 130-139 mmHg OR Diastolic 80-89 mmHg',
        'recommendations': [
            {'text': 'Consult healthcare provider within 1 month'},
            {'text': 'Start lifestyle modifications'},
            {'text': 'Reduce sodium intake to <4g per day'},
            {'text': 'Monitor BP weekly at home'}
        ],
        'next_checkup': 'Follow-up in 2-4 weeks',
        'medication': 'Consider medication if lifestyle changes insufficient'
    },
    2: {
        'name': 'STAGE-2 HYPERTENSION',
        'display_name': 'Stage 2 Hypertension',
        'color': '#dc2626',
        'gradient': 'linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)',
        'light_bg': '#fee2e2',
        'risk': 'High Risk',
        'icon': '🟠',
        'description': 'Your blood pressure indicates Stage 2 hypertension.',
        'clinical_guidelines': 'Systolic ≥ 140 mmHg OR Diastolic ≥ 90 mmHg',
        'recommendations': [
            {'text': 'Schedule medical appointment within 1 week'},
            {'text': 'Medication will likely be prescribed'},
            {'text': 'Daily blood pressure monitoring'},
            {'text': 'Strict dietary changes required'}
        ],
        'next_checkup': 'Follow-up in 1-2 weeks',
        'medication': 'One or more medications likely needed'
    },
    3: {
        'name': 'HYPERTENSIVE CRISIS',
        'display_name': 'Hypertensive Crisis',
        'color': '#7f1d1d',
        'gradient': 'linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)',
        'light_bg': '#fee2e2',
        'risk': 'CRITICAL - EMERGENCY',
        'icon': '🔴',
        'description': 'EMERGENCY: Your blood pressure is at crisis level!',
        'clinical_guidelines': 'Systolic > 180 mmHg AND/OR Diastolic > 120 mmHg',
        'recommendations': [
            {'text': 'CALL EMERGENCY SERVICES (911) IMMEDIATELY'},
            {'text': 'Go to nearest Emergency Room NOW'},
            {'text': 'DO NOT WAIT for appointment'},
            {'text': 'Do not drive yourself - call ambulance'}
        ],
        'emergency_actions': [
            'Call 911 immediately',
            'Do not wait for symptoms to worsen',
            'Go to ER immediately',
            'Stay calm and sit down'
        ],
        'next_checkup': 'IMMEDIATE EMERGENCY CARE',
        'medication': 'Emergency treatment required'
    }
}

# Helper function to convert BP range to midpoint
def bp_to_midpoint(bp_str):
    try:
        bp_str = bp_str.replace(' ', '')
        if '-' in bp_str:
            parts = bp_str.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return float(bp_str)
    except:
        return 120  # default fallback

@app.route('/')
def home():
    """Render the main dashboard"""
    age_groups = ['18-34', '35-50', '51-64', '65+']
    severity_options = ['Mild', 'Moderate', 'Severe']
    time_options = ['<1 Year', '1-3 Years', '3-5 Years', '5+ Years']
    bp_ranges = {
        'systolic': ['90-100', '101-110', '111-120', '121-130', '131-140', '141-150', 
                     '151-160', '161-170', '171-180', '181-190', '191-200'],
        'diastolic': ['60-70', '71-80', '81-90', '91-100', '101-110', '111-120']
    }
    
    # IMPORTANT: Pass stage_info to template
    return render_template('index.html', 
                         age_groups=age_groups,
                         severity_options=severity_options,
                         time_options=time_options,
                         bp_ranges=bp_ranges,
                         stage_info=STAGE_INFO)  # This was missing!

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data with defaults
        gender = 0 if request.form.get('gender', 'Male') == 'Male' else 1
        
        age_map = {'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4}
        age = age_map.get(request.form.get('age_group', '18-34'), 1)
        
        binary_map = {'No': 0, 'Yes': 1}
        family_history = binary_map.get(request.form.get('family_history', 'No'), 0)
        patient_status = binary_map.get(request.form.get('patient_status', 'No'), 0)
        medication = binary_map.get(request.form.get('medication', 'No'), 0)
        
        severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}
        severity = severity_map.get(request.form.get('severity', 'Mild'), 0)
        
        breath = binary_map.get(request.form.get('breath_shortness', 'No'), 0)
        visual = binary_map.get(request.form.get('visual_changes', 'No'), 0)
        nosebleed = binary_map.get(request.form.get('nosebleeds', 'No'), 0)
        
        time_map = {'<1 Year': 1, '1-3 Years': 2, '3-5 Years': 3, '5+ Years': 4}
        time_diag = time_map.get(request.form.get('time_diagnosis', '<1 Year'), 1)
        
        systolic = bp_to_midpoint(request.form.get('systolic', '111-120'))
        diastolic = bp_to_midpoint(request.form.get('diastolic', '81-90'))
        
        diet = binary_map.get(request.form.get('diet_control', 'No'), 0)
        
        # Create features array in correct order
        features = [
            gender, age, family_history, patient_status, medication,
            breath, visual, nosebleed, diet, severity, time_diag,
            systolic, diastolic
        ]
        
        # Scale and predict
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get stage info
        stage_info_result = STAGE_INFO[prediction]
        
        # Prepare result
        result = {
            'success': True,
            'stage_name': stage_info_result['name'],
            'display_name': stage_info_result['display_name'],
            'color': stage_info_result['color'],
            'gradient': stage_info_result['gradient'],
            'light_bg': stage_info_result['light_bg'],
            'risk': stage_info_result['risk'],
            'icon': stage_info_result['icon'],
            'description': stage_info_result['description'],
            'clinical_guidelines': stage_info_result['clinical_guidelines'],
            'recommendations': stage_info_result['recommendations'],
            'next_checkup': stage_info_result['next_checkup'],
            'medication': stage_info_result['medication'],
            'emergency_actions': stage_info_result.get('emergency_actions', []),
            'confidence': float(max(probabilities) * 100),
            'probabilities': {
                'Normal': float(probabilities[0] * 100),
                'Stage-1': float(probabilities[1] * 100),
                'Stage-2': float(probabilities[2] * 100),
                'Crisis': float(probabilities[3] * 100)
            },
            'bp_values': {
                'systolic': request.form.get('systolic', '111-120'),
                'diastolic': request.form.get('diastolic', '81-90')
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Pass all template variables back
        age_groups = ['18-34', '35-50', '51-64', '65+']
        severity_options = ['Mild', 'Moderate', 'Severe']
        time_options = ['<1 Year', '1-3 Years', '3-5 Years', '5+ Years']
        bp_ranges = {
            'systolic': ['90-100', '101-110', '111-120', '121-130', '131-140', '141-150', 
                         '151-160', '161-170', '171-180', '181-190', '191-200'],
            'diastolic': ['60-70', '71-80', '81-90', '91-100', '101-110', '111-120']
        }
        
        return render_template('index.html',
                             result=result,
                             age_groups=age_groups,
                             severity_options=severity_options,
                             time_options=time_options,
                             bp_ranges=bp_ranges,
                             stage_info=STAGE_INFO)  # Pass stage_info here too
        
    except Exception as e:
        # If error, still render template with error message
        age_groups = ['18-34', '35-50', '51-64', '65+']
        severity_options = ['Mild', 'Moderate', 'Severe']
        time_options = ['<1 Year', '1-3 Years', '3-5 Years', '5+ Years']
        bp_ranges = {
            'systolic': ['90-100', '101-110', '111-120', '121-130', '131-140', '141-150', 
                         '151-160', '161-170', '171-180', '181-190', '191-200'],
            'diastolic': ['60-70', '71-80', '81-90', '91-100', '101-110', '111-120']
        }
        
        return render_template('index.html',
                             error=str(e),
                             age_groups=age_groups,
                             severity_options=severity_options,
                             time_options=time_options,
                             bp_ranges=bp_ranges,
                             stage_info=STAGE_INFO)  # Pass stage_info here too

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)