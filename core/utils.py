import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
from django.conf import settings

# Global variables to hold models and encoders
_models = None
_label_encoder_sex = None
_label_encoder_activity = None
_targets = ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fats (g)', 'Water (L)',
            'Vitamin A (mcg)', 'Vitamin B12 (mcg)', 'Vitamin C (mg)', 'Vitamin D (IU)',
            'Calcium (mg)', 'Iron (mg)', 'Magnesium (mg)', 'Zinc (mg)', 'Omega-3 (mg)']

def get_ml_resources():
    global _models, _label_encoder_sex, _label_encoder_activity
    
    if _models is not None:
        return _models, _label_encoder_sex, _label_encoder_activity
    
    # Load the dataset
    data_path = os.path.join(settings.BASE_DIR, 'data', 'nutritional_requirements_extended.csv')
    
    if not os.path.exists(data_path):
        # Fallback for different structures
        data_path = os.path.join(settings.BASE_DIR, 'nutrition app', 'nutritional_requirements_extended.csv')
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    data = pd.read_csv(data_path)
    data.rename(columns=lambda x: x.strip(), inplace=True)
    
    # Label Encoding
    _label_encoder_sex = LabelEncoder()
    _label_encoder_activity = LabelEncoder()
    
    data['Sex'] = _label_encoder_sex.fit_transform(data['Sex'])
    data['Activity Level'] = _label_encoder_activity.fit_transform(data['Activity Level'])
    data['Pregnant'] = data['Pregnant'].replace(['NULL', np.nan], 0).astype(int)
    
    features = ['Age', 'Height (cm)', 'Weight (kg)', 'Sex', 'Activity Level', 'Pregnant']
    X = data[features]
    y = data[_targets]
    
    # Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    _models = {}
    for target in _targets:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train[target])
        _models[target] = model
        
    return _models, _label_encoder_sex, _label_encoder_activity

def predict_nutritional_requirements(age, height, weight, activity, sex, pregnant=0):
    try:
        models, le_sex, le_activity = get_ml_resources()
        
        # Encode inputs
        sex_encoded = le_sex.transform([sex])[0]
        activity_encoded = le_activity.transform([activity])[0]
        
        # Create input array
        user_input = np.array([[age, height, weight, sex_encoded, activity_encoded, pregnant]])
        
        # Predict all targets
        predictions = {}
        for target, model in models.items():
            predictions[target] = float(model.predict(user_input)[0])
        
        return predictions
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

def get_supplement_data():
    return {
        'Vitamin A': {
            'natural_sources': ['Carrots', 'Sweet potatoes', 'Spinach', 'Kale', 'Liver'],
            'benefits': ['Vision health', 'Immune system', 'Skin health'],
            'icon': '👁️'
        },
        'Vitamin B12': {
            'natural_sources': ['Fish', 'Meat', 'Eggs', 'Dairy products'],
            'benefits': ['Red blood cell formation', 'Nerve function', 'DNA synthesis'],
            'icon': '🧠'
        },
        'Vitamin C': {
            'natural_sources': ['Oranges', 'Strawberries', 'Bell peppers', 'Broccoli'],
            'benefits': ['Immune system', 'Collagen formation', 'Antioxidant'],
            'icon': '🛡️'
        },
        'Vitamin D': {
            'natural_sources': ['Sunlight', 'Fatty fish', 'Egg yolks', 'Fortified dairy'],
            'benefits': ['Bone health', 'Calcium absorption', 'Immune function'],
            'icon': '☀️'
        },
        'Calcium': {
            'natural_sources': ['Dairy products', 'Leafy greens', 'Sardines', 'Tofu'],
            'benefits': ['Bone health', 'Muscle function', 'Nerve signaling'],
            'icon': '🦴'
        },
        'Iron': {
            'natural_sources': ['Red meat', 'Beans', 'Spinach', 'Fortified cereals'],
            'benefits': ['Red blood cell formation', 'Oxygen transport', 'Energy production'],
            'icon': '🩸'
        },
        'Magnesium': {
            'natural_sources': ['Nuts', 'Seeds', 'Legumes', 'Whole grains', 'Dark chocolate'],
            'benefits': ['Muscle and nerve function', 'Energy production', 'Bone health'],
            'icon': '⚡'
        },
        'Zinc': {
            'natural_sources': ['Oysters', 'Meat', 'Legumes', 'Seeds', 'Nuts'],
            'benefits': ['Immune function', 'Wound healing', 'DNA synthesis'],
            'icon': '🔬'
        },
        'Omega-3': {
            'natural_sources': ['Fatty fish', 'Flaxseeds', 'Chia seeds', 'Walnuts'],
            'benefits': ['Heart health', 'Brain function', 'Reduced inflammation'],
            'icon': '🐟'
        }
    }
