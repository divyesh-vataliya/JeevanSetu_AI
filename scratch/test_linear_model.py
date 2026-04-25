import os
import django
import sys

# Setup Django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jeevansetu.settings')
django.setup()

from core.utils import _targets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from django.conf import settings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Load data
data_path = os.path.join(settings.BASE_DIR, 'data', 'nutritional_requirements_extended.csv')
data = pd.read_csv(data_path)
data.rename(columns=lambda x: x.strip(), inplace=True)

# Prepare data
le_sex = LabelEncoder()
le_act = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])
data['Activity Level'] = le_act.fit_transform(data['Activity Level'])
data['Pregnant'] = data['Pregnant'].replace(['NULL', np.nan], 0).astype(int)
data['BMI'] = data['Weight (kg)'] / ((data['Height (cm)'] / 100) ** 2)

features = ['Age', 'Height (cm)', 'Weight (kg)', 'Sex', 'Activity Level', 'Pregnant', 'BMI']
X = data[features]
y = data[_targets]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
model.fit(X_train, y_train)

# Calculate R2
y_pred = model.predict(X_test)
scores = {}
for i, target in enumerate(_targets):
    scores[target] = r2_score(y_test[target], y_pred[:, i])

print("\nLinear Regression Accuracy (R2 Scores):")
for target, score in scores.items():
    print(f"{target}: {score:.4f}")

print(f"\nAverage R2 Score: {np.mean(list(scores.values())):.4f}")

# Check model size
import joblib
joblib.dump(model, 'scratch/linear_model.joblib')
print(f"\nModel file size: {os.path.getsize('scratch/linear_model.joblib') / 1024:.2f} KB")
