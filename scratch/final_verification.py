import os
import django
import sys

# Setup Django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jeevansetu.settings')
django.setup()

try:
    from core.utils import get_ml_resources, predict_nutritional_requirements
    print("Testing ML Resource Loading...")
    models, le_sex, le_act = get_ml_resources()
    print("SUCCESS: ML Resources loaded/trained successfully")
    
    print("\nTesting Prediction...")
    # Age=25, Height=175, Weight=70, Activity='Moderate', Sex='Male'
    preds = predict_nutritional_requirements(25, 175, 70, 'Moderate', 'Male')
    if preds:
        print("SUCCESS: Prediction successful!")
        print(f"Sample - Calories: {preds.get('Calories (kcal)', 'N/A')}")
    else:
        print("FAILED: Prediction returned None")
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
