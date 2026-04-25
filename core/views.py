from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from .utils import predict_nutritional_requirements, get_supplement_data

def index(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'core/index.html')

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'core/login.html', {'form': form})

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
        else:
            for error in form.errors.values():
                messages.error(request, error)
    else:
        form = UserCreationForm()
    return render(request, 'core/register.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    return redirect('index')

@login_required
def dashboard(request):
    predictions = None
    gap_analysis = None
    from .utils import get_motivational_quotes, adjust_predictions_by_goal
    quotes = get_motivational_quotes()
    
    if request.method == 'POST':
        try:
            age = int(request.POST.get('age'))
            height = float(request.POST.get('height'))
            weight = float(request.POST.get('weight'))
            activity = request.POST.get('activity')
            sex = request.POST.get('sex')
            goal = request.POST.get('goal')
            pregnant = 1 if request.POST.get('pregnant') == 'yes' and sex == 'Female' else 0
            
            # Get base predictions
            raw_predictions = predict_nutritional_requirements(age, height, weight, activity, sex, pregnant)
            
            if raw_predictions:
                # Adjust based on goal
                predictions = adjust_predictions_by_goal(raw_predictions, goal)
                
                # Categorize results for UI
                from .utils import get_categorized_predictions
                predictions_categorized = get_categorized_predictions(predictions)
                
                # Calculate Deficiency (Gap)
                current_intake = {
                    'Calories (kcal)': float(request.POST.get('current_calories') or 0),
                    'Protein (g)': float(request.POST.get('current_protein') or 0),
                    'Carbohydrates (g)': float(request.POST.get('current_carbs') or 0),
                    'Fats (g)': float(request.POST.get('current_fats') or 0),
                }
                
                gap_analysis = {}
                for key in current_intake:
                    if current_intake[key] > 0:
                        required = predictions.get(key, 0)
                        diff = required - current_intake[key]
                        gap_analysis[key] = {
                            'required': required,
                            'current': current_intake[key],
                            'gap': diff if diff > 0 else 0,
                            'surplus': -diff if diff < 0 else 0,
                            'percentage': (current_intake[key] / required * 100) if required > 0 else 0
                        }
                
                # Identify top deficiencies (where percentage < 90)
                top_deficiencies = []
                if gap_analysis:
                    # Sort by percentage met (lowest first)
                    sorted_gaps = sorted(gap_analysis.items(), key=lambda x: x[1]['percentage'])
                    top_deficiencies = [
                        {'name': name, 'gap': data['gap'], 'percentage': data['percentage']}
                        for name, data in sorted_gaps if data['percentage'] < 90
                    ]

                return render(request, 'core/dashboard.html', {
                    'predictions': predictions_categorized,
                    'gap_analysis': gap_analysis,
                    'raw_predictions': predictions,
                    'top_deficiencies': top_deficiencies,
                    'quotes': quotes
                })
                
            if not predictions:
                messages.error(request, "Error making prediction. Please check your inputs.")
        except Exception as e:
            messages.error(request, f"Error: {str(e)}")
            
    return render(request, 'core/dashboard.html', {
        'predictions': predictions,
        'gap_analysis': gap_analysis,
        'quotes': quotes
    })

@login_required
def supplements(request):
    supplement_data = get_supplement_data()
    return render(request, 'core/supplements.html', {'supplement_data': supplement_data})
