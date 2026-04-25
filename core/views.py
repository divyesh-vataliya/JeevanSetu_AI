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
    if request.method == 'POST':
        try:
            age = int(request.POST.get('age'))
            height = float(request.POST.get('height'))
            weight = float(request.POST.get('weight'))
            activity = request.POST.get('activity')
            sex = request.POST.get('sex')
            pregnant = 1 if request.POST.get('pregnant') == 'yes' and sex == 'Female' else 0
            
            predictions = predict_nutritional_requirements(age, height, weight, activity, sex, pregnant)
            
            if predictions:
                # Calculate Deficiency (Gap)
                # We assume the user might provide their current intake, or we can use a baseline
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
            
            if not predictions:
                messages.error(request, "Error making prediction. Please check your inputs.")
        except Exception as e:
            messages.error(request, f"Error: {str(e)}")
            
    return render(request, 'core/dashboard.html', {
        'predictions': predictions,
        'gap_analysis': gap_analysis
    })

@login_required
def supplements(request):
    supplement_data = get_supplement_data()
    return render(request, 'core/supplements.html', {'supplement_data': supplement_data})
