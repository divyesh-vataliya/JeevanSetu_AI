# JeevanSetu AI - Intelligent Nutritional Guidance

JeevanSetu AI is a modern, Django-powered web application that provides personalized nutritional and supplement recommendations using machine learning. It aims to bridge the gap between complex health data and actionable daily insights.

## ✨ Features

- **Personalized Analytics**: Get custom nutritional requirements based on your age, weight, height, sex, and activity level.
- **ML-Powered Predictions**: Uses a Random Forest Regressor trained on comprehensive nutritional datasets.
- **Supplements Guide**: Detailed information on essential vitamins and minerals, their benefits, and natural food sources.
- **Premium UI/UX**: Built with a modern, responsive design system featuring glassmorphism and smooth animations.
- **Secure Authentication**: Built-in Django authentication for personal health data protection.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/divyesh-vataliya/JeevanSetu_AI.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run migrations:
   ```bash
   python manage.py migrate
   ```

4. Start the development server:
   ```bash
   python manage.py runserver
   ```

5. Access the application at `http://127.0.0.1:8000/`

## 🛠️ Technology Stack

- **Backend**: Django (Python)
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, Vanilla CSS3 (Custom Design System)
- **Database**: SQLite (Default)

## 📊 How it Works

The application takes user physical profile inputs and passes them through a series of trained Random Forest models (one for each nutritional target). These models predict the optimal daily intake for calories, proteins, vitamins, and minerals.

---
Built with ❤️ for a healthier world.
