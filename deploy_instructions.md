# Deployment Instructions

## Deploy on Render (Free)
1. Create account at https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - Name: predictive-pulse
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Click "Create Web Service"

## Deploy on Railway (Free)
1. Create account at https://railway.app
2. Click "New Project" → "Deploy from GitHub"
3. Select your repository
4. Railway auto-detects Python and deploys

## Deploy on PythonAnywhere (Free)
1. Create account at https://pythonanywhere.com
2. Go to "Web" tab → "Add a new web app"
3. Select manual configuration → Python 3.9
4. Upload files via FTP or GitHub
5. Configure WSGI file to point to app.py