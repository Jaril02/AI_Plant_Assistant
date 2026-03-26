# 🌿 AI Plant Assistant

An AI-powered web application to detect plant diseases using image classification and Google Gemini AI for detailed insights, reminders, and PDF reporting. Designed for farmers, researchers, and plant enthusiasts to monitor plant health efficiently.

---

## Features
- Upload plant leaf asserts for disease detection
- Classify diseases using a trained TensorFlow model
- Get detailed AI-generated disease information via Google Gemini AI
- Download comprehensive PDF reports of the analysis
- Receive scheduled reminders for plant care tasks via SMS
- Ask follow-up questions about plant diseases

---

## Tech Stack
- **Frontend & Web UI:** Streamlit
- **Backend & ML:** Python, TensorFlow
- **AI & NLP:** Google Gemini API
- **Scheduler & Notifications:** APScheduler, Twilio (SMS)
- **PDF Reports:** FPDF
- **Environment Management:** dotenv
- **Containerization:** Docker & Docker Compose

---

## Project Structure


Ai_plant/
├── app/ # PDF generator and helper scripts
├── databases/ # SQLite DB and database utilities
├── models/ # Trained ML models
├── reports/ # Generated reports
├── scripts/ # Misc JSON and helper files
├── trained_model/ # Saved TensorFlow models
├── weather/ # Scheduler & notification scripts
├── genai.py # Main app entrypoint
├── requirements.txt
├── docker-compose.yml
├── .dockerignore
└── README.md


---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Jaril02/AI_Plant_Assistant.git
cd AI_Plant_Assistant

2. Create .env file

Add your environment variables (API keys, Twilio credentials, etc.)

GOOGLE_API_KEY=<your_google_gemini_key>
TWILIO_SID=<your_twilio_sid>
TWILIO_AUTH_TOKEN=<your_twilio_auth_token>
3. Run Locally (Python)

Install dependencies:

pip install -r requirements.txt

Run the main app:

python genai.py

Run the scheduler (optional):

python weather/scheduler.py
4. Run with Docker

Build and run the containers:

docker-compose up --build
App: http://localhost:8501
Scheduler: runs in the background, triggering plant reminders in real-time
Notes
The scheduler checks the database every minute and sends SMS reminders if tasks are due.
PDF reports are generated in the reports/ folder.
Large datasets, trained models, and temporary files are ignored using .dockerignore and .gitignore.
Contributing
Open issues or submit pull requests.
Ensure new dependencies are added to requirements.txt and tested with Docker.


![Home Page](asserts/home.png)
![Leaf Upload ](asserts/leaf_upload.png)
![Ai Assistance](asserts/Ai_assistant.png)
![Weather](asserts/weather.png)
![Confidence](asserts/confidence.png)
![Scheduler](asserts/scheduler.png)
![Analyze Result](asserts/output1.png)
![Result 2](asserts/output2.png)
![PDF Generator](asserts/pdf_genrator.png)