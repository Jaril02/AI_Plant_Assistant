from twilio.rest import Client
import os
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)

def send_sms(to, message):
    env_path = BASE_DIR/ ".env"

    load_dotenv(env_path)
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    client = Client(account_sid, auth_token)

    from_number = os.getenv("TWILIO_NUMBER")
    try:
        client.messages.create(
            body=message,
            from_=from_number,
            to=to
        )
        print(f"✅ SMS sent to {to}: {message}")
     
    except Exception as e:
        print(f"❌ SMS sending failed: {e}")
    # print("MOCK msg ver1")