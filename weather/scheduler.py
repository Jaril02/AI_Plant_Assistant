from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime   
import time
from databases.database import get_schedules, init_db
from weather.notify import send_sms  # your SMS sending function

scheduler= BackgroundScheduler()
scheduled_jobs = set()


def send_task(name,phone,task,task_time):
    print(f"[TRIGGERED] Sending to {name}")
    send_sms(phone, f"Reminder: {task} scheduled at {task_time}")

def load_and_schedule_tasks():
    tasks = get_schedules()

    if not tasks:
        print("[INFO] No tasks found")
        return

    for name, phone, task, task_time, freq, day in tasks:
        job_id = f"{name}_{task}_{task_time}"

        if job_id in scheduled_jobs:
            continue  # already scheduled

        hour, minute = map(int, task_time.split(":"))
        if job_id in scheduled_jobs:
            continue
        scheduler.add_job(
            send_task,
            trigger='cron',   # ⏰ runs daily at given time
            hour=hour,
            minute=minute,
            args=[name, phone, task, task_time],
            id=job_id,
            replace_existing=True
        )
        scheduled_jobs.add(job_id)    
        print(f"[SYNCED] {task} for {name} at {task_time}")



def start_scheduler():
    print("🚀 APScheduler Started")

    # Load tasks initially
    load_and_schedule_tasks()

    # 🔁 Optional: reload tasks every 1 minute (if new tasks added)
    scheduler.add_job(load_and_schedule_tasks, 'interval', seconds=10)

    scheduler.start()

# if __name__ == "__main__":
#     start_scheduler()
#     while True:
#         time.sleep(10)
    
    
    
