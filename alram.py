from weather.scheduler import start_scheduler
import time

start_scheduler()

print("🟢 Scheduler is running...")

# 🔥 Keep script alive
try:
    while True:
        time.sleep(10)
except (KeyboardInterrupt, SystemExit):
    print("🔴 Scheduler stopped")