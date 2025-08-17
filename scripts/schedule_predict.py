from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from pathlib import Path
import time, subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INBOX = PROJECT_ROOT / "data" / "processed" / "incoming"
OUTBOX = PROJECT_ROOT / "data" / "processed" / "predictions"
INBOX.mkdir(parents=True, exist_ok=True); OUTBOX.mkdir(parents=True, exist_ok=True)

def run_batch_on_new_files():
    for csv in INBOX.glob("*.csv"):
        out = OUTBOX / (csv.stem + "_pred.csv")
        print(f"[{datetime.now().isoformat()}] Predicting {csv.name} -> {out.name}")
        subprocess.run(["python","-m","utils.batch_predict", str(csv), str(out), "--model","xgb"], check=False)

def main():
    print("Watching for CSVs in:", INBOX)
    sched = BackgroundScheduler()
    sched.add_job(run_batch_on_new_files, 'interval', seconds=30)
    sched.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        sched.shutdown()

if __name__ == "__main__":
    main()