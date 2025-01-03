from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import time
from datetime import datetime

class CSVHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('oect_summary_posted_rf__plus_ml_combined.csv'):
            print(f"CSV file updated at {datetime.now()}")

def monitor_csv():
    csv_path = "datasets"  # Directory containing your CSV
    event_handler = CSVHandler()
    observer = Observer()
    observer.schedule(event_handler, path=csv_path, recursive=False)
    
    print(f"Starting CSV monitoring in {csv_path}")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopping CSV monitoring...")
    
    observer.join()

if __name__ == "__main__":
    monitor_csv()