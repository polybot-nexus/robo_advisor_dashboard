import cv2
import numpy as np
import pyautogui
import time
import os

save_dir = "frames"  
os.makedirs(save_dir, exist_ok=True)
interval = 1         
duration = 1800  

screen_size = pyautogui.size()

start_time = time.time()
frame_count = 0

while True:
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_path = os.path.join(save_dir, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_path, frame)
    frame_count += 1

    time.sleep(interval)

    if time.time() - start_time > duration:
        break

print(f"Saved {frame_count} frames in '{save_dir}' folder.")
