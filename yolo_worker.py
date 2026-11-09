import torch
from yolo_fn import detect_yolo
from time import perf_counter, sleep
from get_models_transforms import get_yolo
from frame_skip_fun import safe_put_to_queue
import numpy as np
import _queue

def yolo_worker(cap_to_yolo_q, yolo_to_tracker_q,
                yolo_stop, width=1280, height=720):
    print("[YOLO] Yolo started")
    x_ratio = width/640
    y_ratio = height/640
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolo, yolo_transform = get_yolo(device=device)
    alt_frame = True
    with torch.inference_mode():
        while True:
            if yolo_stop.value:
                # print("[YOLO] Stop Received")
                break

            try:
                frame_get = perf_counter()
                frame_array, timestamp, cam_id = cap_to_yolo_q.get(timeout=0.5)  # Wait up to 1 second for a frame
                # print(f"[YOLO] Frame get {(perf_counter() - frame_get) * 1000:.2f}ms")

            except _queue.Empty:
#                 print("[YOLO] Frame queue is empty.")
                if yolo_stop.value:
#                     print("[YOLO] Stop Received")
                    break
                sleep(0.01)
                continue
            # yolo_det = perf_counter()

            if alt_frame:
                yolo_detections = detect_yolo(model=yolo,
                                              transform=yolo_transform,
                                              x_ratio=x_ratio, y_ratio=y_ratio,  # # original / resized
                                              frame_rgb=frame_array, conf_thresh=0.5, device=device)
            # print(f"[YOLO] Detection {(perf_counter() - yolo_det) * 1000:.2f}ms")

            else:
                yolo_detections = np.empty((0, 6))

            # print("YOLO sent to tracker")
            safe_put_to_queue(yolo_to_tracker_q, (yolo_detections,
                                                  frame_array,
                                                  timestamp,
                                                  cam_id,
                                                  alt_frame))
            alt_frame = not alt_frame

    print("[YOLO] STOPPED")