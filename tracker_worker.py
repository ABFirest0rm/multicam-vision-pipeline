from get_models_transforms import get_tracker
from tracker_fn import track
from time import perf_counter, sleep
import torch
from datetime import datetime,timedelta
from frame_skip_fun import safe_put_to_queue
import numpy as np
import logging
import _queue
import random
# from queue_clear_function import clear_queues

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler()  # optional: keep for minimal console
    ]
)
log = logging.getLogger("realtime")

def tracker_worker(yolo_to_tracker_q, tracker_to_display_q,
                   track_stop, track_reset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    last_reset = datetime.now()
    reset_interval = timedelta(seconds=36000)
    curr_id = 0
    new_id = 0

    botsort_tracker = get_tracker(name='boosttrack', device=device)
    # botsort_tracker = get_tracker(name='botsort', device=device)

    # alt_frame = True

    with torch.inference_mode():
        while True:
            if track_stop.value:
                print("[TRACKER] Stop Received")
                break

            if track_reset.value:
                print("[TRACKER] RESET SIGNAL RECEIVED")
                log.info("[TRACKER] RESET SIGNAL RECEIVED")
                print(f"TRACKER curr id {curr_id}")
                if (datetime.now() - last_reset) >= reset_interval:
                    botsort_tracker = get_tracker(name='boosttrack', device=device)
                    last_reset = datetime.now()
                    while curr_id == new_id:
                        new_id = random.randint(0, 5)
                    print(f"TRACKER new id {new_id}")
                    curr_id = new_id

                track_reset.value = False
                print(f"RESET TRACKER VALUE IN WORKER {track_stop.value}")
                # logging.info(f"RESET TRACKER VALUE IN WORKER {track_stop.value}")

            try:
                frame_get = perf_counter()
                yolo_detections, frame_array, timestamp, cam_id, alt_frame = yolo_to_tracker_q.get(timeout=1)  # Wait up to 1 second for a frame
                # print(f"[TRACKER] Frame get {(perf_counter() - frame_get) * 1000:.2f}ms")

            except _queue.Empty:
                # print("[TRACKER] Frame queue is empty.")
                if track_stop.value:
                    print("[TRACKER] Stop Received")
                    break
                sleep(0.01)
                continue
            # alt_frame = True
            if alt_frame:
                people_tracks, vehicle_tracks  = track(tracker=botsort_tracker,
                                                       detections_numpy=yolo_detections,
                                                       frame_rgb=frame_array)
                # print(f"[TRACKER] Tracking {(perf_counter() - td) * 1000:.2f}ms")

            else:
                people_tracks =  np.empty((0, 8), dtype=np.float32)
                vehicle_tracks = np.empty((0, 8), dtype=np.float32)


            safe_put_to_queue(tracker_to_display_q, (people_tracks,
                                                     yolo_detections,
                                                     frame_array,
                                                     timestamp,
                                                     cam_id, curr_id))
        # alt_frame = not alt_frame

    print("[TRACKER] STOPPED")