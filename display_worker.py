import cv2
from datetime import datetime, timedelta
from time import perf_counter, sleep
import _queue
from threading import Thread, Lock
import numpy as np
from frame_skip_fun import safe_put_to_queue
from display_helper_functions import display_mem_updater
from frame_skip_fun import safe_put_to_queue
# import logging

display_mem = dict()
display_mem_lock = Lock()
known_ids = []
last_reset = datetime.now()
reset_interval = timedelta(seconds=36000)
flags = [0, 0, 0] # tracker_id, known_ids_update, prev_id
#
# logging.basicConfig(
#     filename="fps_latency_log.txt",
#     filemode='a',  # 'a' = append
#     format='[%(asctime)s] %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     level=logging.INFO
# )


def display_worker(tracker_to_display_q, display_to_face_rec_queue,
                   face_rec_to_display_q,
                   display_stop, track_reset,
                   cam_id,
                   # tracker_to_display_q_lock,
                   # display_to_face_req_q_lock,
                   # output_queue,
                   # op_queue,
                   output_fps=24):


    # logging.basicConfig(
    #     filename=f"{cam_id}fps_latency_log.txt",
    #     filemode='a',  # 'a' = append
    #     format='[%(asctime)s] %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S',
    #     level=logging.INFO
    # )

    print("[DISPLAY] Loop started")
    frame_interval = 1 / output_fps
    fps_counter = 0
    frame_count = 0
    fps_display = 0
    fps_timer = datetime.now()
    # people_tracks = np.empty((0, 8))
    prev_bbox = np.empty((0, 8))
    global display_mem
    global known_ids
    global display_mem_lock
    clear_prev = 0
    known_ids_array = np.array(known_ids, dtype=np.int32)

    Thread(target=display_mem_updater, kwargs={'face_rec_to_display_q':face_rec_to_display_q,
                                               'track_reset':track_reset,
                                               'display_mem': display_mem, 'display_mem_lock': display_mem_lock,
                                               'reset_interval': reset_interval, 'known_ids': known_ids,
                                               'flags': flags
                                               },
           daemon=True).start()

    while True:
        # print(f"DISPLAY MEMORY {display_mem}")
        if display_stop.value:
            break

        curr_time = datetime.now()
        loop_start = perf_counter()
#         print("DISPLAY MEM WORKER", display_mem)

        try:
            frame_get = perf_counter()
            # with tracker_to_display_q_lock:
            people_tracks, yolo_detections, frame_array, timestamp, cam_id, curr_id = tracker_to_display_q.get(timeout=0.5)  # Wait up to 1 second for a frame
            # print(f"[DISPLAY] Frame get {(perf_counter() - frame_get) * 1000:.2f}ms")

            # print(f"DISPLAY SENDING TO FACE REC {(perf_counter() - pf)*1000:.2f}")


            # print("PEOPLE TRACKS", people_tracks)
            if people_tracks.size != 0:
                draw_bbox = people_tracks
                prev_bbox = people_tracks.copy()
            else:
                draw_bbox = prev_bbox
                clear_prev += 1
                if clear_prev == 2:
                    prev_bbox = np.empty((0, 8))
                    clear_prev = 0

            if flags[0] != curr_id:
                print(f"DISPLAY TRACKER ID CHANGED curr {curr_id} old {flags[0]}")
                with display_mem_lock:
                    print("DISPLAY -- MEMORY LOCKED")
                    print(f"DISPLAY display mem {display_mem}")
                    print(f"DISPLAY known_ids {known_ids}")
                    print(f"DISPLAY known_ids array {known_ids_array}")

                    display_mem.clear()
                    known_ids.clear()
                    known_ids_array = np.array(known_ids, dtype=np.int32)
                    flags[2] = flags[0]
                    flags[0] = curr_id

                    print(f"DISPLAY display mem {display_mem}")
                    print(f"DISPLAY known_ids {known_ids}")
                    print(f"DISPLAY known_ids array {known_ids_array}")
                    print(f"DISPLAY TRACKER ID CHANGED curr {curr_id} old {flags[0]} f2 {flags[2]}")


            with display_mem_lock:

                if flags[1] == 1:
                    # print(f"DISPLAY kids changed kid_array {known_ids_array}")
                    known_ids_array = np.array(known_ids, dtype=np.int32)
                    flags[1] = 0
#                     print(f"DISPLAY updated kid_array {known_ids_array}")

                if not len(known_ids) == 0:
                    for tbbox in draw_bbox[np.isin(
                            draw_bbox[:, 4], known_ids_array)]:
                        # print("DISPLAY tbbox")
                        x1, y1, x2, y2, tid, _, _, _ = tbbox
                        name = display_mem[tid]['name']
                        conf = display_mem[tid]['conf']
                        display_mem[tid]['last_seen'] = curr_time

                        # Draw bounding box
                        cv2.rectangle(frame_array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Draw label text above the box
                        # label = f"{tid} {name} ({conf:.2f})"
                        # cv2.putText(
                        #     frame_array, f"{tid} {name} ({conf:.2f})",
                        #     (int(x1), int(y1) - 10),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.6,
                        #     (0, 255, 0),
                        #     2,
                        #     lineType=cv2.LINE_AA
                        # )

                        cv2.putText(
                            frame_array, f"{name} ({conf:.2f})",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            lineType=cv2.LINE_AA
                        )


            elapsed_time = perf_counter() - loop_start
            sleep_time = frame_interval - elapsed_time
            if sleep_time > 0:
                sleep(sleep_time)

            fps_counter += 1
            if (curr_time - fps_timer).total_seconds() >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer = curr_time

            cv2.putText(frame_array, timestamp, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            fps_text = f"FPS: {fps_display}"
            cv2.putText(frame_array, fps_text, (10, 60),  # Adjust Y to be below timestamp
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # safe_put_to_queue(op_queue, (str(cam_id), cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)))

            display_time = datetime.now()
            latency_ms = (display_time - datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")).total_seconds() * 1000

            # print(f'{cam_id} FPS: {fps_display}|Latency: {latency_ms:.2f} ms')
            # print(f'{cam_id},{fps_display},{latency_ms:.2f}')

            cv2.imshow(str(cam_id), cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                display_stop.value = True  # Or set your global stop flag
                break

            # safe_put_to_queue(output_queue, (str(cam_id), cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)))

            if people_tracks.size > 0 and frame_count > 1:
                # pf = perf_counter()
                frame_count = 0
                safe_put_to_queue(display_to_face_rec_queue, (frame_array, people_tracks, timestamp, curr_id))

            frame_count += 1

        except _queue.Empty:
#             print("[DISPLAY] Frame queue is empty.")
            if display_stop.value:
                print("[DISPLAY] Stop Received")
                break
            sleep(0.01)
            continue
    # cv2.destroyAllWindows()
    print("[DISPLAY] STOPPED")