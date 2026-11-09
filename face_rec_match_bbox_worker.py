from face_rec_fn import recognize_faces
from match_fbbox_to_tbbox_fn import match_fbbox_to_tbbox
from get_models_transforms import get_face_rec_model
from time import perf_counter, sleep
import _queue
from load_DB_fn import load_data
from frame_skip_fun import safe_put_to_queue

def face_rec_match_bbox_worker(display_to_face_rec_queue,
                               face_rec_to_display_q, face_db_save_path,
                               name_map_save_path, face_rec_stop,
                               threshold=0.6):

    face_rec_model = get_face_rec_model()
    face_db, name_map = load_data(face_db_save_path, name_map_save_path)

    while True:
        if face_rec_stop.value:
            # print("[TRACKER] Stop Received")
            break
        try:
            frame_rgb, tbboxes, timestamp, curr_id = display_to_face_rec_queue.get(timeout=1)  # Wait up to 1 second for a frame
#             print(f"[FACE_REC] Capture frame get {(perf_counter() - frame_get) * 1000:.2f}ms")

        except _queue.Empty:
#             print("[TRACKER] Frame queue is empty.")
            if face_rec_stop.value:
#                 print("[FACE_REC] Stop Received")
                break
            sleep(0.01)
            continue

        fbbox_track_ids, names, confs = recognize_faces(face_rec_model=face_rec_model,
                                                        frame=frame_rgb,
                                                        face_db=face_db, name_map=name_map,
                                                        conf=threshold)

        if fbbox_track_ids.size == 0:
            safe_put_to_queue(face_rec_to_display_q,
                              (dict(), curr_id))

        else:
            # print(f"[FACE RECOGNIZED] {names}")
            fbboxes, fbbox_names, fbbox_confs, tid_name_map = match_fbbox_to_tbbox(fbbox_track_ids,
                                                                                   tbboxes, names,
                                                                                   confs)
            # print(f"[FACE TRACKED] {tid_name_map}")
            safe_put_to_queue(face_rec_to_display_q, (tid_name_map, curr_id))

    print("[FACE_REC] STOPPED")