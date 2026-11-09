from multiprocessing import Process, Queue, Value, Lock
from time import sleep
from capture import capture_video
from face_rec_match_bbox_worker import face_rec_match_bbox_worker
from yolo_worker import yolo_worker
from tracker_worker import tracker_worker
from display_worker import display_worker
# from load_DB_fn import load_data
from pathlib import Path
from ctypes import c_bool


if __name__ == "__main__":

    cap_to_yolo_q = Queue(maxsize=10)
    yolo_to_tracker_q = Queue(maxsize=10)
    tracker_to_display_q = Queue(maxsize=10)

    display_to_face_rec_queue = Queue(maxsize=10)
    face_rec_to_display_q = Queue(maxsize=10)


    capture_stop = Value(c_bool, False)
    yolo_stop = Value(c_bool, False)
    display_stop = Value(c_bool, False)
    track_stop = Value(c_bool, False)
    face_rec_stop = Value(c_bool, False)
    track_reset = Value(c_bool, False)
    # clear_queues_flag = Value(c_bool, False)
    # face_queue_clear = Value(c_bool, False)
    #
    # tracker_to_display_q_lock = Lock()
    # display_to_face_req_q_lock = Lock()

    all_stop_flags = (capture_stop, yolo_stop, display_stop, track_stop, face_rec_stop, track_reset)

    face_db_save_path = Path(r"E:\AIML\cyberdome\avengers_database.pt")
    name_map_save_path = Path(r"E:\AIML\cyberdome\name_map.npy")

    # face_db, name_map = load_data(face_db_save_path, name_map_save_path)

    source = "video=ACER HD User Facing"
    source_type = "camera"

    source = r"E:\AIML\cyberdome\vid tests\19172.mp4"
    source_type = "vid"

    capture_process = Process(
        target=capture_video,
        kwargs={
            'source': source,
            'cap_to_yolo_q': cap_to_yolo_q,
            'capture_stop': capture_stop,
            'cam_id': "ACER HD User Facing",
            'source_type': source_type,
            'output_fps': 24,
            'width': 1280,
            'height': 720
        }
    )

    yolo_process = Process(
        target=yolo_worker,
        kwargs={
            'cap_to_yolo_q': cap_to_yolo_q,
            'yolo_to_tracker_q': yolo_to_tracker_q,
            'yolo_stop': yolo_stop,
            'width': 1280,
            'height': 720
        }
    )

    tracker_process = Process(
        target=tracker_worker,
        kwargs={
            'yolo_to_tracker_q': yolo_to_tracker_q,
            'tracker_to_display_q': tracker_to_display_q,
            'track_stop': track_stop,
            'track_reset': track_reset,
            # 'tracker_to_display_q_lock': tracker_to_display_q_lock,
            # 'clear_queues_flag': clear_queues_flag,
            # 'face_queue_clear': face_queue_clear
        }
    )

    face_rec_match_bbox_process = Process(
        target=face_rec_match_bbox_worker,
        kwargs={
            'display_to_face_rec_queue': display_to_face_rec_queue,
            'face_rec_to_display_q': face_rec_to_display_q,
            'face_db_save_path': face_db_save_path,
            'name_map_save_path': name_map_save_path,
            'face_rec_stop': face_rec_stop,
            # 'face_queue_clear': face_queue_clear,
            # 'display_to_face_req_q_lock': display_to_face_req_q_lock,
            'threshold': 0.75
        }
    )

    display_process = Process(
        target=display_worker,
        kwargs={
            'tracker_to_display_q': tracker_to_display_q,
            'display_to_face_rec_queue': display_to_face_rec_queue,
            'face_rec_to_display_q': face_rec_to_display_q,
            'display_stop': display_stop,
            'track_reset': track_reset,
            # 'tracker_to_display_q_lock': tracker_to_display_q_lock,
            # 'display_to_face_req_q_lock': display_to_face_req_q_lock,
            # 'clear_queues_flag': clear_queues_flag,
            # 'face_queue_clear': face_queue_clear,
            'output_fps': 24
        }
    )

    # Start all processes
    capture_process.start()
    yolo_process.start()
    tracker_process.start()
    face_rec_match_bbox_process.start()
    display_process.start()


    # while not display_stop.value:
    #     sleep(2)
    #
    #     flags.capture_stop.value = True
    #     flags.yolo_stop.value = True
    #     flags.track_stop = True
    #     flags.display_stop.value = True

    print("[MAIN] Done")