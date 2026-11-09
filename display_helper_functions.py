from time import sleep, perf_counter
from queue import Full, Empty
from datetime import datetime, timedelta
import logging
import _queue
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

def stop_all(all_stop_flags):
    for flag in all_stop_flags:
        flag.value = False



def display_mem_updater(face_rec_to_display_q, track_reset,
                        display_mem, display_mem_lock,
                        reset_interval, known_ids, flags):

    last_reset = datetime.now()
    timeout = timedelta(seconds=3)
    update = False
    while True:
        curr_time = datetime.now()
        if (curr_time - last_reset) > reset_interval:
            print(f"More than {reset_interval} seconds since last reset")
            # log.info(f"More than {reset_interval} seconds since last reset")

            # print(f"DISPLAY MEM{display_mem}")

            print(f"DISPLAY MEM LOCKED HELPER")
            ids_to_del = []
            for tid in known_ids:
                last_seen = display_mem[tid]['last_seen']
                elapsed = curr_time - last_seen

#                     print(f"[{tid}] Last seen: {last_seen.strftime('%H:%M:%S.%f')[:-3]}")
#                     print(f"[{tid}] Elapsed time: {elapsed.total_seconds():.2f} seconds")
#
#                     log.info(f"[{tid}] Last seen: {last_seen.strftime('%H:%M:%S.%f')[:-3]}")
#                     log.info(f"[{tid}] Elapsed time: {elapsed.total_seconds():.2f} seconds")

                if elapsed > timeout:
                    update = True
                    ids_to_del.append(tid)
            print(f"HELPER Ids to del {ids_to_del}")
            if update:
                print(f"HELPER DELETING IDS")
                with display_mem_lock:
                    print(f"HELPER display mem{display_mem}")
                    print(f"HELPER kid{known_ids}")
                    for tid in known_ids:
                        del display_mem[tid]
                    known_ids.clear()
                    known_ids.extend(display_mem.keys())

                    print(f"HELPER display mem after{display_mem}")
                    print(f"HELPER kid after{known_ids}")
                    flags[1] = 1
                    update = False
            # log.info(f"KNOWN IDS INSIDE: {known_ids}")

            if len(known_ids) == 0:
                track_reset.value = True
            last_reset = curr_time

        if not face_rec_to_display_q.empty():
            try:
                frame_get = perf_counter()
                tid_name_map, curr_id = face_rec_to_display_q.get(timeout=0.5)
                if curr_id != flags[0] and curr_id == flags[2]:
                    continue

                if len(tid_name_map) != 0:
                    # update = False
                    temp_dict = dict()
                    for tid in tid_name_map:
                        if tid not in display_mem or display_mem[tid]['conf'] <= tid_name_map[tid]['conf']:
                            # update = True
                            temp_dict[tid] = {'name': tid_name_map[tid]['name'],
                                              'conf': tid_name_map[tid]['conf'],
                                              'last_seen': tid_name_map[tid]['last_seen']}

                    if len(temp_dict) > 0:
                        # print("HELPER NEW DATA RECEIVED UPDATING")
                        # print("UPDATING DICT")
                        # print("BEFORE")
                        # print(display_mem)
                        pf = perf_counter()
                        with display_mem_lock:
#                             print("HELPER UPDATING DICT")
#                             print(f"HELPER before 2 {display_mem}")
#                             print(f"HELPER kid 2 {known_ids}")
                            # print("UPDATING DICT")
                            display_mem.update(temp_dict)
                            # known_ids = np.array(tuple(display_mem.keys()), dtype=np.int32)
                            known_ids.clear()
                            known_ids.extend(display_mem.keys())
#                             print(f"HELPER after 2{display_mem}")
#                             print(f"HELPER kid after 2{known_ids}")
                            flags[1] = 1
#                         print(f"DISPLAY HELPER LOCKING MEM FOR {(perf_counter() - pf)*1000:.2f}")
#                         print("AFTER")
#                         print(display_mem)
                else:
                    pass
                    # print("skipping dict update")
                    # print(known_ids)

            except _queue.Empty:
                pass
                print("[DISPLAY] Face Frame queue is unexpectedly empty.")

        else:
            sleep(1)
