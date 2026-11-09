import time
import av
from time import  perf_counter
from datetime import datetime
from frame_skip_fun import safe_put_to_queue


def capture_video(source, cap_to_yolo_q,
                  capture_stop, cam_id,
                  source_type, output_fps=24,
                  width=None, height=None):
    print("[CAPTURE] Main started")

    # Open video container and stream
    if source_type == "camera":
        container = av.open(source, format='dshow')
    else:
        container = av.open(source)

    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'

    # Determine frame resolution
    if width is None:
        width = stream.width
    if height is None:
        height = stream.height

    print("[CAPTURE] Loop started")
    frame_interval = 1 / output_fps
    frame_count = 1
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                if capture_stop.value:
                    return

                frame_count += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # up to milliseconds
                # timestamp = datetime.now()

                start_time = time.time()
                loop_start = perf_counter()
                if frame.width != width or frame.height != height:
                    frame = frame.reformat(
                        width=width,
                        height=height,
                        format='rgb24'
                    )

                frame_array = frame.to_ndarray(format='rgb24')

                safe_put_to_queue(cap_to_yolo_q, (frame_array, timestamp, cam_id))
                # FPS control (wait if needed)
                elapsed_time = time.time() - start_time
                sleep_time = frame_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if frame_count == output_fps:
                    frame_count = 1

                # print(f"[CAPTURE] Loop time {(perf_counter() - loop_start) * 1000:.2f}ms")

    except Exception as e:
        print(f"[capture_loop] AVError: {e}")

    # Cleanup after capture loop ends
    container.close()
    # print("[CAPTURE] Capture loop stopped")

    print("[CAPTURE] STOPPED..")