from main_single_cam_function import single_cam
# from threading import Thread
from multiprocessing import Process, Queue





















if __name__ == '__main__':
    sources = [
        ("video=ACER HD User Facing", "camera", 'webcam'),
        #        (r"E:\AIML\cyberdome\vid tests\19172.mp4", 'vid', 'vid1'),
        #        (r"E:\AIML\cyberdome\vid tests\1917 run.mp4", 'vid', 'vid2')

        (r"C:\Users\ajita\Desktop\demo\project_presentation occlusion.mp4", 'vid', 'vid2')
    ]






























    # queues = [Queue(maxsize=10) for _ in range(len(sources))]

    # processes = [Process(target=single_cam, kwargs={'source': source, 'source_type': source_type, 'cam_id': cam_id, 'op_queue': queue})
    #                   for queue, (source, source_type, cam_id) in zip(queues, sources)]

    processes = [Process(target=single_cam,
                         kwargs={'source': source, 'source_type': source_type, 'cam_id': cam_id,})
                 for source, source_type, cam_id in sources]

    for process in processes:
        process.start()

