import numpy as np


def track(tracker, detections_numpy, frame_rgb, vehicle_classes=(2, 3, 5, 7)):
    '''
    returns [x1, y1, x2, y2, tracking_id, conf, cls, det_ind]

    x1, y1, x2, y2 From detections_numpy
                   i.e. deteted bboxes from YOLO (or any detector)
    tracking_id    The track ID (unique ID assigned by the tracker to an object over time)
    conf	       The confidence score of the detection that initialized or updated the track
                   passed via detections_numpy
    cls            The class label (e.g., 0 for person, 1 for car, etc., from YOLO)
                   passed via detections_numpy
    det_ind        The detection index from the original detections array
                   i.e. detections_numpy
    np array (N x 8)
    '''
    if detections_numpy.size == 0:
        return (
            np.empty((0, 8), dtype=np.float32),
            np.empty((0, 8), dtype=np.float32))

    tracking_data = tracker.update(detections_numpy, frame_rgb).astype(np.float32)
    # print(tracking_data)
    # print(tracking_data.shape)

    if tracking_data.size == 0:
        people_tracks = np.empty((0, 8), dtype=np.float32)
        vehicle_tracks = np.empty((0, 8), dtype=np.float32)

    else:
        class_ids = tracking_data[:, 6]
        people_mask = class_ids == 0
        vehicle_mask = np.isin(class_ids, np.array(vehicle_classes))
        people_tracks = tracking_data[people_mask]
        vehicle_tracks = tracking_data[vehicle_mask]

    return people_tracks, vehicle_tracks