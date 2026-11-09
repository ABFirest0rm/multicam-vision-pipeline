import torch
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# import json
# with open(r"E:\AIML\cyberdome\yolo_class_names.json", "r") as file:
#     yolo_class_names = json.load(file)

def detect_yolo(model, transform,
                x_ratio, y_ratio,  # # original / resized
                frame_rgb, conf_thresh,
                device):
    '''
    returns people_data, vehicle_data
    tuple of tensors
    people data [[x1, y1, x2, y2, conf, class_ID],...]
    vehicle data [[x1, y1, x2, y2, conf, class_ID],...]

    (tensor(N x 6), tensor(M x 6))
    '''
    detections = model(transform(frame_rgb).to(device).unsqueeze(0), verbose=False)[0]

    # dets = []

    bboxes = detections.boxes.xyxy  # Bounding boxes [N, 4]
    confs = detections.boxes.conf  # Confidence scores [N]
    class_ids = detections.boxes.cls.int()

    conf_mask = confs > conf_thresh

    def scale_bboxes(bboxes):
        # if bboxes.numel() > 0:
        bboxes[:, [0, 2]] = torch.clamp((bboxes[:, [0, 2]] * x_ratio).round(), min=0, max=frame_rgb.shape[1] - 1)
        bboxes[:, [1, 3]] = torch.clamp((bboxes[:, [1, 3]] * y_ratio).round(), min=0, max=frame_rgb.shape[0] - 1)  # y1

        return bboxes

    detections = torch.cat(
        [scale_bboxes(bboxes[conf_mask]),
         confs[conf_mask].unsqueeze(1), class_ids[conf_mask].unsqueeze(1)],
        dim=1
    ) if bboxes.numel() > 0 else torch.empty((0, 6), device=device)

    return detections.cpu().numpy()