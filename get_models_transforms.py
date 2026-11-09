# import torch
# from boxmot import BotSort
from pathlib import Path
from torchvision import transforms
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_tracker(name, device):
    if name == 'botsort':
        print(f"USING TRACKER {name}")
        from boxmot import BotSort

        botsort_tracker = BotSort(
            reid_weights=Path('../osnet_x0_25_msmt17.pt'),  # Path to ReID model
            device=device,  # Use CPU for inference
            half=False,
            with_reid=True
        )

        return botsort_tracker

    else:
        print(f"USING TRACKER {name}")
        from boxmot import BoostTrack
        boosttrack_tracker = BoostTrack(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),  # Path to ReID model
            device=device,  # Use CPU for inference
            half=False,
            with_reid=True,
            use_ecc=False
        )

        return boosttrack_tracker

def get_yolo(device):
    yolo = YOLO('yolo12n', verbose=False).to(device).eval()

    yolo_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    return yolo, yolo_transform

def get_face_rec_model():
    model_pack_name = 'buffalo_l'
    face_rec_model = FaceAnalysis(name=model_pack_name, allowed_modules=['detection', 'recognition'],
                                  providers=['CUDAExecutionProvider'])
    face_rec_model.prepare(ctx_id=0, det_size=(640, 640))

    return face_rec_model


