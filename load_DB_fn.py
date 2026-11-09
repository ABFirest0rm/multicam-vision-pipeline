import torch
import numpy as np


def load_data(face_db_save_path, name_map_save_path):
    if face_db_save_path.exists() and name_map_save_path.exists():
        face_db = torch.load(face_db_save_path)
        name_map = np.load(name_map_save_path)
        print("Loaded successfully!")

    else:
        print("Not found, creating new")
        return None, None

    return face_db, name_map