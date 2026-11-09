import torch
import numpy as np
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def add_to_db(database_folders, database_save_path, name_map_save_path, face_rec_model):
    def load_data(database_save_path):
        if database_save_path.exists() and name_map_save_path.exists():
            face_embeddings = torch.load(database_save_path)
            name_map = np.load(name_map_save_path)

            print("Loaded successfully!")

        else:
            print("Not found, creating new")
            return None, None

        return face_embeddings, name_map

    face_embeddings, name_map = load_data(database_save_path)

    if face_embeddings is None or name_map is None:
        face_embeddings = []
        name_map = []
    else:
        face_embeddings = face_embeddings.tolist()
        name_map = name_map.to_list()

    already_in_db = set(name_map)

    with torch.inference_mode():
        for person in database_folders.iterdir():
            if not person.is_dir():
                continue
            person_name = person.name
            print(person_name)

            if person_name in already_in_db:
                print(f"{person_name} already in DB, skipping...")
                continue

            for pic in person.iterdir():
                if pic.is_dir():
                    continue

                pic_pil = Image.open(str(pic)).convert('RGB')
                print(pic)

                face = face_rec_model.get(np.array(pic_pil))
                if face:
                    face_embeddings.append(face[0].normed_embedding)
                    name_map.append(person_name)

                print()

        face_embeddings = torch.tensor(np.array(face_embeddings), dtype=torch.float32).to(device)
        name_map = np.array(name_map)

        torch.save(face_embeddings, database_save_path)
        np.save(name_map_save_path, name_map)

        print()
        print(name_map)
    return face_embeddings, name_map