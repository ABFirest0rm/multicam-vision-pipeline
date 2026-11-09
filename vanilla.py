from insightface.app import FaceAnalysis
import cv2
import numpy as np
from time import perf_counter
from pathlib import Path

model_pack_name = 'buffalo_l'
face_rec_model = FaceAnalysis(name=model_pack_name, allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider'])
face_rec_model.prepare(ctx_id=0, det_size=(640, 640))

import cv2
import numpy as np


my_pics = Path(r"E:\AIML\cyberdome\database_pics\Abey")
name_map = dict()
face_embeddings = []
for i, pic_path in enumerate(my_pics.iterdir()):
    pic_rgb = cv2.cvtColor(cv2.imread(str(pic_path)), cv2.COLOR_BGR2RGB)
    face = face_rec_model.get(np.array(pic_rgb))[0]
    face_embeddings.append(face.normed_embedding)
    name_map[i] = 'Abey'
face_embeddings = np.array(face_embeddings)


cap = cv2.VideoCapture(0)
dt = 0
lt = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame or end of video")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dt = perf_counter()
    faces = face_rec_model.get(frame_rgb)
    print(f"Det Time: {(perf_counter() - dt) * 1000:.2f}")

    if len(faces) == 0:
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    embs = []
    bboxes = []
    lt = perf_counter()
    for face in faces:
        bboxes.append(face.bbox)
        embs.append(face.normed_embedding)

    embs = np.array(embs)

    similarities = np.matmul(face_embeddings, embs.T)

    max_sim_scores = np.max(similarities, axis=0)
    best_match_indices = np.argmax(similarities, axis=0)
    print(f"Lookup Time: {(perf_counter() - lt) * 1000:.2f}")

    for bbox, sim_score, idx in zip(bboxes, max_sim_scores, best_match_indices):
        if sim_score > 0.5:
            name = name_map.get(idx)
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      color, 2)

        cv2.putText(frame, f"{name} {sim_score:.2f}",
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()