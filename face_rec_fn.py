import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def recognize_faces(face_rec_model, frame,
                    face_db, name_map,
                    conf=0.6, device=device):
    faces = face_rec_model.get(frame)
    bboxes = []
    embeddings = []

    if not faces:
        return (np.empty((0, 5)).astype(np.int32),
                np.empty(0).astype(np.int32),
                np.empty(0).astype(np.int32))
    for face in faces:
        embeddings.append(face.normed_embedding)
        bboxes.append(face.bbox)

    bboxes = np.array(bboxes).astype(np.float32)
    embeddings_tensor = torch.from_numpy(
        np.array(
            embeddings).astype(np.float32)).to(device)

    similarities = torch.mm(face_db, embeddings_tensor.t())

    max_sims, indices = similarities.max(dim=0)

    conf_indices = indices[max_sims > conf].cpu().numpy()
    conf_scores = max_sims[max_sims > conf].cpu().numpy()

    # print("conf ind", conf_indices)
    # print("sim of conf", similarities[conf_indices].argmax(dim=1).cpu().numpy())

    # More than 1 face in DB has a greater than threshold match
    # with multiple faces (bboxes)
    # Filter the one with higest score
    if len(np.unique(conf_indices)) != len(conf_indices):
        new_vals = dict()
        valid_indices = set()
        for i, (ind, sim) in enumerate(zip(conf_indices, max_sims.cpu().numpy())):
            if ind not in new_vals:
                new_vals[ind] = (sim, i)
                valid_indices.add(i)

            else:
                a_s, a_i = new_vals[ind]
                if a_s < sim:
                    new_vals[ind] = (sim, i)
                    valid_indices.remove(a_i)
                    valid_indices.add(i)

        valid_mask = np.array(list(valid_indices), dtype=np.int32)

        conf_indices = conf_indices[valid_mask]
        conf_scores = max_sims.cpu().numpy()[valid_mask]

    conf_bboxes = bboxes[similarities[conf_indices].argmax(dim=1).cpu().numpy()]
    # print("conf_bboxes", conf_bboxes)

    # print(conf_bboxes)

    fbbox_track_ids = np.hstack(
        (conf_bboxes,  # conf_bboxes
         -1 * np.ones(
             (conf_bboxes.shape[0], 1)
         )  # -1 default tracking bbox unknows
         ), dtype=np.float32
    )

    # print(fbbox_track_ids)

    return (fbbox_track_ids,
            # name_map[confident indices]
            name_map[conf_indices],
            # confidence values
            conf_scores
            )