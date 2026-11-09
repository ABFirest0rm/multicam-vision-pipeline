import numpy as np
from collections import defaultdict
from datetime import datetime


def match_fbbox_to_tbbox(fbbox_track_ids, tbboxes, names, confs):
    if tbboxes.size == 0 or fbbox_track_ids.size == 0:
        return fbbox_track_ids.astype(np.int32), names, confs, dict()
    # Filter all tbboxes that encase the fbbox

    mask = (
               # fx1 > tx1 - right
                   fbbox_track_ids[:, 0].reshape(-1, 1) > tbboxes[:, 0].reshape(1, -1)
           ) & (
               # fx2 < tx2 - left
                   fbbox_track_ids[:, 2].reshape(-1, 1) < tbboxes[:, 2].reshape(1, -1)
           )

    '''
    m x n
    m - no. of face bboxes
    n - no of tracking bboxes


    # Row - face bbox
    # Column - True where tracking bbox of at that index
    encases the fbbox corresponding to the row
    [
    [False,  True,  True],
    [False,  True,  True]
    ]

    '''
    fbbox_tbboxes = defaultdict(list)

    # np.nonzero(mask) = array([0, 1, 1], array([2, 1, 2]
    for fbbox_ind, tbbox_ind in (zip(*np.nonzero(mask))):
        fbbox_tbboxes[fbbox_ind].append(tbbox_ind)

    assigned = dict()
    tid_name_map = dict()
    curr_time = datetime.now()
    for fbbox_ind in fbbox_tbboxes.keys():
        best_track_id = -1
        best_y_diff = float("inf")

        for tbbox_ind in fbbox_tbboxes[fbbox_ind]:
            y1_diff = fbbox_track_ids[fbbox_ind, 1] - tbboxes[tbbox_ind, 1]
            y2_diff = fbbox_track_ids[fbbox_ind, 3] - tbboxes[tbbox_ind, 3]
            if y1_diff > 0 and y2_diff < 0 and y1_diff < best_y_diff:
                best_track_id = tbboxes[tbbox_ind, 4]
                best_y_diff = y1_diff

        if best_track_id == -1:
            continue

        elif best_track_id in assigned:
            a_fbbox_ind, a_best_y_diff = assigned[best_track_id]
            if best_y_diff < a_best_y_diff:
                fbbox_track_ids[fbbox_ind, 4] = best_track_id
                fbbox_track_ids[a_fbbox_ind, 4] = -1
                assigned[int(best_track_id)] = (fbbox_ind, best_y_diff)
                tid_name_map[int(best_track_id)] = {'name': names[fbbox_ind],
                                                    'conf': confs[fbbox_ind],
                                                    'last_seen': curr_time}
        else:
            fbbox_track_ids[fbbox_ind, 4] = best_track_id
            assigned[int(best_track_id)] = (fbbox_ind, best_y_diff)
            tid_name_map[int(best_track_id)] = {'name': names[fbbox_ind],
                                                'conf': confs[fbbox_ind],
                                                'last_seen': curr_time}

    fbboxes_mask = fbbox_track_ids[:, 4] == -1
    fbboxes = fbbox_track_ids[fbboxes_mask]
    fbbox_names = names[fbboxes_mask]
    fbbox_confs = confs[fbboxes_mask]

    return fbboxes.astype(np.int32), fbbox_names, fbbox_confs, tid_name_map