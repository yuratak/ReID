# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1) # (33, 724) Increasing sorting with index
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32) # Ordering gallery ids based with increasing distmat values (logical : the smaller the distance, more probable match)
    # g_pids : (724, ) -> g_pids[indices] : (33, 724) / q_pids : (33)
    # match each query image with all the gallery image -> matches : ordered list of gallery vehicle image's ids


    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx] # vehicle id
        q_camid = q_camids[q_idx] # camera id

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid) # 6 removed here for VERI dataset
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep] # orig_cmc = gt(k) # len of the vector = gallery size
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        # the following two lines are used to get the first index where the first positive appears
        cmc = orig_cmc.cumsum() # cumsum() : [1, 2, 3] -> [1, 3, 6]
        cmc[cmc > 1] = 1 # : [1, 3, 6] -> [1, 1, 1] -> first matching happened at index 0

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum() # To calculate the denominator for the precision calculation
        # orig_cmc = [1, 1, 0, 1, 1] : gt(k) : binary vector
        tmp_cmc = orig_cmc.cumsum() # [1, 2, 2, 3, 3] # To calculate the numerator for the precision calculation
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)] # Precision calculation [1/1, 2/2, 2/3, 3/4, 3/5]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc # tmp_cmc=P(k) (Precision at cut-off k) & orig_cmc = gt(k)
        AP = tmp_cmc.sum() / num_rel # num_rel=N_gt (Nb of ground truths)
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32) # shape (33 x 50) : (query_img, max_rank)
    all_cmc = all_cmc.sum(0) / num_valid_q # shape (50) : For each ranking % num_valid_q = 33 (nb of valid query images)
    mAP = np.mean(all_AP)

    return all_cmc, mAP

'''# get the top max_rank
def eval_func_VRAI(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

    #assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    return all_cmc

'''