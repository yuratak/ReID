# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import json
import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func#, eval_func_VRAI
from .re_ranking import re_ranking


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        # Feature extracted for the whole query dataset and the gallery dataset (all the images are treated in the same time)
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        
        # query
        qf = feats[:self.num_query] # query features
        q_pids = np.asarray(self.pids[:self.num_query]) # pids of same length as qf (pid for each of the image)
        q_camids = np.asarray(self.camids[:self.num_query]) # camids of same length as qf (camid for each of the image)
        # gallery
        gf = feats[self.num_query:] # gallery features
        g_pids = np.asarray(self.pids[self.num_query:]) # //
        g_camids = np.asarray(self.camids[self.num_query:]) # //
        m, n = qf.shape[0], gf.shape[0] # distance matrix size (betweem each query and gallery image, distance calculated)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t() # all 2 matrice of shape m (33 query) x n (724 gallery)
        # COSINE DISTANCE CALCULATION
        distmat.addmm_(1, -2, qf, gf.t()) # Beta(1) * distmat + Alpha(-2) * (qf @ gf.t()) with qf (33 x 2048) and gf.t() (2048 x 724) -> (33 x 724)
        # If qf[i] and gf[j] is the same feature -> L2 norm^2 == 1 -> distmat[i, j] == 0
        # distmat > 0 Why ?
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


# YT added
class R1_mAP_VRAI(Metric):
    def __init__(self, num_query, max_rank=1000, feat_norm='yes'):
        super(R1_mAP_VRAI, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()

        num_q, num_g = distmat.shape
        #if num_g < max_rank:
        #    max_rank = num_g
        #    print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        #print(g_pids[indices])
        #gallery = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
        #print(g_pids[indices].shape)
        g_pids_ordered = g_pids[indices]
        print(g_pids_ordered.shape)

        reid_result = []
        for q_idx in range(num_q):
            # get query pid and camid
            #q_pid = q_pids[q_idx]
            reid_result.append({"query_id": q_idx, "ans_ids": list(g_pids_ordered[q_idx][:1000])})
            #print(list(g_pids_ordered[q_idx][:1000]))
        #cmc = eval_func_VRAI(distmat, q_pids, g_pids, q_camids, g_camids)

        results = {'reid_result': reid_result}
            
        with open("./submission.json", "w") as f:
            json.dump(results, f) # .decode('unicode-escape')
        return #cmc


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP