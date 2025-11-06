import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import os
import torch
from tqdm import tqdm

import util

from VisualComponent import VisualComponent
from TemporalComponent import TemporalComponent

class VReID:
    def __init__(self):
        self.visual = VisualComponent()
        self.visual.load_test_model()
        self.visual.process_reid_data()

        self.feats = torch.nn.functional.normalize(self.visual.feats, dim=1, p=2)
        self.feats_train = torch.nn.functional.normalize(self.visual.feats_train, dim=1, p=2)
        self.val_pid, self.val_camid = self.visual.val_pid, self.visual.val_camid # ids ordered as features
        self.train_qg_pid, self.train_qg_camid = self.visual.train_qg_pid, self.visual.train_qg_camid # ids ordered as features
        self.num_query, self.num_query_train = self.visual.num_query, self.visual.num_query_train

        self.qf, self.q_pids, self.q_camids = self.feats[:self.num_query], np.asarray(self.val_pid[:self.num_query]), np.asarray(self.val_camid[:self.num_query])
        self.gf, self.g_pids, self.g_camids = self.feats[self.num_query:], np.asarray(self.val_pid[self.num_query:]), np.asarray(self.val_camid[self.num_query:])

        self.qf_train, self.q_pids_train, self.q_camids_train = self.feats_train[:self.num_query_train], np.asarray(self.train_qg_pid[:self.num_query_train]), np.asarray(self.train_qg_camid[:self.num_query_train])
        self.gf_train, self.g_pids_train, self.g_camids_train = self.feats_train[self.num_query_train:], np.asarray(self.train_qg_pid[self.num_query_train:]), np.asarray(self.train_qg_camid[self.num_query_train:])

        self.temporal = TemporalComponent()

        self.distmat_visual = None
        self.distmat_temporal_dict = {}

        self.visual_dist_mat_fp = os.path.join(self.visual.checkpoint_dir, "visual_distmat.txt")
        self.alpha_cmc_map_fp_dict = {k: os.path.join(self.temporal.checkpoint_dir, f"alpha_cmc_map_{k}.txt") for k in self.temporal.temporal_models.keys()}

        return
    

    def get_distmat_visual(self):
        # If ReUsed ReID Object, do not calculate the distance matrix again
        if self.distmat_visual is not None:
            pass
        # If the distance matrix is saved
        elif os.path.exists(self.visual_dist_mat_fp):
            self.distmat_visual = np.loadtxt(self.visual_dist_mat_fp)
        # Calculate the distance matrix
        else:
            m, n = self.qf.shape[0], self.gf.shape[0] # distance matrix size (betweem each query and gallery image, distance calculated)
            distmat = torch.pow(self.qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(self.gf, 2).sum(dim=1, keepdim=True).expand(n, m).t() # all 2 matrice of shape m (33 query) x n (724 gallery)
            
            # COSINE DISTANCE CALCULATION
            distmat = distmat.addmm(self.qf, self.gf.t(), beta=1, alpha=-2) # Beta(=1) * distmat(=2) + Alpha(=-2) * (qf @ gf.t()) with qf (33 x 2048) and gf.t() (2048 x 724) -> (33 x 724)
            # If qf[i] and gf[j] is the same feature -> L2 norm^2 == 1 -> distmat[i, j] == 0
            self.distmat_visual = distmat.cpu().detach().numpy() / 2
            # Save the distance matrix
            np.savetxt(self.visual_dist_mat_fp, self.distmat_visual)

        return self.distmat_visual
    
    

    def get_distmat_temporal(self, temporal_model_name, sigma=0, window=0):
        if (temporal_model_name, sigma, window) in self.distmat_temporal_dict.keys():
            return self.distmat_temporal_dict[(temporal_model_name, sigma, window)]

        # Used by default, if window it not used
        # Calculate the distance between the observed travel time & the predicted travel time, between a pair of gallery and query vehicles
        def calculate_tanh_dist(ls, model_loss):
            ls = np.absolute(ls)
            ls = np.array([_ if _ < 2*model_loss else model_loss*2 for _ in ls])
            ls /= model_loss
            ls = np.tanh(ls)
            return ls
        
        # Used if window is used
        # If the travel time is less than window, then the distance is 0, otherwise, the distance is 2
        def calculate_binary_dist(ls, window):
            ls = np.absolute(ls)
            ls = np.array([0 if _ < window else 2 for _ in ls])
            return ls

        # test temporal data
        # ['id', 'traj_src', 'vel_src', 'category', 'vld_nb_src', 'vld_nb_dst', 'traj_btwn_vlds', 'avg_travel_time_btwn_vlds', 'y', 'turning_ratio', 'traj_btwn_vlds_nb', 'start_time', 'end_time']
        # Extract what we know : query vehicle's end time & gallery vehicle's start time
        query_end_time = self.temporal.get_df_w_gt_vld_pair().end_time.to_dict() # remove the duplicated index and keep the last occurrence of the duplicate rows.
        gallery_start_time = self.temporal.get_df_w_gt_vld_pair().start_time.to_dict() # remove the duplicated index and keep the last occurrence of the duplicate rows.

        # Construct the temporal distance (between the observed query vehicle and the ground truth / predicted gallery vehicles)
        m, n = self.qf.shape[0], self.gf.shape[0]
        distmat_temporal = np.ones((m, n))
        
        # Ground Truth Travel Time
        if temporal_model_name is None:
            # If sigma, add the controlled noise to the ground truth gallery travel time, oterwise, use the ground truth gallery travel time
            gallery_pred_time = self.temporal.get_df_w_gt_vld_pair().y.to_dict()
            if sigma:
                gallery_pred_time = {k: v + np.random.normal(scale=sigma) for k, v in gallery_pred_time.items()}

            # If model loss = 1.0 they consider time difference of 1.0 to 2.0 to be distributed between 0 and 1
            # If model loss = sigma they consider time difference of sigma to 2*sigma to be distributed between 0 and 1
            model_loss = float(sigma) if sigma else 1.0
            
            for q_i, q_pid in enumerate(self.q_pids):
                for g_i, g_pid in enumerate(self.g_pids):
                    distmat_temporal[q_i, g_i] = query_end_time[q_pid] - (gallery_start_time[g_pid] + gallery_pred_time[g_pid])
        
        elif temporal_model_name is "shockwave":
            gallery_pred_time = self.temporal.test_shockwave()

            for q_i, q_pid in enumerate(self.q_pids):
                query_end_time_i = query_end_time[q_pid]
                for g_i, g_pid in enumerate(self.g_pids):
                    gallery_start_time_i = gallery_start_time[g_pid]
                    gallery_pred_time_i = gallery_pred_time.loc[[g_pid]].set_index("traj_btwn_vlds_nb")

                    # For a gallery vehicle, there are multiple possible travel times depending on the exit VLD
                    # calculate the travel time corresponding to the gallery's entry VLD and the observed exit VLD pair
                    pair_ = (int(self.g_camids[g_i]), int(self.q_camids[q_i]))
                    gallery_pred_time_i_ = gallery_pred_time_i.preds[pair_] #if pair_ in gallery_pred_time_i.index else model_loss*2
                    
                    distmat_temporal[q_i, g_i] = query_end_time_i - (gallery_start_time_i + gallery_pred_time_i_)

        # Predicted Travel Time
        else:
            model_loss = self.temporal.temporal_models_loss[temporal_model_name]
            gallery_pred_time = self.temporal.test(temporal_model_name)

            for q_i, q_pid in enumerate(self.q_pids):
                query_end_time_i = query_end_time[q_pid]
                for g_i, g_pid in enumerate(self.g_pids):
                    gallery_start_time_i = gallery_start_time[g_pid]
                    gallery_pred_time_i = gallery_pred_time.loc[[g_pid]].set_index("traj_btwn_vlds_nb")

                    # For a gallery vehicle, there are multiple possible travel times depending on the exit VLD
                    # calculate the travel time corresponding to the gallery's entry VLD and the observed exit VLD pair
                    pair_ = (int(self.g_camids[g_i]), int(self.q_camids[q_i]))
                    gallery_pred_time_i_ = gallery_pred_time_i.preds[pair_] if pair_ in gallery_pred_time_i.index else model_loss*2
                    
                    distmat_temporal[q_i, g_i] = query_end_time_i - (gallery_start_time_i + gallery_pred_time_i_)

        # From the raw temporal distance, calculate the distance between the observed travel time & the predicted travel time, between a pair of gallery and query vehicles            
        for q_i in range(m):
            if window:
                distmat_temporal[q_i, :] = calculate_binary_dist(distmat_temporal[q_i, :], window)
            else:
                distmat_temporal[q_i, :] = calculate_tanh_dist(distmat_temporal[q_i, :], model_loss)
        
        self.distmat_temporal_dict[(temporal_model_name, sigma, window)] = distmat_temporal

        return distmat_temporal
    

    #def evaluate(self, distmat_temporal, temp_alpha, top_reid=0, hierarchical_flag=False, logging=False):
    def evaluate(self, temp_alpha, temporal_model_name, dynamic, gt_vlds, temporal_window=0, temporal_sigma=0, logging=False, return_assignment=False):
        distmat_visual = self.get_distmat_visual()
        distmat_temporal = self.get_distmat_temporal(temporal_model_name, window=temporal_window, sigma=temporal_sigma)
        distmat = (1-temp_alpha) * distmat_visual + temp_alpha * distmat_temporal # distmat shape : (m, n) = self.qf.shape[0], self.gf.shape[0]
        
        if dynamic:
            # mask gallery vehicles if they crossed the start VLD after that it has crossed the end VLD
            arrival_time = np.array([self.temporal.get_df_w_gt_vld_pair().end_time.to_dict()[k] for k in self.q_pids])
            departure_time = np.array([self.temporal.get_df_w_gt_vld_pair().start_time.to_dict()[k] for k in self.g_pids])
            arrival_time = np.tile(arrival_time, (len(arrival_time), 1)).T # (207, 207) : horizontally tiled
            departure_time = np.tile(departure_time, (len(departure_time), 1)) # (207, 207) : vertically tiled
            distmat = np.where(arrival_time < departure_time, 1.0, distmat)
        
        if gt_vlds:
            # Same Src VLD
            query_vld = np.array([self.temporal.get_df_w_gt_vld_pair().vld_nb_src.to_dict()[k] for k in self.q_pids])
            gallery_vld = np.array([self.temporal.get_df_w_gt_vld_pair().vld_nb_src.to_dict()[k] for k in self.g_pids])
            query_vld = np.tile(query_vld, (len(query_vld), 1)).T
            gallery_vld = np.tile(gallery_vld, (len(gallery_vld), 1))
            distmat = np.where(query_vld - gallery_vld != 0, 1.0, distmat)

            # Same Dst VLD
            query_vld = np.array([self.temporal.get_df_w_gt_vld_pair().vld_nb_dst.to_dict()[k] for k in self.q_pids])
            gallery_vld = np.array([self.temporal.get_df_w_gt_vld_pair().vld_nb_dst.to_dict()[k] for k in self.g_pids])
            query_vld = np.tile(query_vld, (len(query_vld), 1)).T
            gallery_vld = np.tile(gallery_vld, (len(gallery_vld), 1))
            distmat = np.where(query_vld - gallery_vld != 0, 1.0, distmat)

        cmc, mAP = self.eval_func(distmat, self.q_pids, self.g_pids, self.q_camids, self.g_camids)
        
        if logging:
            print("CMC : %.4f, %.4f, %.4f" % (cmc[0], cmc[5], cmc[10]))
            print("mAP : %.4f" % mAP)

        if return_assignment:
            indices = np.argsort(distmat, axis=1)

            return cmc, mAP, self.g_pids[indices]

        return cmc, mAP
    

    def eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=200, logging=False):
        """Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank and logging:
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

            # remove gallery samples that have the same pid and camid with query (same vehicles captured by the same camera as the query vehicle)
            order = indices[q_idx]
            # camid = VLD nb -> remove the gallery vehicles observed at the same VLD as the query vehicle
            remove = ((g_pids[order] == q_pid) & (g_camids[order] == q_camid)) #or (g_camids[order] == q_camid) # YT : Added here or (g_camids[order] == q_camid)
            keep = np.invert(remove)
            

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep] # orig_cmc = gt(k) # len of the vector = gallery size
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue
            # the following two lines are used to get the first index where the first positive appears
            cmc = orig_cmc.cumsum() # cumsum() : [1, 2, 3] -> [1, 3, 6]
            #print(cmc)
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
    
    
    # lw : line width
    # ls : line style
    def alpha_evaluate(self, temporal_model_ls, dynamic, gt_vlds, metrics=["mAP", "CMC-1", "CMC-5", "CMC-10"], colors={"mAP": "royalblue", "CMC-1": "forestgreen", "CMC-5": "orange", "CMC-10": "red"}, title=None, logging=False):
        alphas = np.arange(0, 1.01, 0.01)
        _, ax = plt.subplots(figsize=(6.4*2, 4.8))
        legend_elements = self.get_legend_elements(metrics, colors)

        style_dict = {None: "-", "LSTM": "--", "GRU": ":", "Transformer": "-.", "Shockwave": (5, (10, 3))}
        
        max_mAP, max_alpha, max_ls = 0, 0.22, '-'  # For None model (Ground Truth Travel Time)
    
        for temporal_model_name in temporal_model_ls:
            # Load the evaluation result if it already exists
            if temporal_model_name is not None and os.path.exists(self.alpha_cmc_map_fp_dict[temporal_model_name]):
                with open(self.alpha_cmc_map_fp_dict[temporal_model_name], 'r') as f:
                    cmc_map = eval(f.read())
                    for k, (v1, v2) in cmc_map.items():
                        cmc_map[k] = (eval(v1), v2)
            else:
                cmc_map = {}
                for t_alpha in tqdm(np.arange(0.0, 1.01, 0.01)):
                    cmc_map[t_alpha] = self.evaluate(t_alpha, temporal_model_name, dynamic, gt_vlds, temporal_window=0, logging=False)
            
                # Write the evaluation result in a text file
                if temporal_model_name is not None:
                    with open(self.alpha_cmc_map_fp_dict[temporal_model_name], 'w') as f:
                        cmc_map_tmp = {}
                        for k, (v1, v2) in cmc_map.items():
                            cmc_map_tmp[k]=(util.array_to_str(v1), v2)
                        f.write(str(cmc_map_tmp))
        
            max_alpha_tmp = [_ for _ in cmc_map.keys()][np.argmax([_[1] for _ in cmc_map.values()])]
            ax = self.plot_cmc_map(ax, cmc_map, alphas, style_dict[temporal_model_name], metrics, colors, max_alpha_tmp, logging=False)

            max_map_tmp = max([_[1] for _ in cmc_map.values()])
            if temporal_model_name is not None and max_mAP < max_map_tmp:
                max_mAP, max_alpha, max_ls = max_map_tmp, max_alpha_tmp, style_dict[temporal_model_name]

            if logging:
                print("Performance with model : %s" % temporal_model_name)
                print("Max Alpha : %.2f" % max_alpha_tmp)
                print("mAP : %.4f" % cmc_map[max_alpha_tmp][1])
                print("CMC : %.4f, %.4f, %.4f" % (cmc_map[max_alpha_tmp][0][0], cmc_map[max_alpha_tmp][0][4], cmc_map[max_alpha_tmp][0][9]))
            
            if temporal_model_name is not None:
                legend_elements.append(Line2D([0], [0], color='gray', lw=1, ls=style_dict[temporal_model_name], label="%s" % (temporal_model_name)))
            else:
                legend_elements.append(Line2D([0], [0], color='gray', lw=1, ls=style_dict[temporal_model_name], label='Perfect travel time'))

        plt.axvline(x=max_alpha, c="royalblue", ls=max_ls)
        
        ax.set_xticks(np.arange(0, 1.1, 0.1))  # every 0.1
        ax.set_ylim([0.0, 1.01])
        ax.set_ylabel("Re-Identification Rate")
        ax.set_xlabel("α")
        
        title_content = "Combined Framework with Ground Truth Arrival Time" if temporal_model_name is None else "Combined Framework with Predicted Arrival Time"
        title = title_content if title is None else title
        #plt.title(title)
        plt.legend(handles=legend_elements, loc='best')
        plt.show()

        return


    # eval_type = "window" / "sigma"
    def temporal_evaluate(self, eval_type, dynamic, gt_vlds, metrics=["mAP", "CMC-1", "CMC-5", "CMC-10"], colors={"mAP": "royalblue", "CMC-1": "forestgreen", "CMC-5": "orange", "CMC-10": "red"}, logging=False):
        if eval_type is "window":
            _, ax = plt.subplots()
            vars, vars_ls = [0, 1, 10, 20], ["-", "--", "-.", ":"]
        elif eval_type is "sigma":
            _, ax = plt.subplots(figsize=(6.4*2, 4.8))
            vars, vars_ls = [0, 3, 5, 10], ["-", "--", "-.", ":"]
        
        alphas = np.arange(0, 1.01, 0.01)
        legend_elements = self.get_legend_elements(metrics, colors)
        for var, ls in zip(vars, vars_ls):
            cmc_map = {}
            for t_alpha in tqdm(alphas):
                if eval_type is "window":
                    cmc_map[t_alpha] = self.evaluate(t_alpha, None, dynamic, gt_vlds=gt_vlds, temporal_window=var, logging=logging)
                elif eval_type is "sigma":
                    cmc_map[t_alpha] = self.evaluate(t_alpha, None, dynamic, gt_vlds=gt_vlds, temporal_sigma=var, logging=logging)

            ax = self.plot_cmc_map(ax, cmc_map, alphas, ls, metrics=metrics, colors=colors, logging=False)
            legend_elements.append(Line2D([0], [0], color='gray', lw=1, ls=ls, label="%s = %d [s]" % (eval_type, var)))

        ax.legend(handles=legend_elements, loc='best')
        ax.set_xticks(np.arange(0, 1.1, 0.1))  # every 0.1
        ax.set_ylim([0, 1.01])
        ax.set_xlabel("α")
        ax.set_ylabel("Re-Identification Rate")

        #plt.title("Combined Framework with Ground Truth Travel Time and Controlled Noise Added Travel Time")
        plt.show()

        return
    

    def get_legend_elements(self, metrics, colors):
        legend_elements = [Patch(facecolor=colors[metric], edgecolor=colors[metric], label=metric) for metric in metrics] if len(metrics) > 1 else []
        
        return legend_elements


    # max_alpha_tmp : alpha value maximizing the CMC-1 : 0.22
    def plot_cmc_map(self, ax, cmc_map, alphas, ls, metrics, colors, max_alpha_tmp=0.22, logging=False):
        if "mAP" in metrics:
            metric_name = "mAP"
            ax.plot(alphas, [_[1] for _ in cmc_map.values()], c=colors[metric_name], ls=ls)
        if "CMC-1" in metrics:
            metric_name = "CMC-1"
            ax.plot(alphas, [_[0][0] for _ in cmc_map.values()], c=colors[metric_name], ls=ls)
        if "CMC-5" in metrics:
            metric_name = "CMC-5"
            ax.plot(alphas, [_[0][4] for _ in cmc_map.values()], c=colors[metric_name], ls=ls)
        if "CMC-10" in metrics:
            metric_name = "CMC-10"
            ax.plot(alphas, [_[0][9] for _ in cmc_map.values()], c=colors[metric_name], ls=ls)

        if logging:
            print("mAP : %.4f" % cmc_map[max_alpha_tmp][1])
            print("CMC : %.4f, %.4f, %.4f" % (cmc_map[max_alpha_tmp][0][0], cmc_map[max_alpha_tmp][0][4], cmc_map[max_alpha_tmp][0][9]))

        return ax