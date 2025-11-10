import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import re
import sys
import torch
from torch.backends import cudnn


sys.path.append('reid_strong_baseline/')
from config import cfg
from data import make_data_loader
from data.transforms.build import build_transforms
from engine.trainer import do_train, do_train_with_center
from layers import make_loss, make_loss_with_center
from modeling import build_model
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
from utils.logger import setup_logger

torch.set_grad_enabled(True)


class VisualComponent:
    def __init__(self, \
                 cfg_vrai_fp='configs/softmax_triplet_with_center_vrai.yml', \
                 cfg_kpneuma_fp='configs/softmax_triplet_with_center_kpneuma.yml'):
        print("[INFO] Visual Component Initialization")
        self.cfg_vrai_fp = cfg_vrai_fp
        self.cfg_kpneuma_fp = cfg_kpneuma_fp
        
        cfg.merge_from_file(self.cfg_kpneuma_fp)
        self.checkpoint_dir = cfg.OUTPUT_DIR
        self.num_classes = 576

        self.device = torch.device('cuda:0')

        return

    def process_reid_data(self):
        cfg.merge_from_file(self.cfg_kpneuma_fp)
        cfg.freeze()

        base_path = os.path.join(cfg.DATASETS.ROOT_DIR, str.upper(cfg.DATASETS.NAMES))
        
        train_dir_path, query_dir_path, gallery_dir_path, train_query_dir_path, train_gallery_dir_path = [os.path.join(base_path, _) for _ in ["images_train", "images_query", "images_gallery", "images_train_query", "images_train_gallery"]]
        train_img_path, query_img_path, gallery_img_path, train_query_img_path, train_gallery_img_path = [os.listdir(_) for _ in [train_dir_path, query_dir_path, gallery_dir_path, train_query_dir_path, train_gallery_dir_path]]

        pattern = re.compile(r'([0-9]+)_([0-9]+)_([0-9]+).png')
        
        train_pid, train_camid, train_framenb = np.split(np.array([list(map(str, pattern.search(_).groups())) for _ in train_img_path]), 3, axis=1)
        query_pid, query_camid, query_framenb = np.split(np.array([list(map(str, pattern.search(_).groups())) for _ in query_img_path]), 3, axis=1)
        gallery_pid, gallery_camid, gallery_framenb = np.split(np.array([list(map(str, pattern.search(_).groups())) for _ in gallery_img_path]), 3, axis=1)
        train_query_pid, train_query_camid, train_query_framenb = np.split(np.array([list(map(str, pattern.search(_).groups())) for _ in train_query_img_path]), 3, axis=1)
        train_gallery_pid, train_gallery_camid, train_gallery_framenb = np.split(np.array([list(map(str, pattern.search(_).groups())) for _ in train_gallery_img_path]), 3, axis=1)
        
        train_img_path = [os.path.join(train_dir_path, _) for _ in train_img_path]
        query_img_path = [os.path.join(query_dir_path, _) for _ in query_img_path]
        gallery_img_path = [os.path.join(gallery_dir_path, _) for _ in gallery_img_path]
        train_query_img_path = [os.path.join(train_query_dir_path, _) for _ in train_query_img_path]
        train_gallery_img_path = [os.path.join(train_gallery_dir_path, _) for _ in train_gallery_img_path]
        
        self.num_query, self.num_query_train = len(query_img_path), len(train_query_img_path)
        self.train_fp, self.query_fp, self.gallery_fp, self.val_fp, self.train_qg_fp = train_img_path, query_img_path, gallery_img_path, query_img_path+gallery_img_path, train_query_img_path+train_gallery_img_path
        self.train_pid, self.train_camid = train_pid.flatten(), train_camid.flatten().astype(int)
        self.val_pid, self.val_camid = np.concatenate([query_pid.flatten(), gallery_pid.flatten()]), np.concatenate([query_camid.flatten(),gallery_camid.flatten()]).astype(int)
        self.train_qg_pid, self.train_qg_camid = np.concatenate([train_query_pid.flatten(), train_gallery_pid.flatten()]), np.concatenate([train_query_camid.flatten(), train_gallery_camid.flatten()]).astype(int)

        self.feats = self.feature_extractor(self.val_fp).to(self.device)
        self.feats_train = self.feature_extractor(self.train_qg_fp).to(self.device)

        return


    def read_imgs(self, img_ls):
        if type(img_ls[0]) is str:
            img_ls = [Image.open(_).convert('RGB') for _ in img_ls]
        else:
            img_ls = [Image.fromarray(_.detach().numpy()) for _ in img_ls]
        img_ls = [self.transform(_) for _ in img_ls]
        torch_data = torch.stack(img_ls)
        
        return torch_data


    def feature_extractor(self, img_ls, batchsize=64):
        feats = []

        for b in torch.utils.data.DataLoader(img_ls, batch_size=batchsize):
            b_imgs = self.read_imgs(b).to(self.device)
            
            with torch.no_grad():
                b_feats = self.model(b_imgs)
            feats.append(b_feats)

        feats = torch.cat(feats)
        
        return feats


    # 1 cfg file for VRAI training and 1 cfg file for KPNEUMA training
    def train(self, method="last", VRAI_train=False):
        if VRAI_train:
            # Pretrain with VRAI dataset
            cfg.merge_from_file(self.cfg_vrai_fp)
            cfg.freeze()

            self._train(cfg, freeze_ResNet=False) # Train the entire model
        
        #given the previous trained model, finetune with kpneuma dataset
        vrai_ep, vrai_model_fp = self.get_best_ep_model("VRAI", method, viz=False)
        print("[INFO] VRAI Best Epoch: %d with method %s" % (vrai_ep, method))

        cfg.merge_from_file(self.cfg_kpneuma_fp)
        cfg.merge_from_list(['DATASETS.ROOT_DIR', "('data')", \
                            'MODEL.PRETRAIN_PATH', "('%s')" % vrai_model_fp])
        cfg.freeze()
    
        # Train with KPNEUMA dataset
        self._train(cfg, freeze_ResNet=True) # Train only classifier layer

        return
    

    def _train(self, cfg, freeze_ResNet=False):
        # prepare log
        output_dir = cfg.OUTPUT_DIR
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger = setup_logger(f"reid_baseline_{cfg.DATASETS.NAMES}", output_dir, 0)
        cfg_path = self.cfg_kpneuma_fp if cfg.DATASETS.NAMES == "kpneuma" else self.cfg_vrai_fp
        logger.info("Loaded configuration file {}".format(cfg_path))
        with open(cfg_path, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

        if cfg.MODEL.DEVICE == "cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
        cudnn.benchmark = True
        
        # prepare dataset
        train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

        # prepare model
        model = build_model(cfg, num_classes)

        if freeze_ResNet:
            for param in model.base.parameters():
                param.requires_grad = False

        if cfg.MODEL.IF_WITH_CENTER == 'yes':
            print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        
            loss_func, center_criterion = make_loss_with_center(cfg, num_classes)  # modified by gu
            optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
        else:
            print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
            loss_func = make_loss(cfg, num_classes)
            optimizer = make_optimizer(cfg, model)

        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            # YT : Separate VRAI and KPNEUMA Training
            start_epoch = 0 #eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
            print('Path to the checkpoint of center_param:', path_to_center_param)
            path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
            print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
            
            param_dict = torch.load(cfg.MODEL.PRETRAIN_PATH)
            # X loading the classifier layer
            # To avoid conflict when loading different models trained on different datasets
            for i in param_dict:
                if 'classifier' in i:
                    continue
                model.state_dict()[i].copy_(param_dict[i])

            # Freeze 90% initial layers
            if cfg.MODEL.FREEZE > 0.0:
                for i, param in enumerate(model.parameters()):
                    if i < int(len(list(model.parameters()))*cfg.MODEL.FREEZE):
                        param.requires_grad = False

            optimizer = make_optimizer(cfg, model)
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        if cfg.MODEL.IF_WITH_CENTER == 'yes':
            do_train_with_center(
                cfg,
                model,
                center_criterion,
                train_loader,
                val_loader,
                optimizer,
                optimizer_center,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch,     # add for using self trained model
                state_dict=True
            )
        else:
            do_train(
                cfg,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                loss_func,
                num_query,
                start_epoch,
                state_dict=True # Boolean
            )
        return
    

    def test(self):
        self.load_test_model()
        self.process_reid_data()
        cmc, m = self.evaluate()

        print(f"mAP : {m:.3f}")
        print(f"CMC-1 : {cmc[0]:.3f}")
        print(f"CMC-5 : {cmc[4]:.3f}")
        
        return cmc, m


    def load_test_model(self):
        _, test_weight_fp = self.get_best_ep_model("KPNEUMA", "best")

        cfg.merge_from_file(self.cfg_kpneuma_fp)
        cfg.merge_from_list(['TEST.WEIGHT', "%s" % test_weight_fp])
        cfg.freeze()

        if cfg.MODEL.DEVICE == "cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        cudnn.benchmark = True

        model = build_model(cfg, self.num_classes) # VRAI dataset's train num classes = 576
        model.load_param(cfg.TEST.WEIGHT)

        self.model = model.to(self.device)
        self.model = self.model.eval()
        self.transform = build_transforms(cfg, is_train=False)

        return


    def get_elbow_model_ep(self, log_data, stop_threshold, sub_ep=0):
        log_data = log_data.copy()
        if sub_ep > 0:
            for k in log_data.keys():
                log_data[k] = log_data[k][sub_ep:]
        for i, ep in enumerate(log_data["epoch"]):
            if ep+5 < np.max(log_data["epoch"]) and ep % 5 == 0:
                mAP_gain = np.sum([log_data["mAP"][sub_i+1] - log_data["mAP"][sub_i] for sub_i in range(i, i+5)]) / 5
                #print(mAP_gain)
                if mAP_gain > 0 and mAP_gain < stop_threshold:
                    return ep
        
        return np.max(log_data["epoch"])


    # method = "elbow", "best", "last"
    def get_best_ep_model(self, dataset, method, stop_threshold=0.02, viz=False):
        # Get the best VRAI epoch
        if dataset == "VRAI":
            cfg.merge_from_file(self.cfg_vrai_fp)
        elif dataset == "KPNEUMA":
            cfg.merge_from_file(self.cfg_kpneuma_fp)
        else:
            print(f"[INFO] Extracting the best epoch model for the unknown dataset : {dataset}")
            return
        
        log_data = self.parse_training_log(os.path.join(cfg.OUTPUT_DIR, "log.txt"))
        
        if method == "elbow":
            ep_opt = self.get_elbow_model_ep(log_data, stop_threshold)
        elif method == "best":
            ep_opt = log_data["epoch"][log_data["mAP"].index(max(log_data["mAP"]))]
        elif method == "last":
            ep_opt = log_data["epoch"][-1]
        model_fp = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_model_{ep_opt}.pth")
        
        if viz:
            self.visualize_log_data(log_data, dataset_name=dataset, ep_opt=ep_opt)
        print(f"[Visual Component] model file path {model_fp}")
        
        return ep_opt, model_fp


    def evaluate(self, logging=False):
        feats = torch.nn.functional.normalize(self.feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query] # query features
        q_pids = np.asarray(self.val_pid[:self.num_query]) # pids of same length as qf (pid for each of the image)
        q_camids = np.asarray(self.val_camid[:self.num_query]) # camids of same length as qf (camid for each of the image)
        # gallery
        gf = feats[self.num_query:] # gallery features
        g_pids = np.asarray(self.val_pid[self.num_query:]) # //
        g_camids = np.asarray(self.val_camid[self.num_query:]) # //
        m, n = qf.shape[0], gf.shape[0] # distance matrix size (betweem each query and gallery image, distance calculated)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t() # all 2 matrice of shape m (33 query) x n (724 gallery)
        # COSINE DISTANCE CALCULATION
        #distmat.addmm_(1, -2, qf, gf.t()) # Beta(1) * distmat + Alpha(-2) * (qf @ gf.t()) with qf (33 x 2048) and gf.t() (2048 x 724) -> (33 x 724)
        distmat = distmat.addmm(qf, gf.t(), beta=1, alpha=-2) # Beta(1) * distmat + Alpha(-2) * (qf @ gf.t()) with qf (33 x 2048) and gf.t() (2048 x 724) -> (33 x 724)
        # If qf[i] and gf[j] is the same feature -> L2 norm^2 == 1 -> distmat[i, j] == 0
        distmat = distmat.cpu().detach().numpy()

        cmc, mAP = self.eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        
        if logging:
            print("CMC : %.2f, %.2f, %.2f" % (cmc[0], cmc[5], cmc[10]))
            print("mAP : %.2f" % mAP)

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
    

    def parse_training_log(self, log_fp):
        log_data = {"epoch": [], "mAP": [], "CMC1": [], "CMC5": [], "CMC10": []}
        
        # Plot the VRAI training log
        with open(log_fp) as f:
            lines = f.readlines()
        
        log_start_idx = [i for i, l in enumerate(lines) if "Start training" in l][0]
        lines = lines[log_start_idx:]

        for l_i, l in enumerate(lines):
            if "Validation Results" in l:
                ep = int(l.split("Epoch: ")[-1])
                log_data["epoch"].append(ep)
                log_data["mAP"].append(float(lines[l_i+1].split()[-1].replace("%", "").replace(":", "")))
                log_data["CMC1"].append(float(lines[l_i+2].split()[-1].replace("%", "").replace(":", "")))
                log_data["CMC5"].append(float(lines[l_i+3].split()[-1].replace("%", "").replace(":", "")))
                log_data["CMC10"].append(float(lines[l_i+4].split()[-1].replace("%", "").replace(":", "")))

        return log_data


    def visualize_log_data(self, log_data, dataset_name, ep_opt=None, ep_opt_=None):
        _, ax = plt.subplots()
        ax.set_title("Strong Baseline Training - %s Dataset" % dataset_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Re-ID Performance")
        ax.set_ylim([30, 100])
        plt.plot(log_data["epoch"], log_data["mAP"], label="mAP")
        plt.plot(log_data["epoch"], log_data["CMC1"], label="CMC-1")
        plt.plot(log_data["epoch"], log_data["CMC5"], label="CMC-5")
        plt.plot(log_data["epoch"], log_data["CMC10"], label="CMC-10")
        if ep_opt is not None:
            plt.axvline(x=ep_opt, color="red", linestyle="--", label="Optimal Epoch")
        if ep_opt_ is not None:
            plt.axvline(x=ep_opt_, color="green", linestyle="--", label="Optimal Epoch (KPNEUMA)")
        plt.legend()
        plt.show()

        return