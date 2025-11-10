import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('time-prediction/')
from src.DataManager import DataManager
from src.time_models import LSTM, GRU, Transformer


class TemporalComponent:
    def __init__(self, 
                 checkpoint_dir = "saved_models/temporal/",
                 data_path = "data/KPNEUMA/",
                 batch_size=512,
                 nepoch=2000,
                 train=False) -> None:

        self.temporal_models = { "LSTM": None,
                            "GRU": None,
                            "Transformer": None}
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.nepoch = nepoch

        self.data_path = data_path
        self.train_path = os.path.join(self.data_path, "temporal_train.txt")
        self.test_path = os.path.join(self.data_path, "temporal_test.txt")

        self.train_df, self.test_df = self.load_temporal_data()
        self.train_data, self.test_data = self.load_temporal_data_input()
        self.timeseries_temporal_dim = self.train_data["timeseries"].shape[2] # 14 : Nb of samples
        self.timeseries_feat_dim = self.train_data["timeseries"].shape[1] # 6 (x, y, t, v, traj_btwn_vlds_x, traj_btwn_vlds_y)
        self.feat_dim = self.train_data["features"].shape[1] # 7 : Feature dimenstion

        # timeseires : from (train_data, 6, 14) to (6, 14) to (6), then to (1, 6, 1)
        # features : from (train_data, 7) to (1, 7)
        self.scaler = {"timeseries": (self.train_data["timeseries"].min(axis=0).min(axis=-1)[np.newaxis, :, np.newaxis], self.train_data["timeseries"].max(axis=0).max(axis=-1)[np.newaxis, :, np.newaxis]), \
                       "features": (self.train_data["features"].min(axis=0), self.train_data["features"].max(axis=0)), \
                       "label": (self.train_data["label"].min(axis=0), self.train_data["label"].max(axis=0))}

        for k in self.train_data.keys():
            self.train_data[k] = self.min_max_scaler(self.train_data[k], self.scaler[k])
            self.test_data[k] = self.min_max_scaler(self.test_data[k], self.scaler[k])

        # Feature Normalization : Inside of the Data Manager
        self.dm = DataManager(self.train_data, self.test_data, self.batch_size)

        for temporal_model_name in self.temporal_models.keys():
            self.temporal_models[temporal_model_name] = self.create_temporal_model(temporal_model_name)

        # For confidence interval in termporal distance matrix creation
        self.temporal_models_loss = {}
        self.temporal_models_ep = {}

        if not train:
            self.load_temporal_model()
        else:
            self.train()

        return
    

    def min_max_scaler(self, d, scaler, inverse=False):
        scaler_min, scaler_max = scaler
        if inverse:
            return d * (scaler_max - scaler_min) + scaler_min
        else:
            return (d - scaler_min) / (scaler_max - scaler_min)
    
    
    def load_temporal_data(self):
        def str_to_array(x):
            return np.array([eval(_) for _ in x[1:-1].replace(', ', ',').split(" ")])
        
        col_names = ["id", "traj_src", "vel_src", "category", "vld_nb_src", "vld_nb_dst", "lane_nb_src", "lane_nb_dst", "lane_signalized_src", "time_to_green", "distance", "traj_btwn_vlds", "avg_travel_time_btwn_vlds", "y"]

        self.train_df = pd.read_csv(self.train_path, sep=";", names=col_names, dtype={"id": str, "category": int})
        self.test_df = pd.read_csv(self.test_path, sep=";", names=col_names, dtype={"id": str, "category": int})
        
        # Pre-processing trajectories string
        for df in [self.train_df, self.test_df]:
            for c in ["traj_src", "vel_src"]:
                df[c] = df[c].map(lambda x: str_to_array(x))

            df["traj_btwn_vlds"] = df["traj_btwn_vlds"].map(lambda x: {k: eval(v) for k, v in eval(x).items()})
            df["avg_travel_time_btwn_vlds"] = df["avg_travel_time_btwn_vlds"].map(lambda x: eval(x))

        # Feature
        # Turning Ratio : calculate the turning ratio for each of the pair of VLDs using TRAINING data only
        # The calculated turning ratios will be added as features to train and test data
        turning_ratio = self.train_df.groupby(["vld_nb_dst", "vld_nb_src"]).count()["id"].reset_index()
        turning_ratio = turning_ratio.set_index("vld_nb_dst")
        turning_ratio["id"] /= turning_ratio.groupby(["vld_nb_dst"]).sum()["id"]
        # Dictionary of turning ratio : {(vld_nb_dst, vld_nb_src) : turning_ratio)}
        turning_ratio = turning_ratio.reset_index().set_index(["vld_nb_dst", "vld_nb_src"]).to_dict()["id"]
        for df in [self.train_df, self.test_df]:
            df["turning_ratio"] = df.apply(lambda x: turning_ratio[x["vld_nb_dst"], x["vld_nb_src"]], axis=1)
        
        # Process traj_btwn_vlds and avg_travel_time_btwn_vlds
        self.train_df["traj_btwn_vlds"] = self.train_df.apply(lambda x: x["traj_btwn_vlds"][(x["vld_nb_src"], x["vld_nb_dst"])][:14], axis=1)
        self.train_df["avg_travel_time_btwn_vlds"] = self.train_df.apply(lambda x: x["avg_travel_time_btwn_vlds"][(x["vld_nb_src"], x["vld_nb_dst"])],axis=1)

        self.test_df["traj_btwn_vlds_nb"] = self.test_df["traj_btwn_vlds"].map(lambda x: x.keys())
        self.test_df["traj_btwn_vlds"] = self.test_df["traj_btwn_vlds"].map(lambda x: list(x.values()))
        self.test_df["avg_travel_time_btwn_vlds"] = self.test_df["avg_travel_time_btwn_vlds"].map(lambda x: x.values())
        self.test_df = self.test_df.explode(["traj_btwn_vlds_nb", "traj_btwn_vlds", "avg_travel_time_btwn_vlds"])
        self.test_df["traj_btwn_vlds"] = self.test_df["traj_btwn_vlds"].map(lambda x: x[:14])

        self.train_df["start_time"] = self.train_df["traj_src"].map(lambda x: x[0][0][2])
        self.test_df["start_time"] = self.test_df["traj_src"].map(lambda x: x[0][0][2])
        self.train_df["end_time"] = self.train_df["start_time"] + self.train_df["y"]
        self.test_df["end_time"] = self.test_df["start_time"] + self.test_df["y"]

        return self.train_df, self.test_df
    

    def load_temporal_data_input(self):
        train_data, test_data = {}, {}
        data = [train_data, test_data]

        for i, df_ in enumerate([self.train_df.copy(), self.test_df.copy()]):
            data[i]["label"] = df_["y"].values.astype(float)

            # Timeseries
            df_["traj_src"] = df_["traj_src"].map(lambda x: np.split(np.array(x), 3, axis=-1))
            df_["traj_src_x"], df_["traj_src_y"], df_["traj_src_time"] = df_["traj_src"].map(lambda x: x[0]), df_["traj_src"].map(lambda x: x[1]), df_["traj_src"].map(lambda x: x[2])
            for c in ["traj_src_x", "traj_src_y", "traj_src_time", "vel_src"]:
                df_[c] = df_[c].map(lambda x: np.array(x).flatten()) #data[i][c].map(lambda x: [_[0] for _ in x])

            # without destination information
            df_ = df_.drop(columns=["id", "y", "traj_src"]).reset_index(drop=False)

            tmp_timeseries = df_[["traj_src_x", "traj_src_y", "traj_src_time", "vel_src", "traj_btwn_vlds"]].values
            tmp_timeseries = np.array([np.column_stack([np.vstack(y) for y in x]) for x in tmp_timeseries])
            tmp_timeseries = np.moveaxis(tmp_timeseries, 1, 2) # (nb_sample, temporal_dim, nb_feature)

            data[i]["timeseries"] = tmp_timeseries.astype(float)
            data[i]["features"] = df_[["vld_nb_src", "vld_nb_dst", "lane_nb_src", "lane_signalized_src", "time_to_green", "distance", "avg_travel_time_btwn_vlds"]].values.astype(float)

        return train_data, test_data


    # Get the test_df or a dataframe with the ground truth vld pairs
    def get_df_w_gt_vld_pair(self, df=None, idx_flag=False):
        if df is None:
            df = self.test_df
            df = df.set_index("id") # df's index should be id (XXxxxxxx)
        
        gt_vld_pair = list(zip(self.test_df["vld_nb_src"], self.test_df["vld_nb_dst"]))

        if idx_flag:
            df = self.test_df.traj_btwn_vlds_nb.reset_index(drop=True)
            df = pd.concat([df, pd.Series(gt_vld_pair, name="gt_vld_pair")], axis=1)
            return list(df[df["gt_vld_pair"] ==  df["traj_btwn_vlds_nb"]].index)
        else:
            df["gt_vld_pair"] = gt_vld_pair
            df = df[df["gt_vld_pair"] ==  df["traj_btwn_vlds_nb"]]
            df = df.drop("gt_vld_pair", axis=1)
            return df
        

    # criterion = nn.MSELoss() / nn.L1Loss()
    def create_temporal_model(self, temporal_model_name):
        def set_model(model_type):
            if model_type == 'GRU':
                model = GRU(self.timeseries_feat_dim, self.feat_dim, hidden=16, num_layer=2, out_feature=1) # D : 2 if bidirectional / 1 otherwise
            if model_type == 'LSTM':
                model = LSTM(self.timeseries_feat_dim, self.feat_dim, hidden=16, num_layer=2, out_feature=1) # num_layer : D : 2 if bidirectional / 1 otherwise
            if model_type == "Transformer":
                model = Transformer(self.timeseries_feat_dim, self.feat_dim)

            return model
        
        def weight_init(m):
            if isinstance(m, torch.nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            
            return
        
        temporal_model = set_model(temporal_model_name)
        temporal_model = temporal_model.to(self.device)
        temporal_model.apply(weight_init)
        
        return temporal_model
    

    def train_temporal_model_(self, model, optimizer, scheduler, criterion, checkpoint_dir, temporal_model_name):
        log_path = os.path.join(self.checkpoint_dir, temporal_model_name+'.txt')
        torch.manual_seed(7)
        
        # Avoid to train if the model is already trained
        #if os.path.isfile(log_path):
        #    print("[INFO] %s Model is already trained" % temporal_model_name)
        #    return None
        
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        with open(log_path, "w") as f:
            f.write("[INFO] Starting the training - %s Model\n" % temporal_model_name)
        for epoch in range(self.nepoch):
            total_train_loss = []
            total_test_loss = []
            for input_timeseries, input_features, label, _ in self.dm.train_loader:
                input_timeseries = input_timeseries.to(self.device)
                input_features = input_features.to(self.device)
                label = label.to(self.device)
                model = model.to(self.device)

                output = model(input_timeseries, input_features)
                
                output_1 = output.squeeze().reshape(-1)
                label = label.reshape(-1)
                
                loss = criterion(output_1, label)
                total_train_loss.append(loss.detach().item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            if epoch % 100 == 0:
                with torch.no_grad():
                    for test_input_timeseries, test_input_features, test_label, _ in self.dm.test_loader:
                        test_input_timeseries = test_input_timeseries.to(self.device)
                        test_input_features = test_input_features.to(self.device)
                        test_label = test_label.to(self.device)
                       
                        test_output = model(test_input_timeseries, test_input_features)
                        
                        test_output_1 = test_output.squeeze().reshape(-1)
                        test_label = test_label.reshape(-1)
                        test_loss = criterion(test_output_1, test_label)
                        total_test_loss.append(test_loss.detach().item())
                print("==============================================")
                print('Train:Epoch:%d/%d, loss:%.3f' % (epoch, self.nepoch, np.mean(total_train_loss)))
                print('Test:Epoch:%d/%d, loss:%.3f,' % (epoch, self.nepoch, np.mean(total_test_loss)))
                
                with open(log_path, "a") as f:
                    f.write("==============================================\n")
                    f.write('Train:Epoch:%d/%d, loss:%.3f\n' % (epoch, self.nepoch, np.mean(total_train_loss)))
                    f.write('Test:Epoch:%d/%d, loss:%.3f\n' % (epoch, self.nepoch, np.mean(total_test_loss)))
                    
                save_path = os.path.join(checkpoint_dir, temporal_model_name+'_{:02d}.pth'.format(epoch))
                torch.save(model.cpu().state_dict(), save_path)
                model.to(self.device)
        
        return model


    def train(self):
        # Train Temporal Model 
        for temporal_model_name in self.temporal_models.keys():
            temporal_model = self.temporal_models[temporal_model_name]
            criterion = nn.L1Loss()
            
            if temporal_model_name in ["GRU", "LSTM", "Transformer"]:
                optimizer = optim.Adam(temporal_model.parameters(), lr=0.00005, eps=1e-8, betas=(0.5, 0.999))
            else:
                optimizer = optim.Adam(temporal_model.parameters())
            
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

            self.temporal_models[temporal_model_name] = self.train_temporal_model_(temporal_model, optimizer, scheduler, criterion, self.checkpoint_dir, temporal_model_name)
        
        return

    def load_temporal_model(self):
        for temporal_model_name in self.temporal_models.keys():
            # Load the best model for each temporal model parsing the training log
            if os.path.isfile(os.path.join(self.checkpoint_dir, "%s.txt" % temporal_model_name)):
                log_data = self.parse_training_log(os.path.join(self.checkpoint_dir, "%s.txt" % temporal_model_name))
                eps, loss_te = log_data["epoch"], log_data["mse_te"]
                ep_opt = eps[loss_te.index(min(loss_te))]

                print(f"[Temporal Component] model file path {os.path.join(self.checkpoint_dir, temporal_model_name+'_{:02d}.pth'.format(ep_opt))}")

                checkpoint_path = os.path.join(self.checkpoint_dir, temporal_model_name+'_{:02d}.pth'.format(ep_opt))

                self.temporal_models[temporal_model_name].load_state_dict(torch.load(checkpoint_path))
                self.temporal_models[temporal_model_name].to(self.device)

                self.temporal_models_ep[temporal_model_name] = ep_opt
                self.temporal_models_loss[temporal_model_name] =self.min_max_scaler(min(loss_te), self.scaler["label"], inverse=True) #self.scaler_label.inverse_transform(np.sqrt(min(loss_te)).reshape(1, -1))[0][0]
        return
    

    # avg_v calculated from train_df : ( (y - ttg) / d ).mean() # 0.007326853622979876
    def baseline_travel_time(self, ttg, d, avg_v=0.007326853622979876):
        avg_v = ( (self.train_df["y"] - self.train_df["time_to_green"]) / self.train_df["distance"] ).mean()
        
        return d * avg_v + ttg


    def test(self, temporal_model_name, order=None):
        # YT : baseline model
        if temporal_model_name == "baseline":
            preds = self.test_df.apply(lambda x: self.baseline_travel_time(x["time_to_green"], x["distance"]), axis=1).values

            tmp = pd.DataFrame(preds, index=self.test_df.index, columns=["preds"])
            tmp["id"] = self.test_df.id
            tmp["traj_btwn_vlds_nb"] = self.test_df["traj_btwn_vlds_nb"]

            return tmp.set_index("id")

        data_timeseries, data_features = torch.FloatTensor(self.test_data["timeseries"]).to(self.device), torch.FloatTensor(self.test_data["features"]).to(self.device)

        model = self.temporal_models[temporal_model_name]
        
        with torch.no_grad():
            preds = model(data_timeseries, data_features).cpu().numpy().flatten()
        # for inverse transformation
        preds = self.min_max_scaler(preds, self.scaler["label"], inverse=True).flatten() #self.scaler_label.inverse_transform(preds.reshape(-1, 1)).flatten()
        
        # clip negative prediction to 0
        preds = np.clip(preds, 0, None)
        
        tmp = pd.DataFrame(preds, index=self.test_df.index, columns=["preds"])
        tmp["id"] = self.test_df.id
        tmp["traj_btwn_vlds_nb"] = self.test_df["traj_btwn_vlds_nb"]

        return tmp.set_index("id")


    def extract_temporal_feature(self, temporal_model_name):
        model = self.temporal_models[temporal_model_name]
        model.eval()
        
        train_data_timeseries, train_data_features = torch.FloatTensor(self.train_data["timeseries"]).to(self.device), torch.FloatTensor(self.train_data["features"]).to(self.device)
        test_data_timeseries, test_data_features = torch.FloatTensor(self.test_data["timeseries"]).to(self.device), torch.FloatTensor(self.test_data["features"]).to(self.device)

        with torch.no_grad():
            train_feats = model(train_data_timeseries, train_data_features, feat_flag=True).cpu().numpy()
            test_feats = model(test_data_timeseries, test_data_features, feat_flag=True).cpu().numpy()
        
        gt_test_idxs = self.get_df_w_gt_vld_pair(idx_flag=True)
        test_feats = test_feats[gt_test_idxs]

        return train_feats, test_feats

    
    # RMSE of model
    def evaluate(self, temporal_model_name, metric="RMSE"):
        y_pred = self.get_df_w_gt_vld_pair(self.test(temporal_model_name)).preds.values
        y_true = self.get_df_w_gt_vld_pair().y.values

        if metric == "RMSE":
            error = mean_squared_error(y_true, y_pred, squared=False)
        elif metric == "MAE":
            error = mean_absolute_error(y_true, y_pred)
        
        return error
    

    def visualize_histogram_temporal_data(self):
        # Test dataframe contains multiple entires for unique vehicle -> Remove redundant travel time values
        test_travel_times = self.get_df_w_gt_vld_pair().y.values
        train_travel_times = self.train_df.y.values

        fig = plt.figure(figsize=(12, 5))
        # Travel Time Histogram for Train
        plt.hist(x=train_travel_times, bins=100, alpha=0.2 ,label="Train", histtype='stepfilled', ec="blue", density=True)
        plt.hist(x=test_travel_times, bins=100, alpha=0.2, label="Test", histtype='stepfilled', ec="orangered", density=True)
        plt.hist(np.concatenate([train_travel_times, test_travel_times]), bins=100, color="lightgreen", alpha=0.1, label="Train and Test", histtype='stepfilled', ec="green", density=True)
        plt.xlabel("Travel Time [s]")
        plt.legend()
        plt.savefig('travel_time_hist.pdf', bbox_inches='tight', format='pdf')
        plt.show()

        return


    def parse_training_log(self, log_fp):
        log_data = {"epoch": [], "mse_tr": [], "mse_te": []}
        
        # Plot the VRAI training log
        with open(log_fp) as f:
            lines = f.readlines()
        
        for l in lines:
            if l.startswith("Train"):
                ep = int(l.split("/")[0].split("Epoch:")[-1])
                log_data["epoch"].append(ep)
                log_data["mse_tr"].append(float(l.split()[-1].split(":")[-1]))
            elif l.startswith("Test"):
                log_data["mse_te"].append(float(l.split()[-1].split(":")[-1]))

        return log_data
    

    def visualize_log_data(self):
        # Plot training curve by reading the training log text file
        for temporal_model_name in self.temporal_models.keys():
            log_data = self.parse_training_log(os.path.join(self.checkpoint_dir, "%s.txt" % temporal_model_name))

            eps, loss_tr, loss_te = log_data["epoch"], log_data["mse_tr"], log_data["mse_te"]

            ep_opt = eps[loss_te.index(min(loss_te))]

            plt.plot(eps, loss_tr, label="Train MSE")
            plt.plot(eps, loss_te, label="Test MSE")
            plt.axvline(x=ep_opt, color="red", linestyle="--", label="Optimal Epoch")
            plt.xlabel("[ep]")
            plt.ylabel("[MSE]")
            plt.legend()
            plt.title("%s Model - MSE Loss" % temporal_model_name)
            plt.show()

        return
    

    def visualize_scatter_gts_preds(self, temporal_model_name="GRU"):
        tmp = self.test(temporal_model_name)

        prs = self.get_df_w_gt_vld_pair(tmp).preds.values
        gts = self.get_df_w_gt_vld_pair().y.values

        _, ax = plt.subplots(figsize=(5, 5))

        plt.scatter(gts, prs, c='orange')
        plt.axis('square')
        plt.plot([0, 1], [0, 1], transform=ax.transAxes)

        ax.set_xlabel("Ground Truth Travel Time [s]", fontsize=20)
        ax.set_ylabel("Predicted Travel Time [s]", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(f'temporal_scatter_{temporal_model_name}.pdf', bbox_inches='tight', format='pdf')
        plt.show()

        return
    

    def visualize_histogram_errors(self, temporal_model_name="GRU"):
        tmp = self.test(temporal_model_name)

        prs = self.get_df_w_gt_vld_pair(tmp).preds.values
        gts = self.get_df_w_gt_vld_pair().y.values
        errs = prs - gts

        _, ax = plt.subplots(figsize=(5, 5))

        ax.set_xlabel("Error [s]", fontsize=20)
        ax.set_ylabel("Error Distribution", fontsize=20)
        ax.set_xlim(-120, 120)
        ax.set_ylim(0, 0.3)

        plt.hist(errs, bins=100, density=True)

        # Draw Gaussian Distribution of the error histogram
        mu = np.mean(errs)
        sigma = np.std(errs)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100) 
        p = scipy.stats.norm.pdf(x, mu, sigma) 
        
        print(f"Error Distribution for {temporal_model_name} : mu={mu}, sigma={sigma}")

        density = stats.kde.gaussian_kde(errs)
        plt.plot(x, density(x))
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(f'temporal_hist_{temporal_model_name}.pdf', bbox_inches='tight', format='pdf')
        plt.show()

        return