# %%
import numpy as np

from IPython.display import display

from DroneVideo import DroneVideo
from VLD import VirtualLoopDetector
from Intersection import Intersection

from VideoLog import VideoLog


# For query vehicle : always look for leader_end
# For gallery vehicle : always look for leader_start
class SpatialComponent:
    def __init__(self):
        self.intersection = Intersection("S")
        self.q_pids_leader_dict, self.g_pids_leader_dict = self.get_leader_vehicle_dict()

        return
    

    # Leader vehicle dictionary {pid: pid_leader}
    def get_leader_vehicle_dict(self):
        # leader_start and leader_end
        # If no leader observed from the video, then output itself
        def leader_vehicle(df, v_id, vld_nb, lane_nb, abs_t, type_="start"):
            vld_nb_col = "start_vld_nb" if type_ == "start" else "end_vld_nb"
            lane_nb_col = "lane_nb_start" if type_ == "start" else "lane_nb_end"
            abs_t_col = "start_abs_t" if type_ == "start" else "end_abs_t"

            df = df[df[vld_nb_col] == vld_nb]
            df = df[df[lane_nb_col] == lane_nb]
            df = df[df[abs_t_col] < abs_t].sort_values(abs_t_col, ascending=False)

            # There are vehicles crossing the VLD before the vehicle v_id
            if len(df) > 0 :
                # TODO : If type_ is start : check that both vehicles passing vlds are not split between red and green
                #df.iloc[0][abs_t_col] and abs_t should be in the same phase
                potential_leader_v_id = int(df.index[0])
                if potential_leader_v_id not in veh_ids_train:
                    return int(df.index[0])
            
            return v_id
    
        q_pids_leader_dict, g_pids_leader_dict = {}, {}

        log = VideoLog("/home/yura/data/KPNEUMA/log.txt")
        for _, row in log.log_df.iterrows():
            drone_vid = DroneVideo(row["video_path"])
            video_code = row["video_code"]
            veh_ids_train = row["veh_ids"]

            vld = VirtualLoopDetector(drone_vid)
            intersection = Intersection("S")
            vld = intersection.assign_lane(vld)

            veh_ids_test = log.get_test_veh_ids(vld, row["veh_ids"])

            # Consider only test data
            df = vld.get_veh_snapshot_df(v_ids=veh_ids_test)

            df["leader_start"] = df.apply(lambda x: leader_vehicle(df, x.name, x.start_vld_nb, x.lane_nb_start, x.start_abs_t, type_="start"), axis=1)
            df["leader_end"] = df.apply(lambda x: leader_vehicle(df, x.name, x.end_vld_nb, x.lane_nb_end, x.end_abs_t, type_="end"), axis=1)

            g_pids_leader_dict = {**g_pids_leader_dict, **dict(zip(df.index.map(lambda x: f"{video_code:02}{x:06}"), df.leader_start.map(lambda x: f"{video_code:02}{x:06}")))}
            q_pids_leader_dict = {**q_pids_leader_dict, **dict(zip(df.index.map(lambda x: f"{video_code:02}{x:06}"), df.leader_end.map(lambda x: f"{video_code:02}{x:06}")))}

        return g_pids_leader_dict, q_pids_leader_dict
    
    # type : "query" or "gallery
    def get_leader_vehicle(self, pids, type_="query"):
        if type_ == "query":
            return [self.q_pids_leader_dict[i] for i in pids]
        else:
            return [self.g_pids_leader_dict[i] for i in pids]


    #def get_lane_group(self):





# %%
########################################################
########################################################

# %%
#from VReID import VReID
#reid = VReID()