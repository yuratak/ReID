# %%
import cv2
from datetime import datetime
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from VideoLog import VideoLog
from FeatureMatcher import FeatureMatcher
from Intersection import Intersection
import util


# Drone Video
# Input : video_path : fp of the video
#        frame_granularity : by default set to 4
# Output : DroneVideo object
#          ** annot : DataFrame
#          ** annot_per_veh_id : DataFrame
#          ** veh_category : dictionary {v_id (int): category (int)}
# Access : get_df(aligned, per_veh) 
#          per frame : get_df(aligned, per_veh=False).loc[frame_nb]
#          per vehicle : get_df(aligned, per_veh=True).loc[v_id]
#                        get_v_patches(frame_nb, v_id) : only for a single frame_nb
#                        get_v_traj(v_id, start_frame_nb=None, historical_nb=None, chronological=False)
# Assumption : for a video file xxx.mp4, the corresponding annotation file is xxx.txt or if already processed xxx.pkl
#              for a video file xxx.mp4, the corresponding drone log file is xxx.csv
#              for a video file xxx.mp4, the corresponding aligned video file is xxx_aligned.mp4
#              for a video file xxx.mp4, the corresponding aligned annotation file is xxx_aligned.pkl
class DroneVideo:
    video_path = None
    annot_raw_path = None
    annot_path = None
    drone_log_path = None

    # For KPNEUMA 1, 2, ..., 10
    drone_id = None

    # By default, we process the video with 4 frames granularity, i.e. we consider only 1 frame every 4 frames
    frame_granularity = None
    
    # Vehicle Information DataFrame **PER FRAME**
    # Columns : ['frame', 'timestamp', 'iso', 'shutter', 'fnum', 'ev', 'ct', 'color_md', 'focal_len', 'latitude', 'longitude', 'rel_alt', 'abs_alt', 'ID', 'bbox', 'category', 'confidence', 't_relative_s']
    # frame : frame number (int)
    # timestamp : YYYY-MM-DD hh:mm:ss.fff (str)
    # iso : ISO (int) : 130
    # shutter : shutter speed (float) : 1/1000.0
    # fnum : f-number (float) : 2.8
    # ev : exposure value (float) : 0.0
    # ct : color temperature (int) : 5263
    # color_md : color mode (str) : default
    # focal_len : focal length (float) : 24.2
    # latitude : latitude (float) : 37.38011
    # longitude : longitude (float) : 126.65487
    # rel_alt : relative altitude (float) : 148.8
    # abs_alt : absolute altitude (float) : 152.057
    # ID : Vehicle IDs : [1, 2, 3, 4, ...]
    # bbox : bounding box (list) : [ [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ... ]
    # category : vehicle category (list) : [0, 0, 0, 0, ...]
    # confidence : detection confidence score (list) : [0.999, 0.999, 0.999, 0.999, ...]
    # t_relative_s : absolute time in seconds (float) : 54232.46
    annot = None
    # Vehicle Information DataFrame **PER VEHICLE**
    # Columns : ['frame', 'bbox', 't_relative_s']
    annot_per_veh_id = None


    # Frame size (2160, 3840, 3) : x axis = 3840 / y axis = 2160
    video_frame_size = (2160, 3840)


    def __init__(self, video_path, frame_granularity=4) -> None:
        self.video_path = video_path
        #self.video_code = VideoLog().get_video_code(video_path) else (hash(video_path)%100)

        self.annot_raw_path = self.video_path.replace(".mp4", ".txt")
        self.annot_path = self.video_path.replace(".mp4", ".pkl")

        self.video_aligned_path = self.video_path.replace(".mp4", "_aligned.mp4")
        self.annot_aligned_path = self.video_aligned_path.replace(".mp4", ".pkl")

        self.drone_log_path = self.video_path.replace(".mp4", ".csv")

        self.drone_id = int(video_path.split("/")[-4].split("D")[1])
        self.intersection = Intersection(video_path.split("/")[-2][0])

        self.frame_granularity = frame_granularity


        ########################################################
        # **Raw Data Loading**
        self.cap = cv2.VideoCapture(self.video_path, 0)
        self.cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap_frame_nb = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 4

        ########################################################
        # **Drone and Annotation Pre-processing**
        # Input : raw drone video fp
        # 1. Video Pre-processing
        #   - If _aligned.mp4 not exists
        #       # Video Stabilization
        #       Create a new video file (_aligned.mp4)
        # 2. Annotation Pre-processing
        #   - If .pkl not exists
        #       # Annotation Parsing
        #   - If _aligned.pkl not exists
        #       # Video-Annotation Alignment
        #        create a new annotation file (_aligned.pkl)
        # NB : 1 & 2 Merged
        #      First, Annotation Parsing. Then Video Stabilization and the Video-Annotation Alignment toegether
        # 3. Loading the preprocessed video and the annotation file
        #   - video_path = video_aligned_path
        #   - annot_path = annot_aligned_path
        #   - And then load the video and annotation file
        if not os.path.isfile(self.annot_path):
            self.annot = self.create_annot()

        # Then align the video
        if not os.path.isfile(self.video_aligned_path):
            self.align_video_and_annot()

        ########################################################
        # **Data Loading**
        self.cap_aligned = cv2.VideoCapture(self.video_aligned_path)

        self.annot = pd.read_pickle(self.annot_path)
        self.annot_aligned = pd.read_pickle(self.annot_aligned_path)
        
        ########################################################
        # **Annotation Processing**
        # Vehicle ID Prefix using Video Code
        #self.annot.ID = self.annot.ID.map(lambda x: list(map(lambda _: int(f"{self.video_code:02d}{_:06d}"), x)))
        
        # Convert the datetime to daily relative time in miliseconds : t_relative_s
        self.annot["t_relative_s"] = self.extract_relative_timestamp(self.annot)
        self.annot_aligned["t_relative_s"] = self.extract_relative_timestamp(self.annot_aligned)

        # 2 columns : frame, bbox / index : ID (redundant indexes : multiple entries per vehicle ID)
        self.annot_per_veh_id = self.annot[["ID", "frame", "bbox", "t_relative_s"]].explode(['ID', 'bbox']).set_index("ID")
        self.annot_aligned_per_veh_id = self.annot_aligned[["ID", "frame", "bbox", "t_relative_s"]].explode(['ID', 'bbox']).set_index("ID")
        
        # dictionary {v_id (int): category (int)}
        # Needed for the temporal component features
        # TODO : Put it inside of the annot_per_veh_id and create a getter .to_dict() BUT should verify how it handles redundant keys (veh_ids) entries
        self.veh_category = self.annot[["ID", "category"]].explode(['ID', 'category']).groupby("ID").agg(list).category.map(lambda x: round(np.mean(x))).to_dict()
        #self.annot_per_veh_id["category"] = self.annot[["ID", "category"]].explode(['ID', 'category']).groupby("ID").agg(list).category.map(lambda x: round(np.mean(x)))


    # Process the raw annotation file & combine with the drone log file and save it as a pickle file
    def create_annot_old(self):
        # filter_category : [0, 1] to filter car and bus
        def read_yolo_annot(annot_fp, filter_category=None):
            with open(annot_fp) as f:
                tmp = f.readlines()

            # Each Entry
            # <frame_number> <vehicle_id> <x_left> <y_top> <width> <height> <category> <confidence_score> <category>
            # Where <category> : 0 (Car) / 1 (Bus) / 2 (Truck) / 3 (Motorcycle)
            tmp = [_.split() for _ in tmp]
            tmp = [[eval(x) for x in _] for _ in tmp]
            tmp = [ [x[0]-1, x[1], np.array([[x[2], x[3]], [x[2]+x[4], x[3]], [x[2]+x[4], x[3]+x[5]], [x[2], x[3]+x[5]]]), x[-2], x[-1]]  for x in tmp]
            tmp_df = pd.DataFrame(tmp, columns=["frame", "ID", "bbox", "category", "confidence"])
            tmp_df = tmp_df.sort_values(by=['frame', 'ID'])
            if filter_category is not None:
                tmp_df = tmp_df.loc[tmp_df['category'].isin(filter_category)]
            tmp_df = tmp_df.groupby(["frame"]).agg(list)
            
            return tmp_df

        def combine_annot_and_drone_log(drone_log, annot):
            drone_log = drone_log[drone_log["frame"] % self.frame_granularity == 0]
            
            annot = annot[annot.index % self.frame_granularity == 0]
            annot = annot.merge(drone_log, on="frame")
            annot.index = annot.frame
            
            return annot

        annot_tmp = read_yolo_annot(self.annot_raw_path, filter_category=[0])
        drone_log = pd.read_csv(self.drone_log_path)

        annot = combine_annot_and_drone_log(drone_log, annot_tmp)
        annot.to_pickle(self.annot_path)

        return annot
        

    # Process the raw annotation file & combine with the drone log file and save it as a pickle file
    def create_annot(self):
        # filter_category : [0, 1] to filter car and bus
        def read_yolo_annot(annot_fp, filter_category=None):
            with open(annot_fp) as f:
                tmp = f.readlines()

            # Each Entry
            # <frame_number>, <vehicle_id>, <x_center>, <y_center>, <width>, <height>, <x_center_stab>, <y_center_stab>, <width_stab>, <height_stab>, <category>, <confidence_score>, <category>, <bb_height>, <bb_width>
            # Where <category> : 0 (Car) / 1 (Bus) / 2 (Truck) / 3 (Motorcycle)
            tmp = [_.split(',')[:-2] for _ in tmp]
            tmp = [[eval(x) for x in _] for _ in tmp]
            tmp = [ [x[0], x[1], np.array([[x[2]-(x[4]/2), x[3]-(x[5]/2)], [x[2]+(x[4]/2), x[3]-(x[5]/2)], [x[2]+(x[4]/2), x[3]+(x[5]/2)], [x[2]-(x[4]/2), x[3]+(x[5]/2)]]), x[-2], x[-1]]  for x in tmp]
            tmp_df = pd.DataFrame(tmp, columns=["frame", "ID", "bbox", "category", "confidence"])
            tmp_df["bbox"] = tmp_df["bbox"].map(np.int0)
            tmp_df = tmp_df.sort_values(by=['frame', 'ID'])
            if filter_category is not None:
                tmp_df = tmp_df.loc[tmp_df['category'].isin(filter_category)]
            tmp_df = tmp_df.groupby(["frame"]).agg(list)
            
            return tmp_df

        def combine_annot_and_drone_log(drone_log, annot):
            drone_log = drone_log[drone_log["frame"] % self.frame_granularity == 0]
            
            annot = annot[annot.index % self.frame_granularity == 0]
            annot = annot.merge(drone_log, on="frame")
            annot.index = annot.frame
            
            return annot

        annot_tmp = read_yolo_annot(self.annot_raw_path, filter_category=[0])
        drone_log = pd.read_csv(self.drone_log_path)

        annot = combine_annot_and_drone_log(drone_log, annot_tmp)
        annot.to_pickle(self.annot_path)

        return annot


    def align_video_and_annot(self):
        cap = cv2.VideoCapture(self.video_path, 0)
        cap_fourcc = cv2.VideoWriter_fourcc(*"mp4v") #cv2.VideoWriter_fourcc(*'h264') #int(cap.get(cv2.CAP_PROP_FOURCC))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        annot = pd.read_pickle(self.annot_path)

        out = cv2.VideoWriter(self.video_aligned_path, cap_fourcc, cap_fps, self.video_frame_size[::-1], True)

        matcher = FeatureMatcher()

        for frame_nb in tqdm(range(0, self.cap_frame_nb)):
            frame = util.get_frame(cap, frame_nb)

            H = matcher.feature_match_superglue(frame, self.intersection.ref_frame)
            frame_aligned = matcher.transform_image(H, frame, self.intersection.ref_frame)
            out.write(frame_aligned) # Write the aligned image

            if frame_nb % self.frame_granularity == 0:
                bboxes_from = annot.at[frame_nb, "bbox"]
                bboxes_aligned, keep_idxs = matcher.transform_bbox(H, bboxes_from, post_processing=False, frame_size=self.video_frame_size)
                # Useful only if post_processing = True as it may filter out some vehicles
                bboxes_aligned_IDs = np.array(annot.at[frame_nb, "ID"])[keep_idxs].tolist()
                assert(len(bboxes_aligned) == len(bboxes_aligned_IDs))
                annot.at[frame_nb, "bbox"] = bboxes_aligned
                annot.at[frame_nb, "ID"] = bboxes_aligned_IDs

        annot.to_pickle(self.annot_aligned_path)

        out.release()
        
        # Sanity Check : whehter two video have the same lengths
        #cap_aligned = cv2.VideoCapture(self.video_aligned_path, 0)
        #assert(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == int(cap_aligned.get(cv2.CAP_PROP_FRAME_COUNT)))

        return
    

    # Extract relative time (hour, min, sec) in seconds (float)
    def extract_relative_timestamp(self, annot):
        # Get the video timestamp (Year, Month, Day)
        t_suff = datetime.strptime(annot.timestamp.loc[0].split()[0], "%Y-%m-%d") # Access the frame 0
        
        # Calculate a relative timestamp in seconds
        return annot.timestamp.map(lambda x: round((datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") - t_suff).total_seconds(), 2)) # float


    # Video relative time (min, sec) to frame number
    def from_time_to_frame(self, m, s):
        sec = m * 60 + s
        
        return sec * self.cap_fps

    
    # Not used for now
    # TODO : use this function to create the annotation file
    # Need a detection annotation and can performa naive tracking (from util)
    '''def create_annot_with_detector_and_tracking(self):
        # read the drone log
        if not os.path.isfile(self.annot_path):
            annot = pd.read_csv(self.drone_log_path)
            annot = annot[annot["frame"] %  self.frame_granularity == 0]
        else:
            annot = pd.read_pickle(self.annot_path)

        if not "bbox" in annot.columns:
            annot["bbox"] = None
        
        # detection
        model = util.get_oriented_rcnn_model()
        for frame_nb in tqdm(range(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.frame_granularity)):
            if annot.loc[frame_nb, "bbox"] is None:
                frame = util.get_frame(self.cap, frame_nb)
                bbox = util.get_bbox(frame, model)
                annot.at[frame_nb, "bbox"] = bbox

        # tracking
        annot["ID"] = None
        annot.at[0, "ID"] = [_+1 for _ in range(len(annot.loc[0, "bbox"]))]
        id_max = max(annot.loc[0, "ID"])

        for i in tqdm(annot.index):
            if annot.loc[i, "ID"] is None:
                ID_to, id_max = util.naive_tracking(annot.loc[i-4, "bbox"], annot.loc[i, "bbox"], annot.loc[i-4, "ID"], id_max)
                annot.at[i, "ID"] = ID_to

        annot.to_pickle(self.annot_path)
        self.annot = annot

        return annot'''
    
    
    
    # Crop the vehicle patch of id (v_id) from the frame (frame_nb)
    # Possibility to extract multiple v_ids from a frame
    # Return : a list of vehicle patches
    #          if a vehicle is not detected, return [..., None, ...]
    def get_v_patches(self, frame_nb, v_ids):
        if type(v_ids) is not list:
            v_ids = [v_ids]

        tmp_frame = util.get_frame(self.cap, frame_nb)
        tmp_v_patches = []
        for v_id in v_ids:
            veh_df = self.annot_per_veh_id.loc[v_id].set_index("frame")
            bb = veh_df.loc[frame_nb].bbox
            tmp_v_patch = util.crop_img_rect(bb, tmp_frame) if len(bb)>0 else None
            tmp_v_patches.append(tmp_v_patch)

        return tmp_v_patches


    # Extract the trajectory (vehicle center) of a single vehicle (v_id)
    # Default, extract position of the vehicle at a frame (frame_nb)
    # Possibility to extract its historical trajectories (historical_nb > 0 and chronological = False)
    # Possibility to extract its future trajectories (historical_nb > 0 and chronological = True)
    # Return an array of trajectories : [(x, y, t), ...] of shape (historical_nb, 3)
    def get_v_traj(self, v_id, start_frame_nb=None, historical_nb=None, chronological=False):
        df_vid = self.get_df(aligned=True, per_veh=True).loc[v_id].set_index("frame").sort_index(ascending=True)

        # Cut out the trajectory from the start_frame_nb
        if start_frame_nb is not None:
            if chronological:
                df_vid = df_vid.loc[start_frame_nb:] # 528, 532, 536, ...
            else:
                # Reverse the trajectory if chronological = False (for historical trajectories)
                # df_vid.loc[:frame_nb][::-1]
                df_vid = df_vid.loc[:start_frame_nb][::-1] # 528, 524, 520, ...

        # Cut out only trajectory of length historical_nb
        # Using cut-out index (start_idx and end_idx)
        if historical_nb is not None:
            start_idx = 0
            end_idx = len(df_vid) if historical_nb is None else min(start_idx + historical_nb, len(df_vid))
            idxs = list(range(start_idx, end_idx)) # 0, 1, 2, 3, 4, 5, ...

            # Padding with the last values if length of trajectories < historical_nb
            if len(idxs) < historical_nb:
                idxs += [idxs[-1]] * (historical_nb - len(idxs))

            df_vid = df_vid.iloc[idxs]
        
        # From bbox [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] to center point (x, y)
        traj = np.vstack(df_vid["bbox"].map(lambda x: list(map(int, x.mean(axis=0)))).values)
        # Add the timestamp
        timestamp = df_vid["t_relative_s"].values.reshape(-1, 1) # float array (should not be int)
        # Concatenate the trajectory and the timestamp
        out = np.hstack([traj,timestamp]) # (historical_nb, 3)

        out = out if chronological else out[::-1] # Post processing such that it follows the temporal order (ascending time)

        return out


    # Getter
    # annot / annot_per_veh_id / annot_aligned / annot_aligned_per_veh_id
    def get_df(self, aligned, per_veh):
        if aligned and per_veh:
            return self.annot_aligned_per_veh_id
        elif aligned and not per_veh:
            return self.annot_aligned
        elif not aligned and per_veh:
            return self.annot_per_veh_id
        elif not aligned and not per_veh:
            return self.annot
        

    # Given a frame number, get the frame and the annotation of the frame
    # Based on get_df
    def get_frame_annot(self, frame_nb, aligned=False):
        if aligned:
            frame = util.get_frame(self.cap_aligned, frame_nb)
        else:
            frame = util.get_frame(self.cap, frame_nb)
        annot_i = self.get_df(aligned=aligned, per_veh=False) \
                      .loc[[frame_nb]][["bbox", "ID", "t_relative_s"]] \
                      .explode(['bbox', 'ID']).set_index("ID")

        return frame, annot_i
        
        
        

# %%
########################################################
########################################################
debugging = False

from IPython.display import display


# %% Verify the raw (not-aligned) annotation file of vehicle ID 1
if debugging:
    video_path = "/home/yura/data/KPNEUMA/raw/2022-10-04/D10/PM1/S1/2022-10-04_D10_PM1_S1.mp4"
    drone_vid = DroneVideo(video_path)

    display(drone_vid.get_df(aligned=True, per_veh=True).loc[1])



# %% Verify get_v_traj()
if debugging:
    video_path = "/home/yura/data/KPNEUMA/raw/2022-10-04/D10/PM1/S1/2022-10-04_D10_PM1_S1.mp4"
    drone_vid = DroneVideo(video_path)

    v_id = 1
    start_frame_nb = 296
    historical_nb = 2

    print(drone_vid.get_v_traj(v_id, start_frame_nb=start_frame_nb, historical_nb=historical_nb, chronological=False)[::-1])
    print(drone_vid.get_v_traj(v_id, start_frame_nb=start_frame_nb, historical_nb=historical_nb, chronological=True))

# %%
# Visualize just raw bounding boxes on the frame : to check annotation parsing
if debugging:
    from Intersection import Intersection
    import util

    video_path = "/home/yura/data/KPNEUMA/raw/2022-10-04/D10/AM2/T4/2022-10-04_D10_AM2_T4.mp4"

    drone_vid = DroneVideo(video_path)

    fr_400, bbs_400 = drone_vid.get_frame_annot(400, aligned=False)

    frame = Intersection("T").ref_frame
    util.show_image_inline(util.draw(fr_400, bbs_400.bbox, objects_type="bbox"))

# %% Visualize the bboxes of the aligned frame 400 on the reference frame
if debugging:
    from Intersection import Intersection
    import util

    video_path = "/home/yura/data/KPNEUMA/raw/2022-10-04/D10/AM2/S2/2022-10-04_D10_AM2_S2.mp4"
    drone_vid = DroneVideo(video_path)

    frame = Intersection("S").ref_frame
    annot = drone_vid.get_df(aligned=True, per_veh=False).loc[400]
    util.show_image_inline(util.draw(frame, annot.bbox, objects_type="bbox"))

# %% Get frame and annot of frame 0 of the aligned video in two ways
if debugging:
    video_path = "/home/yura/data/KPNEUMA/raw/2022-10-04/D10/AM2/S2/2022-10-04_D10_AM2_S2.mp4"
    drone_vid = DroneVideo(video_path)

    fr, annot_i = drone_vid.get_frame_annot(0, aligned=True)
    util.show_image_inline(fr)
    display(annot_i.head())

    annot_i_ = drone_vid.get_df(aligned=True, per_veh=False).loc[[0]]
    annot_i_ = annot_i_[["ID", "bbox", "t_relative_s"]].explode(['ID', 'bbox']).set_index("ID")
    display(annot_i_.head())

    print(annot_i.equals(annot_i_))
# %%
########################################################
########################################################