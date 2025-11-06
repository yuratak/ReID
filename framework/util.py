import cv2
import decimal
from geopy.distance import distance
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.geometry import Point, Polygon


def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)


def get_pixel_distance(pts):
    return np.array([euclidean_distance(x[:2], y[:2]) for x, y in zip(pts[:-1], pts[1:])]).sum()   


# method = "temporal" / "spatial"
def sample_traj(traj, nb_sample=15, method="spatial"):
    if method == "temporal":
        idxs = list(range(0, len(traj), int(len(traj)/nb_sample)))
    elif method == "spatial":
        # Sample the "perfect" uniform equal distances betweens points
        interval_ls = (get_pixel_distance(traj) / nb_sample-1) * np.array([i for i in range(0, nb_sample)])
        
        # The relative distance between points
        traj_dist = np.array([euclidean_distance(p1_[:2], p2_[:2]) for p1_, p2_ in zip(traj[:-1], traj[1:])])
        # The cumulative relative distance between points
        traj_dist_cum = np.cumsum(traj_dist)
        
        # Get the idxs with smaller distance difference with the perfect uniform distances
        idxs = [(np.abs(traj_dist_cum - value)).argmin() for value in interval_ls]

    return traj[idxs]


def get_VLD_pair(vld_nb=4):
    return [(i, j) for i in range(vld_nb) for j in range(vld_nb) if i!=j]


def array_to_str(arr):
    return np.array2string(arr, separator=',').replace(" ", "").replace("\n","")


# get the frame image from the cap
def get_frame(cap, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    return frame


# BBOX, CIRCLE, LINE, TEXTS
# color in BGR
def draw(frame, objects, objects_type="bbox", color=(0, 0, 255), width=None):
    frame = frame.copy()
    if objects_type == "bbox":
        width = 3 if width is None else width
        for obj in objects:
            frame = cv2.drawContours(frame, np.array([obj]), 0, color, width)
    elif objects_type == "circle":
        for obj in objects:
            thickness = 12 if width is None else width
            frame = cv2.circle(frame, obj, radius=0, color=color, thickness=thickness)
    elif objects_type == "line":
        width = 3 if width is None else width
        for obj in objects:
            frame = cv2.line(frame, obj[0], obj[1], color, width)
    elif objects_type == "text":
        width = 6 if width is None else width
        bb_objects, text_objects = objects
        for (bb_obj, txt_obj) in zip(bb_objects, text_objects):
            if len(bb_obj) != 2:
                bb_obj = tuple(map(int, np.min(bb_obj, axis=0)))
            #color = (255, 255, 0) # in bgr format : skyblue
            frame = cv2.putText(frame, str(txt_obj), bb_obj, cv2.FONT_HERSHEY_SIMPLEX, 2, color, width, cv2.LINE_AA)
    return frame


# Visualize lanes
# lanes_dict : either lanes_dict or lanes_trajs_dict
def visualize_lanes(lanes_dict, frame, write=False, title="lane_traj_median.png"):
    frame = frame.copy()
    colors_map_bgr = get_colors_map(bgr=True)

    for vld_nb in lanes_dict.keys():
        for c, ls in lanes_dict[vld_nb].items():
            # single lane case
            if len(ls.shape) == 2:
                ls = ls[np.newaxis, ...]
            for i in range(ls.shape[0]):
                l = ls[i]
                for pt_i in range(1, len(l)):
                    frame = draw(frame, np.array([l[pt_i-1:pt_i+1]]), objects_type='line', color=colors_map_bgr[c], width=2)
                
                frame = draw(frame, np.array(l), objects_type='circle', color=colors_map_bgr[c], width=8)

    if write:
        cv2.imwrite(title, frame)
        
    show_image_inline(frame, figsize=(15, 40))

    return


def get_vid_fps(intersection_name="S", root_dir="/home/yura/data/KPNEUMA/raw/"):
    drone_vid_fps = [os.path.join(root, file) for root, subdirs, files in os.walk(root_dir) for file in files if file.endswith(".mp4") and not file.endswith("aligned.mp4")]
    drone_vid_fps = [fp for fp in drone_vid_fps if fp.split(".")[0].split("_")[-1][0] == intersection_name]
    drone_vid_fps = sorted(drone_vid_fps)

    return drone_vid_fps


# to get grayscale image
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def predictions_on_frame(frame, bbox, color=(0, 0, 255), texts=None, lines=None):
    for i in range(len(bbox)):
        b = bbox[i]
        if len(b) == 4:
            if lines is not None:
                lines_color = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
                frame = cv2.drawContours(frame, np.array([b]), 0, lines_color[lines[i]])
            else:
                frame = cv2.drawContours(frame, np.array([b]), 0, color, 3) # R
            if texts is not None:
                c = tuple(map(int, np.mean(b, axis=0)))
                frame = cv2.putText(frame, str(texts[i]), c, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif len(b) == 2:
            frame = cv2.circle(frame, b, radius=0, color=color, thickness=20)
    
    return frame


def show_image_inline(img, trans_level=1, figsize=None, bgr2rgb=True, axis_off=False):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if figsize:
        plt.figure(figsize=figsize)
    plt.imshow(img, alpha=trans_level)
    if axis_off:
        plt.axis('off')

    plt.show(block=False)
        
    return


# Crop vehicle patch from the frame
def crop_img_rect(pts, img, rotated=False):
    if rotated:
        rect = cv2.minAreaRect(pts)
        im_crop, _ = crop_rect(img, rect)
    else:
        [x_min, y_min], [x_max, y_max] = np.min(pts, axis=0), np.max(pts, axis=0)
        im_crop = img[y_min:y_max, x_min:x_max]
    
    return im_crop


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    rows, cols = img.shape[0], img.shape[1]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))
    out = cv2.getRectSubPix(img_rot, size, center)

    return out, img_rot


def crop_img_rect2(pts, img):
    rect = cv2.minAreaRect(pts)

    # the order of the box points: bottom left, top left, top right, bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # get width and height of the detected rectangle
    width, height = int(rect[1][0]), int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    
    return warped


def get_tuple_bbox(list_bb):
    return tuple(np.concatenate(list_bb))


def get_list_bbox(tuple_bb):
    x1, y1, x2, y2, x3, y3, x4, y4 = tuple_bb
    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


def calculate_IoU(b1, b2):
    poly1, poly2 = Polygon(b1), Polygon(b2)
    
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    
    return intersection/union


def IoMin(p1, p2):
    return p1.intersection(p2).area / min(p1.area, p2.area)


def calculate_union_overlay(row, df1, df2):
    geom1 = df1.iloc[row[row.keys()[0]]]['geometry']
    geom2 = df2.iloc[row[row.keys()[1]]]['geometry']
    geom12 = geom1.union(geom2)

    return geom12.area


def from_bbox_to_polygon(x):
    poly = [Polygon([Point(y) for y in poly]) for poly in x]
    
    return poly


def from_rbbox_to_4pts(bbox):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    xc, yc, w, h, ag = bbox[:5]
    wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
    hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
    p1 = (xc - wx - hx, yc - wy - hy)
    p2 = (xc + wx - hx, yc + wy - hy)
    p3 = (xc + wx + hx, yc + wy + hy)
    p4 = (xc - wx + hx, yc - wy + hy)
    poly = np.int0(np.array([p1, p2, p3, p4]))
        
    return poly, bbox[-1]


def from_4pts_to_rbbox(pts):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    Test:
        from_rbbox_to_4pts(from_4pts_to_rbbox(list(map(int, ['881', '332', '918', '317', '926', '335', '889', '351']))))
    """
    bboxps = np.array(pts).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
    if w < 2 or h < 2:
        return
    while not 0 < a <= 90:
        if a == -90:
            a += 180
        else:
            a += 90
            w, h = h, w
    a = a / 180 * np.pi
    assert 0 < a <= np.pi / 2
    
    return x, y, w, h, a


# to calcualte the distance between 2 GPS coordinates
# Unit : m
def gps_dist(coord_a, coord_b):
    return distance(coord_a, coord_b).m


def float_to_string(number, precision=20):
    return '{0:.{prec}f}'.format(
        decimal.Context(prec=100).create_decimal(str(number)),
        prec=precision,
    ).rstrip('0').rstrip('.') or '0'


def iqr_filtering(a, idx=False):
    q3, q1 = np.percentile(a, [75 ,25])
    iqr = q3 - q1
    
    keep_idx = [i for i in range(len(a)) if a[i] >= np.median(a) - iqr * 1.5]
    
    if idx:
        return keep_idx
    
    return np.array(a)[keep_idx]


def get_colors_map(bgr=False):
    colors_map = { -1: "gray", 0: "blue", 1: "green", 2: "red", 3: "cyan", 4: "magenta", 5: "yellow", 6: "black", 7: "orange", 8: "pink", 9: "purple", 10: "brown", 11: "orchid"}
    colors_map_all = list(set([_ for _ in matplotlib.colors.get_named_colors_mapping().keys() if not _.startswith("xkcd:")]) - set(colors_map.values()))
    
    start_idx = np.max(list(colors_map.keys())) + 1
    for i in range(len(colors_map_all)):
        colors_map[start_idx + i] = colors_map_all[i]
    
    if bgr:
        colors_map_bgr = {k: tuple(int(_*255) for _ in matplotlib.colors.to_rgb(v))[::-1] for (k, v) in colors_map.items()}
        return colors_map_bgr
    else:
        return colors_map