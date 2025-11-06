import cv2
import numpy as np
import rasterio
import torch

from superglue.models.matching import Matching
import util

torch.set_grad_enabled(False)

class FeatureMatcher:
    ### SuperGlue Parameters ###
    resize = None
    resize_float = None

    def __init__(self, name="SuperGlue") -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = name
        
        if self.name == "SuperGlue":
            # Due to the GPU memory issue, resize the image to 512 x ... (keep the aspect ratio)
            self.matcher = self.load_superglue(match_threshold = 0.7, max_keypoints = 2048)
            self.match = self.feature_match_superglue
        elif self.name == "SIFT":
            self.matcher = self.load_SIFT()
            self.match = self.feature_match_SIFT


    def load_SIFT(self):
        sift = cv2.SIFT_create() # cv2.xfeatures2d.SIFT_create()

        return sift
    
    def load_superglue(self, **kwargs):
        # Handle kwargs
        self.resize = kwargs.get('resize', -1)
        self.resize_float = kwargs.get('resize_float', False)
        superglue = kwargs.get('superglue', 'outdoor')
        max_keypoints = kwargs.get('max_keypoints', 1024)
        keypoint_threshold = kwargs.get('keypoint_threshold', 0.005)
        nms_radius = kwargs.get('nms_radius', 4)
        sinkhorn_iterations = kwargs.get('sinkhorn_iterations', 20)
        match_threshold = kwargs.get('match_threshold', 0.6)

        if self.resize == -1:
            pass
        elif self.resize > 0:
            print('Will resize max dimension to {}'.format(self.resize))
        elif len(self.resize) == 2:
            print('Will resize to {}x{} (WxH)'.format(
                self.resize[0], self.resize[1]))
        else:
            raise ValueError('Cannot specify more than two integers for --resize')

        # Load the SuperPoint and SuperGlue models.
        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }

        matching = Matching(config).eval().to(self.device)
        
        return matching


    def feature_match_superglue(self, from_image, to_image):
        rot0, rot1 = 0, 0

        # Gray the image
        from_image_gray, target_image_gray = util.gray(from_image), util.gray(to_image)

        # Load the image pair.
        image0, inp0, scales0 = self.read_image(target_image_gray, self.device, self.resize, rot0, self.resize_float)
        image1, inp1, scales1 = self.read_image(from_image_gray, self.device, self.resize, rot1, self.resize_float)
        
        if image0 is None or image1 is None:
            print('Problem reading image pair!!!')
            exit(1)

        # Perform the matching.
        # set_grad_enabled prevents tracking via autograd, making the inference mode more efficient
        # it solves the CUDA memory error
        # By not calculating the gradient, the memory is not allocated for the gradient
        with torch.set_grad_enabled(False):
            pred = self.matcher({'image0': inp0, 'image1': inp1})
        #pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        #mconf = conf[valid]

        if len(mkpts1) >= 4:
            homography_matrix, _ = cv2.findHomography(mkpts1, mkpts0, method=cv2.LMEDS)
            H_affine, _ = cv2.estimateAffinePartial2D(mkpts1, mkpts0, method=cv2.RANSAC, ransacReprojThreshold=4)
            
            return {'perspective': homography_matrix, 'affine': H_affine}
        
        else:
            # If SuperGlue fails, try SIFT
            return self.feature_match_SIFT(from_image, to_image, matcher_explicit=self.load_SIFT())

    

    def feature_match_SIFT(self, from_image, to_image, matcher_explicit=None):
        def filter_matches(matches, ratio=0.75):
            filtered_matches = []
            for m in matches:
                if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                    filtered_matches.append(m[0])

            return filtered_matches

        def imageDistance(matches):

            sumDistance = 0.0

            for match in matches:
                sumDistance += match.distance

            return sumDistance

        from_frame_gray, to_frame_gray = util.gray(from_image), util.gray(to_image)

        # use SIFT feature detector
        if matcher_explicit is None:
            kp1, des1 = self.matcher.detectAndCompute(from_frame_gray, None)
            kp2, des2 = self.matcher.detectAndCompute(to_frame_gray, None)
        else:
            kp1, des1 = matcher_explicit.detectAndCompute(from_frame_gray, None)
            kp2, des2 = matcher_explicit.detectAndCompute(to_frame_gray, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        matches_subset = filter_matches(matches, ratio=0.7)

        distance = imageDistance(matches_subset)
        
        averagePointDistance = distance / float(len(matches_subset)) if len(matches_subset) > 0 else 0

        if len(matches_subset) >= 4:
            src = np.float32([kp1[m.queryIdx].pt for m in matches_subset]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in matches_subset]).reshape(-1, 1, 2)
            H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            H_affine, status_affine = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4)
        else:
            return None
            raise AssertionError('Can’t find enough keypoints.')

        transformation_result = {'perspective': H, 'affine': H_affine}

        return transformation_result



    ####################### FEATURE MATCHING

    # Input: x1, y1 : pixel position(s) in the 'from' frame
    # Output: x0, y0 : pixel position(s) in the 'to' frame (row / col respectively)
    def from_pixel_to_pixel_position(self, x1, y1, H_to_ref):
        #y0, x0 = rasterio.transform.xy(H_to_ref, rows=x1, cols=y1, offset='center')
        x0, y0 = rasterio.transform.xy(H_to_ref, rows=y1, cols=x1, offset='center')
        
        return int(x0), int(y0)


    def transform_image(self, H, frame_from, frame_ref):
        aligned_frame = cv2.warpPerspective(frame_from, H["perspective"], frame_ref.shape[:2][::-1]) # perspective = homography (has bigger degree of freedom) # affine = has lower degree of freedom

        return aligned_frame


    def transform_bbox(self, H, bboxes_from, post_processing=True, frame_size=(2160, 3840)):
        keep_idx = []
        
        if len(bboxes_from) > 0:
            try:
                H_affine = rasterio.Affine(*H['affine'].flatten())
                bboxes_to = [[self.from_pixel_to_pixel_position(p[0], p[1], H_affine) for p in bbs] for bbs in bboxes_from]
                if post_processing:
                    '''# remove all negative bboxes
                    bboxes_to = [bbs for bbs in bboxes_to if np.all([True if b>=0 else False for bb in bbs for b in bb])]
                    # remove all out of 960 x 1080 -> 
                    bboxes_to = [bbs for bbs in bboxes_to if np.all([True if bb[0]<=frame_size[1] and bb[1]<=frame_size[0] else False for bb in bbs])]
                    # check if the 4 corners of a bounding box is of distinct corner points
                    bboxes_to = [bbs for bbs in bboxes_to if len(set([tuple(bb) for bb in bbs]))==4]
                    # check if the cropped rectangle has a normal area > 100
                    #bbs_pred = [bbs for bbs in bbs_pred if Polygon(bbs).area > 100]
                    # Remove all unchecked bbs above
                    bboxes_to = [bbs for bbs in bboxes_to if len(bbs)>0]'''

                    for bbs_idx, bbs in enumerate(bboxes_to):
                        keep_bb = False
                        # remove all negative bboxes
                        if np.all([True if b>=0 else False for bb in bbs for b in bb]):
                            keep_bb = True
                        # remove all out of 3840 x 2160
                        if np.all([True if bb[0]<=frame_size[1] and bb[1]<=frame_size[0] else False for bb in bbs]):
                            keep_bb = True
                        # check if the 4 corners of a bounding box is of distinct corner points
                        if len(set([tuple(bb) for bb in bbs]))==4:
                            keep_bb = True
                        # Remove all unchecked bbs above
                        if len(bbs)>0:
                            keep_bb = True

                        if keep_bb:
                            keep_idx.append(bbs_idx)
                else:
                    keep_idx = list(range(len(bboxes_from)))

            except:
                bboxes_to = []
        else:
            bboxes_to = []
            
        return list(np.array(bboxes_to)[keep_idx]), keep_idx

    

    # --- PREPROCESSING ---
    def process_resize(self, w, h, resize):
        # assert(len(resize) > 0 and len(resize) <= 2)
        if resize > 0:
            scale = resize / max(h, w)
            w_new, h_new = int(round(w*scale)), int(round(h*scale))
        elif resize == -1:
            w_new, h_new = w, h
        else:  # len(resize) == 2:
            w_new, h_new = resize[0], resize[1]

        # Issue warning if resolution is too small or too large.
        '''if max(w_new, h_new) < 160:
            print('Warning: input resolution is very small, results may vary')
        elif max(w_new, h_new) > 2000:
            print('Warning: input resolution is very large, results may vary')'''

        return w_new, h_new


    def frame2tensor(self, frame, device):
        return torch.from_numpy(frame/255.).float()[None, None].to(device)


    def read_image(self, image, device, resize, rotation, resize_float):
        if image is None:
            return None, None, None
        w, h = image.shape[1], image.shape[0]
        w_new, h_new = self.process_resize(w, h, resize)
        scales = (float(w) / float(w_new), float(h) / float(h_new))

        if resize_float:
            image = cv2.resize(image.astype('float32'), (w_new, h_new))
        else:
            image = cv2.resize(image, (w_new, h_new)).astype('float32')

        if rotation != 0:
            image = np.rot90(image, k=rotation)
            if rotation % 2:
                scales = scales[::-1]

        inp = self.frame2tensor(image, device)
        
        return image, inp, scales