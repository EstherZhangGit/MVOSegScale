import argparse
import sys

import cv2
import numpy as np
import torch
from thirdparty.SuperGlue.models.matching import Matching

parser = argparse.ArgumentParser(
    description="SuperGlue demo", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--superglue",
    choices={"indoor", "outdoor"},
    default="indoor",
    help="SuperGlue weights",
)
parser.add_argument(
    "--max_keypoints",
    type=int,
    default=-1,
    help="Maximum number of keypoints detected by Superpoint"
    " ('-1' keeps all keypoints)",
)
parser.add_argument(
    "--keypoint_threshold",
    type=float,
    default=0.004,
    help="SuperPoint keypoint detector confidence threshold",
)
parser.add_argument(
    "--nms_radius",
    type=int,
    default=3,
    help="SuperPoint Non Maximum Suppression (NMS) radius" " (Must be positive)",
)
parser.add_argument(
    "--sinkhorn_iterations",
    type=int,
    default=20,
    help="Number of Sinkhorn iterations performed by SuperGlue",
)
parser.add_argument(
    "--match_threshold", type=float, default=0.2, help="SuperGlue match threshold"
)

parser.add_argument(
    "--show_keypoints", action="store_true", help="Show the detected keypoints"
)
parser.add_argument(
    "--no_display",
    action="store_true",
    help="Do not display images to screen. Useful if running remotely",
)
parser.add_argument(
    "--force_cpu", action="store_true", help="Force pytorch to run in CPU mode."
)

opt = parser.parse_args()
print(opt)

device = "cuda" if torch.cuda.is_available() and not opt.force_cpu else "cpu"
print('Running inference on device "{}"'.format(device))
config = {
    "superpoint": {
        "nms_radius": opt.nms_radius,
        "keypoint_threshold": opt.keypoint_threshold,
        "max_keypoints": opt.max_keypoints,
    },
    "superglue": {
        "weights": opt.superglue,
        "sinkhorn_iterations": opt.sinkhorn_iterations,
        "match_threshold": opt.match_threshold,
    },
}
matching = Matching(config).eval().to(device)
keys = ["keypoints", "scores", "descriptors"]

lk_params = dict(
    winSize=(21, 21),
    # maxLevel = 3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class FeatureDetector:
    def __init__(self, threshold=0.000007, bucket_size=30, density=2):
        # self.akaze = cv2.AKAZE_create(threshold=threshold)
        # self.akaze = cv2.AKAZE_create()
        # self.akaze = cv2.BRISK_create(thresh=15)
        # self.akaze = cv2.ORB_create(nfeatures=3000,
        #                             scaleFactor=1.1,
        #                             nlevels=20)
        self.akaze = cv2.SIFT_create(nOctaveLayers=8)
        # self.akaze = cv2.GFTTDetector_create()
        self.bucket_size = bucket_size
        self.density = density

    def detect(self, image, mask=None):
        # frame_tensor = frame2tensor(frame, device)
        # last_data = matching.superpoint({"image": frame_tensor})
        # last_data = {k + "0": last_data[k] for k in keys}
        # last_data["image0"] = frame_tensor
        # last_frame = frame
        # last_image_id = 0
        # frame_tensor = frame2tensor(image, device)
        # last_data = matching.superpoint({"image": frame_tensor})
        # kp_akaze = last_data["keypoints"][0].cpu().numpy()
        # indices = np.argwhere(mask == 1)
        kp_akaze = self.akaze.detect(image, mask)
        px_cur = np.array([x.pt for x in kp_akaze], dtype=np.float32)
        # return kp_akaze
        return bucket(px_cur, self.bucket_size, self.density)

    def Bi_detect(self, image, mask=None):
        kp_GFTT = self.bi_dete.detect(image, mask)
        px_cur = np.array([x.pt for x in kp_GFTT], dtype=np.float32)
        return bucket(px_cur, self.bucket_size, self.density)

    def bucket(self, features, bucket_size=30, density=2):
        u_max, v_max = np.max(features, 0)
        u_min, v_min = np.min(features, 0)
        print(u_min, v_min, u_max, v_max)
        bucket_x = 1 + (u_max) // bucket_size
        bucket_y = 1 + (v_max) // bucket_size
        print(bucket_y)
        bucket = []
        for i in range(int(bucket_y)):
            buc = []
            for j in range(int(bucket_x)):
                buc.append([])
            bucket.append(buc)
        print(len(bucket))
        i_feature = 0
        for feature in features:
            u = int(feature[0]) // bucket_size
            v = int(feature[1]) // bucket_size
            bucket[v][u].append(i_feature)
            i_feature += 1

        # print(bucket)
        new_feature = []
        for i in range(int(bucket_y)):
            for j in range(int(bucket_x)):
                feature_id = bucket[i][j]
                np.random.shuffle(feature_id)
                for k in range(min(density, len(feature_id))):
                    new_feature.append(features[feature_id[k]])

        return np.array(new_feature)


def akaze(image):
    akaze = cv2.AKAZE_create(threshold=0.000007)
    kp_akaze = akaze.detect(image, None)
    px_cur = np.array([x.pt for x in kp_akaze], dtype=np.float32)
    return px_cur


def fast(image):
    detector = cv2.FastFeatureDetector_create(10, nonmaxSuppression=True)
    detector.detect(image, None)
    # img_akaze = cv2.drawKeypoints(image,kp_akaze,image,color=(255,0,0))
    # cv2.imshow('AKAZE',img_akaze)
    # cv2.waitKey(0)


# ref libviso
def bucket(features, bucket_size=30, density=2):
    u_max, v_max = np.max(features, 0)
    u_min, v_min = np.min(features, 0)
    # print(u_min, v_min, u_max, v_max)
    bucket_x = 1 + (u_max) // bucket_size
    bucket_y = 1 + (v_max) // bucket_size
    # print(bucket_y)
    bucket = []
    for i in range(int(bucket_y)):
        buc = []
        for j in range(int(bucket_x)):
            buc.append([])
        bucket.append(buc)
    print(len(bucket))
    i_feature = 0
    for feature in features:
        u = int(feature[0]) // bucket_size
        v = int(feature[1]) // bucket_size
        bucket[v][u].append(i_feature)
        i_feature += 1

    # print(bucket)
    new_feature = []
    for i in range(int(bucket_y)):
        for j in range(int(bucket_x)):
            feature_id = bucket[i][j]
            np.random.shuffle(feature_id)
            for k in range(min(density, len(feature_id))):
                new_feature.append(features[feature_id[k]])

    return np.array(new_feature)


def motion_estimarion(feature_ref, feature_target, fx, cx, cy):
    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = camera_matrix[1, 1] = fx
    camera_matrix[0, 2] = cx
    camera_matrix[1, 2] = cy

    E, mask = cv2.findEssentialMat(
        feature_ref,
        feature_target,
        cameraMatrix=camera_matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    _, R, t, mask, points_3d = cv2.recoverPose(
        E, feature_ref, feature_target, cameraMatrix=camera_matrix, distanceThresh=100
    )
    print(t)
    mask_bool = np.array(mask > 0).reshape(-1)
    points_3d[:, mask_bool].T


def feature_tracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(
        image_ref, image_cur, px_ref, None, **lk_params
    )  # shape: [k,2] [k,1] [k,1]
    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


def feature_detection(image, image_show):
    features = akaze(image)
    fast(image)
    features = bucket(features)
    print(features.shape)
    draw_feature(image_show, features)
    return features


def draw_feature(img, feature, color=(255, 255, 0)):
    for i in range(feature.shape[0]):
        cv2.circle(img, (int(feature[i, 0]), int(feature[i, 1])), 3, color, -1)


def main():
    image_name_file = open(sys.argv[1])
    image_name_file.readline()
    image_names = image_name_file.read().split("\n")
    begin_id = 312
    image_id = 0
    image_last = None
    detector = FeatureDetector()
    for image_name in image_names:
        if image_id < begin_id:
            image_id += 1
            continue
        print(image_name)
        image = cv2.imread(image_name)
        image_show = image.copy()
        # features = feature_detection(image,image_show)
        features = detector.detect(image)
        if image_last is not None:
            feature_ref, feature_target = feature_tracking(image, image_last, features)
            draw_feature(image_show, feature_ref, (0, 255, 255))
            draw_feature(image_last, feature_target, (0, 255, 255))
            motion_estimarion(feature_ref, feature_target, 716, 607, 189)
            print(len(feature_ref))
            cv2.imshow("image_last", image_last)
        cv2.imshow("image", image_show)
        cv2.waitKey(0)
        image_last = image.copy()
        image_id += 1


if __name__ == "__main__":
    main()
