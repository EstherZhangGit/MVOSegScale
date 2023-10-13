# %%
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# %%
import os
import sys

import cv2

# import matplotlib.pyplot as plt
# from scipy import signal
import numpy as np

sys.path.append("SegNeXt")

from mmseg.apis import inference_segmentor, init_segmentor

os.environ["SegNeXt_DIR"] = "/home/ros-melodic/wzy/Seg_Scale/SegNeXt/"

config_file = (
    os.environ["SegNeXt_DIR"]
    + "local_configs/segnext/small/segnext.small.1024x1024.city.160k.py"
)
checkpoint_file = (
    os.environ["SegNeXt_DIR"] + "checkpoints/segnext_small_1024x1024_city_160k.pth"
)

model = init_segmentor(config_file, checkpoint_file, device="cuda")

import param
from rescale import ScaleEstimator
from thirdparty.MonocularVO.visual_odometry import PinholeCamera, VisualOdometry


# %%
def main(i_count, img_path):
    # real_scale = None
    seg = True
    # tag = ".test_Rt_SIFT_DynEro7_guassmedian5_"
    # tag = ".abl_AKAZE_"
    # tag = ".forplot_"
    tag = ".robot_car"
    tag_conut = "{}".format(i_count).zfill(3)
    tag = tag + tag_conut
    images_path = img_path
    # print(images_path)
    print(images_path)
    # exit()
    # seq = images_path.split("_")[-1][:2]
    # seq = ""
    # # print(seq)
    # calib = open("../dataset/dataset/sequences/" + str(seq) + "/calib.txt").read()
    # calib = calib.split(" ")
    f = float(param.img_fx)
    cx = float(param.img_cx)
    cy = float(param.img_cy)

    # if len(sys.argv) > 2:
    #     tag = sys.argv[2]
    #    real_scale = np.loadtxt(sys.argv[2])
    res_addr = "../result/" + images_path.split(".")[-2].split("/")[-1] + "_"
    print(res_addr, tag)
    images = open(images_path)
    image_name = images.readline()
    # first line is not pointing to a image, so we read and skip it
    image_names = images.read().split("\n")
    h, w, c = cv2.imread(image_names[0]).shape

    print(f, cx, cy, h, w, c)

    # exit()
    cam = PinholeCamera(w, h, f, f, cx, cy)

    vo = VisualOdometry(cam)
    scale_estimator = ScaleEstimator(absolute_reference=param.camera_h, window_size=5)

    image_id = 0
    path = []
    scales = []
    error = []
    pitchs = []
    motions = []
    feature2ds = []
    feature3ds = []
    move_flags = []
    scale = 1
    path.append([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    # motions.append([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    scales.append(0)
    error.append(100)
    begin_id = 0

    end_id = None
    # img_last = []
    for image_name in image_names:
        if image_id < begin_id:
            image_id += 1
            continue
        if end_id is not None and image_id > end_id:
            break
        if len(image_name) == 0:
            break
        img = cv2.imread(image_name, 0)
        img = cv2.resize(img, (cam.width, cam.height))
        img_bgr = cv2.imread(image_name)
        if seg:
            result = np.array(inference_segmentor(model, img_bgr))
            # base_mask = np.array(np.where(result == 0, 1, 0), dtype=np.uint8)[0]
            dynamic_mask = np.array(np.where(result > 10, 0, 1), dtype=np.uint8)[0]
            # 定义腐蚀核的大小
            kernel_size = 7
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # 对掩膜进行腐蚀操作
            erode_mask = cv2.erode(dynamic_mask, kernel, iterations=1)
            erode_mask
        # img_bgr = cv2.resize(img_bgr, (cam.width, cam.height))
        move_flag = vo.update(img, image_id, None)

        if (not move_flag) and image_id > 1:
            path.append(path[-1])
            motions.append([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
            scales.append(0)
            error.append(0)
            feature2ds.append([])
            feature3ds.append([])
            image_id += 1
            move_flags.append(move_flag)
            continue
        # print(vo.motion_t)
        # print(vo.motion_R)
        if image_id > begin_id:
            # if vo.feature3d is None:
            #     continue
            print(len(vo.feature3d))
            continue
            feature2d = vo.feature3d[:, 0:2].copy()
            feature2d[:, 0] = feature2d[:, 0] * cam.fx / vo.feature3d[:, 2] + cam.cx
            feature2d[:, 1] = feature2d[:, 1] * cam.fx / vo.feature3d[:, 2] + cam.cy
            # feature3ds.append(vo.feature3d.copy())
            # feature2ds.append(feature2d.copy())
            move_flags.append(True)
            # np.savetxt('feature_3d.txt',vo.feature3d)
            print("feature mumber", vo.feature3d.shape)
            if vo.feature3d.shape[0] > param.minimum_feature_for_scale:
                print("Calculate Scale")
                pitch = scale_estimator.initial_estimation(vo.motion_t.reshape(-1))
                pitchs.append(pitch)
                scale, std = scale_estimator.scale_calculation(
                    vo.feature3d, feature2d, img_bgr, None
                )
                # if(np.abs(scale-real_scale[image_id-1])>0.3 and std<0.3):
                # if False:  #  and abs(real_scale[image_id-1]-scale)>0.3):
                #     scale_estimator.check_full_distribution(
                #         vo.feature3d.copy(),
                #         feature2d.copy(),
                #         real_scale[image_id - 1],
                #         img_bgr,
                #     )
                #     scale_estimator.plot_distribution(str(image_id), img_bgr)
                # uncomment to visualize the feature and triangle
                # scale_estimator.visualize_distance(vo.feature3d,feature2d,img_bgr)
                # scale_estimator.visualize(vo.feature3d,feature2d,img_bgr)
                # scale_estimator.visualize_distance(vo.feature3d,ref_warp,img_last)
                # re = reconstructer.visualize(vo.feature3d,feature2d,img_bgr)
                # if re==False:
                #    break
                R, t = vo.get_current_state(scale)
                M = np.zeros((3, 4))
                M[:, 0:3] = R
                M[:, 3] = t.reshape(-1)
                M = M.reshape(-1)
                motion = np.zeros((3, 4))
                motion[:, 0:3] = vo.motion_R
                motion[:, 3] = vo.motion_t.reshape(-1)
                motion = motion.reshape(-1)
                path.append(M)
                motions.append(motion)
                scales.append(scale)
                error.append(std)
            else:
                path.append(path[-1])
                motions.append(motions[-1])
                scales.append(scales[-1])
                error.append(error[-1])
            print("id  ", image_id, " scale ", scale)
        # img_last = img_bgr.copy()
        image_id += 1
    # np.savetxt(res_addr+'features.txt',scale_estimator.all_features)
    # data_to_save = {}
    # # data_to_save["motions"] = motions
    # data_to_save["feature3ds"] = feature3ds
    # # data_to_save["feature2ds"] = feature2ds
    # # data_to_save["move_flags"] = move_flags
    # np.save(res_addr + "result.npy" + tag, data_to_save)

    np.savetxt(res_addr + "path.txt" + tag, path)
    # np.savetxt(res_addr + "motions.txt" + tag, motions)
    np.savetxt(res_addr + "scales.txt" + tag, scales)
    # np.savetxt(res_addr + "error.txt" + tag, error)
    # np.savetxt(res_addr + "pitch.txt" + tag, pitchs)


# %%
if __name__ == "__main__":
    import glob

    path_list = glob.glob("../dataset/201*.txt")
    print(path_list)
    # exit()
    # print(len(path_list))
    # path_list = "../dataset/kitti_image_07.list"
    for img_list in path_list:
        for i in range(5):
            main(i, img_list)

# %%
