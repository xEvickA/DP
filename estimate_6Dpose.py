import cv2
import os
import argparse
import pycolmap
from pycolmap import extract_features, match_exhaustive, match_sequential
import shutil
import poses_and_intrins
import numpy as np
from read_write_model import read_points3D_binary
from termcolor import colored
import warnings


def parse_video(inpath, outpath, images_path, fps):
    count = 0
    image_number = 0
    folder = 0

    vidcap = cv2.VideoCapture(inpath)
    success,image = vidcap.read()
    
    interval = int(vidcap.get(cv2.CAP_PROP_FPS) / fps) 
    
    while success:
        success, image = vidcap.read()
        if not success:
            break

        if count % interval == 0:    
            # in case CUDA out of memory deacrease the number
            if image_number % 18 == 0:
                outpath = outpath[:-1] + str(folder)
                os.makedirs(outpath)
                folder += 1
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite( outpath + f'/{image_number}.png', image)     
            cv2.imwrite(images_path + f'/{image_number}.png', image)
            image_number += 1
        count = count + 1
    return folder - 1 

def apply_mask(images_path, masks_path, output_path):
    for image_path in os.listdir(images_path):
        image = cv2.imread(f'{images_path}/{image_path}', cv2.IMREAD_UNCHANGED)
        mask_path = f'{masks_path}/{image_path}.png'
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite(f'{output_path}/{image_path}', masked_image)

def create_object_center(points3D_path, output_path):
    points3D = read_points3D_binary(f"{points3D_path}/points3D.bin")
    points = np.array([p.xyz for p in points3D.values()])
    mean_point = points.mean(axis=0)

    np.savetxt(f'{output_path}/center.txt', mean_point)


if __name__=="__main__":
    fps = 1
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--video_path', required=True)
    argparser.add_argument('--fps', required=False)
    arguments = argparser.parse_args()

    video = arguments.video_path
    video_name = video.split('/')[-1].split(".")[0]
    if arguments.fps:
        fps = int(arguments.fps)

    outpath = f'./parsed_videos/{video_name}/{video_name}-0' # /{video_name}-0
    if os.path.exists(outpath):
        shutil.rmtree(outpath[:outpath.rindex('/')])
    onePose_input_parent = f'{os.getcwd()}/6D_pose_data/{video_name}'
    if os.path.exists(onePose_input_parent):
        shutil.rmtree(onePose_input_parent)
    onePose_input = f'{onePose_input_parent}/{video_name}'
    images_path = f'{onePose_input}/color_full'
    os.makedirs(images_path)
    
    folders_num = parse_video(video, outpath, images_path, fps)
    image_count = len(os.listdir(images_path))

    outpath = outpath[:outpath.rindex('/')][1:]
    os.system(f"cd HOISTFormer && CUDA_VISIBLE_DEVICES=0 python demo.py --video_frames_path {os.getcwd() + outpath}")
    masks_path = f'{os.getcwd()}/HOISTFormer/masks/{video_name}'
    
    masked_images_path = f'{os.getcwd()}/6D_pose_data/{video_name}/{video_name}/color'
    os.makedirs(masked_images_path)
    apply_mask(images_path, masks_path, masked_images_path)

    model_output_path = f"{os.getcwd()}/6D_pose_model/{video_name}"
    if os.path.exists(model_output_path):
        shutil.rmtree(model_output_path)
    sparse_path = f'{model_output_path}/outputs_superpoint_superglue/sfm_ws'
    os.makedirs(sparse_path)

    # create sparse model
    database_path = f'{sparse_path}/database.db'
    extract_features(database_path=database_path,
                     image_path=masked_images_path)
    match_exhaustive(database_path)
    reconstruction = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=masked_images_path,
        output_path=sparse_path
    )
    model_folders = sorted(os.listdir(sparse_path))[:-1]
    largest_model = max(
        model_folders,
        key=lambda f: len(pycolmap.Reconstruction(os.path.join(sparse_path, f)).images)
    )
    model_path = f'{sparse_path}/model'
    # incremental_mapping creates more models, save just the largest
    os.rename(f'{sparse_path}/{largest_model}', model_path)

    poses_and_intrins.create_poses_and_intrins(model_path, onePose_input)

    # delete images which wasn't used to create model
    poses_and_intrins.delete_images(onePose_input)
    image_count_after = len(os.listdir(images_path))
    create_object_center(model_path, onePose_input_parent)

    results_path = f"{os.getcwd()}/6D_results/{video_name}"

    if image_count != image_count_after:
        warnings.warn(f"Recontruction contains {image_count_after}/{image_count} images.", category=UserWarning)

    print(colored("RUN COMMANDS IN ONEPOSE ENV:", "light_magenta"))
    print(f'python run.py +preprocess=sfm_spp_spg_own.yaml dataset.data_dir="{onePose_input_parent} {video_name}" dataset.outputs_dir={model_output_path} && ')
    print(f'python inference.py +experiment=test_own.yaml input.data_dirs={onePose_input} input.sfm_model_dirs={model_output_path} output.vis_dir={results_path}/vis output.eval_dir={results_path}/eval demo_root={results_path}/demo')

