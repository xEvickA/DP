import cv2
import os
import argparse
import pycolmap
from pycolmap import extract_features, match_exhaustive, match_sequential
import shutil
import poses_and_intrins
import numpy as np
from read_write_model import read_points3D_binary
import warnings
warnings.simplefilter("always") 

def parse_video(inpath, outpath, images_path, fps, img_count=0):
    count = 0
    image_number = 0
    folder = 0

    vidcap = cv2.VideoCapture(inpath)
    success,image = vidcap.read()
    
    interval = int(vidcap.get(cv2.CAP_PROP_FPS) / fps) 
    dash_index = outpath.rindex("-") + 1
    while success:
        success, image = vidcap.read()
        if not success:
            break

        if count % interval == 0:    
            # in case CUDA out of memory deacrease the number
            if image_number % 5 == 0:
                outpath = outpath[:dash_index] + str(folder)
                os.makedirs(outpath)
                folder += 1
            cv2.imwrite( outpath + f'/{image_number + img_count}.png', image)     
            cv2.imwrite(images_path + f'/{image_number + img_count}.png', image)
            image_number += 1
        count = count + 1
    return image_number

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

def get_video_name(video):
    return video.split('/')[-1].split(".")[0]

def preprocessing(video, img_count=0):
    video_name = get_video_name(video)
    outpath = f'./parsed_videos/{video_name}/{video_name}-0' # /{video_name}-0
    if os.path.exists(outpath):
        shutil.rmtree(outpath[:outpath.rindex('/')])
    onePose_input_parent = f'{os.getcwd()}/6D_pose_data/{video_name}'
    if os.path.exists(onePose_input_parent):
        shutil.rmtree(onePose_input_parent)
    onePose_input = f'{onePose_input_parent}/{video_name}'
    images_path = f'{onePose_input}/color_full'
    os.makedirs(images_path)
    
    image_count = parse_video(video, outpath, images_path, fps, img_count)

    outpath = outpath[:outpath.rindex('/')][1:]
    os.system(f"cd HOISTFormer && CUDA_VISIBLE_DEVICES=0 python demo.py --video_frames_path {os.getcwd() + outpath}")
    masks_path = f'{os.getcwd()}/HOISTFormer/masks/{video_name}'
    
    masked_images_path = f'{os.getcwd()}/6D_pose_data/{video_name}/{video_name}/color'
    os.makedirs(masked_images_path)
    apply_mask(images_path, masks_path, masked_images_path)
    return onePose_input, image_count

def make_reconstruction(video_name, onePose_input, merged=False):
    model_output_path = f"{os.getcwd()}/6D_pose_model/{video_name}"
    if os.path.exists(model_output_path):
        shutil.rmtree(model_output_path)
    sparse_path = f'{model_output_path}/outputs_superpoint_superglue/sfm_ws'
    os.makedirs(sparse_path)

    # create sparse model
    database_path = f'{sparse_path}/database.db'
    extract_features(database_path=database_path,
                     image_path=f'{onePose_input}/color')
    match_exhaustive(database_path)
    reconstruction = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=f'{onePose_input}/color',
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
    poses_and_intrins.delete_images(onePose_input, merged)
    onePose_input_parent = onePose_input[:onePose_input.rindex('/')]
    create_object_center(model_path, onePose_input_parent)
    return model_output_path

def copy_files(source, destination):
    os.makedirs(destination, exist_ok=True)
    for filename in os.listdir(source):
        source_path = os.path.join(source, filename)
        destination_path = os.path.join(destination, filename)

        # Check if it's a file (not a subdirectory)
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)

if __name__=="__main__":
    fps = 1
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--video_path1', required=True)
    argparser.add_argument('--video_path2', required=False)
    argparser.add_argument('--fps', required=False)
    arguments = argparser.parse_args()

    if arguments.fps:
        fps = int(arguments.fps)

    video1 = arguments.video_path1
    video_name1 = get_video_name(video1)
    onePose_input1, num_imgs = preprocessing(video1)
    model_output_path = make_reconstruction(video_name1, onePose_input1)
    onePose_input_parent1 = onePose_input1[:onePose_input1.rindex('/')]

    if arguments.video_path2:
        video2 = arguments.video_path2
        video_name2 = get_video_name(video2)
        onePose_input2, _ = preprocessing(video2, num_imgs)
        onePose_input_parent2 = onePose_input2[:onePose_input2.rindex('/')]
        copy_files(onePose_input1, onePose_input2)
        copy_files(onePose_input_parent1, onePose_input_parent2)

        merged_folder = f'{os.getcwd()}/6D_pose_data/{video_name1}+{video_name2}/{video_name1}+{video_name2}'
        if os.path.exists(merged_folder):
            shutil.rmtree(merged_folder)
        copy_files(f'{onePose_input1}/color', f'{merged_folder}/color')
        copy_files(f'{onePose_input2}/color', f'{merged_folder}/color')
        merged_model_path = make_reconstruction(f'{video_name1}+{video_name2}', merged_folder, True)
        results_path = f"{os.getcwd()}/6D_results/{video_name2}"
        print(f'python run.py +preprocess=sfm_spp_spg_own.yaml dataset.data_dir="{onePose_input_parent1} {get_video_name(video1)}" dataset.outputs_dir={model_output_path} && python inference.py +experiment=test_own.yaml input.data_dirs={onePose_input2} input.sfm_model_dirs={model_output_path} output.vis_dir={results_path}/vis output.eval_dir={results_path}/eval demo_root={results_path}/demo +fps={fps}')
    
    else:
        results_path = f"{os.getcwd()}/6D_results/{video_name1}"
        print(f'python run.py +preprocess=sfm_spp_spg_own.yaml dataset.data_dir="{onePose_input_parent1} {get_video_name(video1)}" dataset.outputs_dir={model_output_path} && python inference.py +experiment=test_own.yaml input.data_dirs={onePose_input1} input.sfm_model_dirs={model_output_path} output.vis_dir={results_path}/vis output.eval_dir={results_path}/eval demo_root={results_path}/demo +fps={fps}')
