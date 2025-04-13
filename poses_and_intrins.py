import shutil
import numpy as np 
import read_write_model
import os
    
def get_K_from_colmap_camera(cam):
    """
    Convert COLMAP camera parameters into a 3x3 K matrix.
    """
    model = cam.model
    p = cam.params
    if model in ["SIMPLE_PINHOLE"]:       # [f, cx, cy]
        f, cx, cy = p
        return f, f, cx, cy
    elif model in ["PINHOLE"]:            # [fx, fy, cx, cy]
        return p
    elif model in ["SIMPLE_RADIAL", "RADIAL"]:      # [f, cx, cy, k]
        f, cx, cy = p[:3]
        return f, f, cx, cy
    elif model in ["OPENCV"]:             # [fx, fy, cx, cy, ...]
        return p[:4]
    else:
        raise NotImplementedError(f"Camera model {model} is not supported for K extraction.")

def create_poses_and_intrins(model_path, output_dir):
    poses_dir = f'{output_dir}/poses_ba'
    if os.path.exists(poses_dir):
        shutil.rmtree(poses_dir)
    os.makedirs(poses_dir)
    intrin_dir = f'{output_dir}/intrin_ba'
    if os.path.exists(intrin_dir):
        shutil.rmtree(intrin_dir)
    os.makedirs(intrin_dir)
    cameras, images, _ = read_write_model.read_model(model_path, ext='.bin')
    FX, FY, CX, CY = 0, 0, 0, 0
    for image_id, image in images.items():
        image_name = os.path.basename(image.name)
        img = image_name.rindex('.')
        image_name = image_name[:img]

        R_colmap = read_write_model.qvec2rotmat(image.qvec)
        t_colmap = image.tvec
        T_colmap = np.eye(4)
        T_colmap[:3, :3] = R_colmap
        T_colmap[:3, 3] = t_colmap
        pose_txt = os.path.join(poses_dir, f"{image_name}.txt")
        np.savetxt(pose_txt, T_colmap)

        cam = cameras[image.camera_id]
        fx, fy, cx, cy = get_K_from_colmap_camera(cam)
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        FX += fx
        FY += fy
        CX += cx
        CY += cy
        intrin_txt = os.path.join(intrin_dir, f'{image_name}.txt')
        np.savetxt(intrin_txt, K)
    num_images = len(images.items())
    intrinsics_txt = os.path.join(output_dir, f'intrinsics.txt')
    with open(intrinsics_txt, 'w') as intrin_file:
        intrin_file.write(f'fx: {FX / num_images}\n')
        intrin_file.write(f'fy: {FY / num_images}\n')
        intrin_file.write(f'cx: {CX / num_images}\n')
        intrin_file.write(f'cy: {CY / num_images}\n')

def add_pose_intrin(onePose_input_path):
    """
    For images that are not used in reconstruction - copy previous pose and intrin
    """
    masked_path = f"{onePose_input_path}/color"
    poses_path = f"{onePose_input_path}/poses_ba"
    intrins_path = f"{onePose_input_path}/intrin_ba"

    imgs = os.listdir(masked_path)
    imgs = sorted([int(img.split('.')[0]) for img in imgs])
    poses = os.listdir(poses_path)
    poses = [int(pose.split('.')[0]) for pose in poses]
    last_pose = 0
    for i in imgs:
        if i not in poses:
            pose = np.loadtxt(f'{poses_path}/{last_pose}.txt')
            # print(pose)
            np.savetxt(f'{poses_path}/{i}.txt', pose, fmt="%.8f")
            intrin = np.loadtxt(f'{intrins_path}/{last_pose}.txt')
            # print(intrin)
            np.savetxt(f'{intrins_path}/{i}.txt', intrin, fmt="%.8f")
        else:
            last_pose = i

def delete_images(onePose_input_path):
    """
    Delete images which are not in recontruction
    """
    masked_path = f"{onePose_input_path}/color"
    poses_path = f"{onePose_input_path}/poses_ba"
    images_path = f"{onePose_input_path}/color_full"

    imgs = os.listdir(masked_path)
    imgs = sorted([int(img.split('.')[0]) for img in imgs])
    poses = os.listdir(poses_path)
    poses = [int(pose.split('.')[0]) for pose in poses]
    for img in imgs:
        # print(img)
        if img not in poses:
            os.remove(f'{masked_path}/{img}.png')
            os.remove(f'{images_path}/{img}.png')
