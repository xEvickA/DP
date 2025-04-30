import argparse
import os
import numpy as np

def angle(x, y):
    x = x.ravel()
    y = y.ravel()

    cos_theta = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_theta))

def rotation_angle(R1, R2):
    return np.rad2deg(np.real(np.arccos((np.trace(R1.T @ R2) - 1) / 2 + 0.0j)))

def R_t_errors(onepose_poses_path, gt_poses_path):
    onepose_poses = os.listdir(onepose_poses_path)
    gt_poses = os.listdir(gt_poses_path)
    
    R_errors = []
    t_errors = []
    for pose1 in onepose_poses:
        for pose2 in onepose_poses:
            if pose1 is not pose2 and pose1 in gt_poses and pose2 in gt_poses:
                onepose_pose1 = np.loadtxt(f'{onepose_poses_path}/{pose1}')
                R_onepose1 = onepose_pose1[:3, :3]
                t_onepose1 = onepose_pose1[:3, 3]

                onepose_pose2 = np.loadtxt(f'{onepose_poses_path}/{pose2}')
                R_onepose2 = onepose_pose2[:3, :3]
                t_onepose2 = onepose_pose2[:3, 3]

                R_relative_onepose = np.dot(R_onepose2, R_onepose1.T)
                t_relative_onepose = t_onepose2 - np.dot(R_relative_onepose, t_onepose1)

                gt_pose1 = np.loadtxt(f'{gt_poses_path}/{pose1}')
                R_gt1 = gt_pose1[:3, :3]
                t_gt1 = gt_pose1[:3, 3]

                gt_pose2 = np.loadtxt(f'{gt_poses_path}/{pose2}')
                R_gt2 = gt_pose2[:3, :3]
                t_gt2 = gt_pose2[:3, 3]

                R_relative_gt = np.dot(R_gt2, R_gt1.T)
                t_relative_gt = t_gt2 - np.dot(R_relative_gt, t_gt1)   
                
                t_err = angle(t_relative_onepose, t_relative_gt)
                R_err = rotation_angle(R_relative_onepose, R_relative_gt)

                R_errors.append(R_err)
                if not np.isnan(t_err):
                    t_errors.append(t_err)
    return R_errors, t_errors

onepose_pose_folder = ... # .../poses
gt_pose_folder = ... # .../poses_ba

R_errors, t_errors = R_t_errors(onepose_pose_folder, gt_pose_folder)

r_errs = np.array([r for r in R_errors])
r_errs[np.isnan(r_errs)] = 180
r_res = np.array([np.sum(r_errs < t) / len(r_errs) for t in range(1, 11)])
r_auc_10 = np.mean(r_res)
r_avg = np.mean(r_errs)
r_med = np.median(r_errs)
print(f"Nan in R {np.isnan(r_errs).sum()} from {len(r_errs)}")
print(f"R AUC10: {r_auc_10}")
print(f'R avg: {r_avg}')
print(f'R med: {r_med}')
t_errs = np.array([t for t in t_errors])
print(f"Nan in t {np.isnan(t_errs).sum()} from {len(t_errs)}")
t_errs[np.isnan(t_errs)] = np.random.uniform(-1, 1)
t_res = np.array([np.sum(t_errs < t) / len(t_errs) for t in range(1, 11)])
t_auc_10 = np.mean(t_res)
t_avg = np.mean(t_errs)
t_med = np.median(t_errs)
print(f't auc {t_auc_10}')
print(f't avg: {t_avg}')
print(f't med: {t_med}')
