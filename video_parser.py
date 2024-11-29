import cv2
import os
import argparse

def create_folder(name):
    try:
        os.makedirs(name)
        print(f'Folder {name} created.')
    except Exception as e:
        print(f'Folder {name} exists.')
        pass

def parse_video(inpath, outpath, fps):
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
            if image_number % 18 == 0:
                print(outpath)
                outpath = outpath[:-1] + str(folder)
                create_folder(outpath)
                folder += 1
            print (f'Image saved in {outpath}/frame{image_number}.jpg')
            cv2.imwrite( outpath + f'/frame{image_number}.jpg', image)     # save frame as JPEG file
            image_number += 1
        count = count + 1
    return folder - 1 


if __name__=="__main__":
    fps = 1
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--video_path', required=True)
    argparser.add_argument('--fps', required=False)
    arguments = argparser.parse_args()

    inpath = arguments.video_path
    if arguments.fps:
        fps = int(arguments.fps)
    video = inpath + '/' + os.listdir(arguments.video_path)[0]
    outpath = f'./parsed_videos/{os.listdir(inpath)[0].split(".")[0]}/{os.listdir(inpath)[0].split(".")[0]}-0'
    
    folders_num = parse_video(video, outpath, fps)

    outpath = outpath[:outpath.rindex('/')]

    os.system(f"cd HOISTFormer && CUDA_VISIBLE_DEVICES=0 python demo.py --video_frames_path {os.getcwd() + outpath[1:]}")
