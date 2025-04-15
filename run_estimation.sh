# conda run -n hoist estimate_6Dpose.py --video_path ~/DP/videos/vid7/video7.mp4 --fps 4
# conda run -n onepose 
echo "Data preprocessing..."
video=$1
fps=${2:-2}
cmd=$(conda run -n hoist python estimate_6Dpose.py --video_path $video --fps $fps)

cd OP/OnePose
echo "OnePose is running..."
conda run -n onepose bash -c "$cmd"