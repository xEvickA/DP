# conda run -n hoist estimate_6Dpose.py --video_path ~/DP/videos/vid7/video7.mp4 --fps 4
# conda run -n onepose 
echo "Data preprocessing..."
FPS=2
for ARG in "$@"
do
  case $ARG in
    --video1=*)
      VIDEO1="${ARG#*=}"
      shift
      ;;
    --video2=*)
      VIDEO2="${ARG#*=}"
      shift
      ;;
    --fps=*)
      FPS="${ARG#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $ARG"
      exit 1
      ;;
  esac
done

if [ -z "$VIDEO1" ]; then
  echo "Error: --video1 is required."
  exit 1
fi

if [ -z "$VIDEO2" ]; then
    cmd=$(conda run -n hoist python data_preprocessing.py --video_path1 $VIDEO1 --fps $FPS | tail -n 1)
else
    cmd=$(conda run -n hoist python data_preprocessing.py --video_path1 $VIDEO1 --video_path2 $VIDEO2 --fps $FPS | tail -n 1)
fi

OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
unset LD_LIBRARY_PATH
cd OnePose
echo "OnePose is running..."
conda run -n onepose bash -c "$cmd"
export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"