echo "Data preprocessing..."
source ~/miniconda3/etc/profile.d/conda.sh

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

conda activate hoist

if [ -z "$VIDEO2" ]; then
    cmd=$(python -u data_preprocessing.py --video_path1 "$VIDEO1" --fps "$FPS" | tail -n 1)
else
    cmd=$(python -u data_preprocessing.py --video_path1 "$VIDEO1" --video_path2 "$VIDEO2" --fps "$FPS" | tail -n 1)
fi

cd OnePose
echo "OnePose is running..."

conda activate onepose

bash -c "$cmd"
