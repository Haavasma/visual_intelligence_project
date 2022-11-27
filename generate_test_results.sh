#!/bin/bash

image_folder="../data/Norway/test/images/"
res_file="./custom_data/test/resolution.pkl"
yolo_label_folder="./custom_data/results/$1/yolo_labels/"

# module purge
#
# # Load modules
# module load Python/3.10.4-GCCcore-11.3.0
#
# # Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
# export PS1=\$
#
# # activate the virtual environment
# source /cluster/home/haavasma/haavasma_visual_intel/bin/activate


cd /lhome/haavasma/Documents/visual_intelligence/visual_intel

# python create_resolution_pickle.py --output $res_file --images $image_folder
#
# rm -r $yolo_label_folder

# python detect.py --source $image_folder --augment --conf-thres 0.25 --weights "/lhome/haavasma/Documents/visual_intelligence/visual_intel/runs/train/$1/weights/best.pt" --img-size 640 --nosave --save-txt --project $yolo_label_folder 

python generate.py --yolo-labels $yolo_label_folder --resolution-file $res_file --output "./custom_data/results/$1/results.csv" --image-folder $image_folder
