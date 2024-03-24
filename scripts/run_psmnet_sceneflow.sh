# train
export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 openstereo/main.py --config ./configs/0_debugs/PSMNet_sceneflow.yaml --scope train --master_port 12570 >> ./logs/PSMnet_sceneflow_G4_4567.txt 2>&1 &

# python3 openstereo/main.py --config ./configs/0_debugs/PSMNet_sceneflow.yaml --scope train >> ./logs/PSMnet_sceneflow_4567.txt 2>&1 &
# val
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#python openstereo/main.py --config ./configs/igev/igev_sceneflow.yaml --scope val
