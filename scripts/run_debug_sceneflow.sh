# train
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 openstereo/main.py --config ./configs/0_debugs/igev_sceneflow.yaml --scope train >> ./logs/IGEV_sceneflow_0123.txt 2>&1 &

# val
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#python openstereo/main.py --config ./configs/igev/igev_sceneflow.yaml --scope val
