# train
export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 openstereo/main.py --config ./configs/igev_agg/igev_agg_sceneflow_sift_1.yaml --scope train >> ./logs/igev_agg_sceneflow_sift_1_G3_4567.txt 2>&1 &


# >> ./logs/IGEV_sceneflow_0123.txt 2>&1 &

# val
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#python openstereo/main.py --config ./configs/igev/igev_sceneflow.yaml --scope val
