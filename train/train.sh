YAML="i2v-train"
TORCH_CUDNN_SDPA_ENABLED=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
-c configs/${YAML} \
--use-cluster 0 \
--num-gpus 8 \
