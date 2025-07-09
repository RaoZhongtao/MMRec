CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--model VBPR \
--dataset douyin \
--mode train > ../logs/train_VBPR_douyin_128.log 2>&1 &