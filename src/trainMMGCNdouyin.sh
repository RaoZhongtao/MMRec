CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model MMGCN \
--dataset douyin \
--mode train > ../logs/train_MMGCN_douyin_128.log 2>&1 &