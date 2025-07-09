# CUDA_VISIBLE_DEVICES=0 nohup python main.py \
# --model BM3 \
# --dataset beauty \
# --mode train > ../logs/train_BM3_beauty_128.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python main.py \
# --model BM3 \
# --dataset toys \
# --mode train > ../logs/train_BM3_toys_lvlmemb.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--model BM3 \
--dataset sports \
--mode train > ../logs/train_BM3_sports_lvlmemb_0707.log 2>&1 &