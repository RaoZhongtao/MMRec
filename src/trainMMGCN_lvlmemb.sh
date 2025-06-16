CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model MMGCN \
--dataset beauty \
--mode train > ../logs/train_mmgcn_beauty_lvlmemb.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--model MMGCN \
--dataset sports \
--mode train > ../logs/train_mmgcn_sports_lvlmemb.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--model MMGCN \
--dataset toys \
--mode train > ../logs/train_mmgcn_toys_lvlmemb.log 2>&1 &

