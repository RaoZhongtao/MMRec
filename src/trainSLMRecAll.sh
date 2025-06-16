CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model SLMRec \
--dataset beauty \
--mode train > ../logs/train_SLMRec_beauty_lvlmemb.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python main.py \
# --model SLMRec \
# --dataset toys \
# --mode train > ../logs/train_SLMRec_toys_lvlmemb.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--model SLMRec \
--dataset sports \
--mode train > ../logs/train_SLMRec_sports_lvlmemb.log 2>&1 &

