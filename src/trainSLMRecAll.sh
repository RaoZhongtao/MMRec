CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--model SLMRec \
--dataset beauty \
--mode train > ../logs/train_SLMRec_beauty_clip_768.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python main.py \
# --model SLMRec \
# --dataset toys \
# --mode train > ../logs/train_SLMRec_toys_with_lvlm_image.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python main.py \
# --model SLMRec \
# --dataset sports \
# --mode train > ../logs/train_SLMRec_sports_lvlmemb_768.log 2>&1 &

