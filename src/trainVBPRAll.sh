CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model VBPR \
--dataset beauty \
--mode train > ../logs/train_VBPR_beauty_clip_768.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python main.py \
# --model VBPR \
# --dataset sports \
# --mode train > ../logs/train_VBPR_sports_lvlmemb_768.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python main.py \
# --model VBPR \
# --dataset toys \
# --mode train > ../logs/train_VBPR_toys_with_lvlm_image.log 2>&1 &

