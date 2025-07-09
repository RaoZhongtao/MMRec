# CUDA_VISIBLE_DEVICES=2 nohup python main.py \
#                         --model SMORE \
#                         --dataset toys \
#                         --mode train > ../logs/train_SMORE_toys_lvlmemb.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python main.py \
#                         --model SMORE \
#                         --dataset sports \
#                         --mode train > ../logs/train_SMORE_sports_lvlmemb_768.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
                        --model SMORE \
                        --dataset beauty \
                        --mode train > ../logs/train_SMORE_beauty_clip_768.log 2>&1 &