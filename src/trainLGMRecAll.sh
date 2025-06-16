CUDA_VISIBLE_DEVICES=0 nohup python main.py \
                        --model LGMRec \
                        --dataset toys \
                        --mode train > ../logs/train_LGMRec_toys_lvlmemb.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python main.py \
#                         --model LGMRec \
#                         --dataset sports \
#                         --mode train > ../logs/train_LGMRec_sports_128.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python main.py \
#                         --model LGMRec \
#                         --dataset beauty \
#                         --mode train > ../logs/train_LGMRec_beauty_128.log 2>&1 &