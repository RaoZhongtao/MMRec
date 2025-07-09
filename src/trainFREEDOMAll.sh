# CUDA_VISIBLE_DEVICES=0 nohup python main.py \
# --model FREEDOM \
# --dataset beauty \
# --mode train > ../logs/train_freedom_beauty_128.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python main.py \
# --model FREEDOM \
# --dataset toys \
# --mode train > ../logs/train_freedom_toys_lvlmemb.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python main.py \
# --model FREEDOM \
# --dataset sports \
# --mode train > ../logs/train_freedom_sports_lvlmemb_0_shot.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--model FREEDOM \
--dataset sports \
--mode train > ../logs/train_freedom_sports_lvlmemb_0708.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 nohup python main.py \
# --model FREEDOM \
# --dataset toys \
# --mode train > ../logs/train_freedom_toys_128.log 2>&1 &


# python main.py --model FREEDOM --dataset toys --mode train > logs/train_freedom_toys.log