CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model MMGCN \
--dataset beauty \
--mode train > train_mmgcn_beauty.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--model MMGCN \
--dataset clothing \
--mode train > train_mmgcn_clothing.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--model MMGCN \
--dataset sports \
--mode train > train_mmgcn_sports.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model MMGCN \
--dataset toys \
--mode train > train_mmgcn_toys.log 2>&1 &

