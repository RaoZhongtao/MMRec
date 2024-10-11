CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model BM3 \
--dataset beauty \
--mode train > train_BM3_beauty.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python main.py \
--model BM3 \
--dataset clothing \
--mode train > train_BM3_clothing.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python main.py \
--model BM3 \
--dataset sports \
--mode train > train_BM3_sports.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main.py \
--model BM3 \
--dataset toys \
--mode train > train_BM3_toys.log 2>&1 &