CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model mgcn \
--dataset beauty \
--mode train > train_mgcn_beauty.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python main.py \
--model mgcn \
--dataset clothing \
--mode train > train_mgcn_clothing.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python main.py \
--model mgcn \
--dataset sports \
--mode train > train_mgcn_sports.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main.py \
--model mgcn \
--dataset toys \
--mode train > train_mgcn_toys.log 2>&1 &