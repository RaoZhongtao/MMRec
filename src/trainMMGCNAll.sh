CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model MMGCN \
--dataset beauty \
--extractor llama \
--moe_num 4 \
--mode train > ../logs/MMGCN_efficiency_beauty_llama.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model MMGCN \
--dataset sports \
--extractor llama \
--moe_num 4 \
--mode train > ../logs/MMGCN_efficiency_sports_llama.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--model MMGCN \
--dataset toys \
--extractor llama \
--moe_num 4 \
--mode train > ../logs/MMGCN_efficiency_toys_llama.log 2>&1 &

