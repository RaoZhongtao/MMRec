CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model MGCN \
--dataset beauty \
--extractor default \
--moe_num 4 \
--mode train > ../logs/train_MGCN_beauty_128.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--model MGCN \
--dataset beauty \
--extractor llama \
--moe_num 4 \
--mode train > ../logs/train_MGCN_beauty_lvlmemb_llama_768.log 2>&1 &