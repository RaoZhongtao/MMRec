CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model VBPR \
--dataset toys \
--mode train > ../logs/train_VBPR_toys_lvlmemb_llama.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model MMGCN \
--dataset toys \
--mode train > ../logs/train_mmgcn_toys_lvlmemb_llama.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--model SLMRec \
--dataset toys \
--mode train > ../logs/train_SLMRec_toys_lvlmemb_llama.log 2>&1 &