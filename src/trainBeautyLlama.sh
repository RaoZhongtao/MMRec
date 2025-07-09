CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model VBPR \
--dataset beauty \
--mode train > ../logs/train_VBPR_beauty_lvlmemb_llama.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model MMGCN \
--dataset beauty \
--mode train > ../logs/train_mmgcn_beauty_lvlmemb_llama.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--model SLMRec \
--dataset beauty \
--mode train > ../logs/train_SLMRec_beauty_lvlmemb_llama.log 2>&1 &