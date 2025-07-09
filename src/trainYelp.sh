CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model VBPR \
--dataset yelp \
--mode train > ../logs/train_VBPR_yelp_qwen_lvlmemb.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--model MMGCN \
--dataset yelp \
--mode train > ../logs/train_MMGCN_yelp_qwen_lvlmemb.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--model SLMRec \
--dataset yelp \
--mode train > ../logs/train_SLMRec_yelp_qwen_lvlmemb.log 2>&1 &