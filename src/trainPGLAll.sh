CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--model PGL \
--dataset beauty \
--extractor llama \
--moe_num 4 \
--mode train > ../logs/train_PGL_beauty_lvlmemb_llama_moe4_768.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model PGL \
--dataset sports \
--extractor llama \
--moe_num 4 \
--mode train > ../logs/train_PGL_sports_lvlmemb_llama_moe4_768.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python main.py \
--model PGL \
--dataset toys \
--extractor qwen_text \
--moe_num 4 \
--mode train > ../logs/modality_replacement_PGL_toys_moe4_qwen_text.log 2>&1 &