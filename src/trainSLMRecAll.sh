CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model SLMRec \
--dataset beauty \
--extractor default \
--moe_num 4 \
--mode train > ../logs/SLMRec_efficiency_beauty.log.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python main.py \
--model SLMRec \
--dataset toys \
--extractor qwen_image \
--moe_num 4 \
--mode train > ../logs/modality_replacement_SLMRec_toys_moe4_qwen_image.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--model SLMRec \
--dataset sports \
--extractor default \
--moe_num 4 \
--mode train > ../logs/SLMRec_efficiency_sports.log 2>&1 &

