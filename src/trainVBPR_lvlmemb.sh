CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model VBPR \
--dataset beauty \
--extractor qwen \
--moe_num 4 \
--mode train > ../logs/VBPR_efficiency_beauty_qwen.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model VBPR \
--dataset sports \
--extractor default \
--moe_num 4 \
--mode train > ../logs/VBPR_efficiency_sports.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model VBPR \
--dataset toys \
--extractor qwen_image \
--moe_num 4 \
--mode train > ../logs/modality_replacement_VBPR_toys_moe4_qwen_image.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model VBPR \
--dataset toys \
--extractor qwen_text \
--moe_num 4 \
--mode train > ../logs/modality_replacement_VBPR_toys_moe4_qwen_text.log 2>&1 &

