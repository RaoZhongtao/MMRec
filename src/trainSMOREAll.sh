CUDA_VISIBLE_DEVICES=0 nohup python main.py \
                        --model SMORE \
                        --dataset beauty \
                        --extractor qwen \
                        --moe_num 4 \
                        --mode train > ../logs/SMORE_efficiency_beauty_qwen.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
                        --model SMORE \
                        --dataset sports \
                        --extractor qwen \
                        --moe_num 4 \
                        --mode train > ../logs/SMORE_efficiency_sports_qwen.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python main.py \
#                         --model SMORE \
#                         --dataset toys \
#                         --extractor default \
#                         --moe_num 4 \
#                         --mode train > ../logs/SMORE_efficiency_toys.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
                        --model SMORE \
                        --dataset toys \
                        --extractor qwen_text \
                        --moe_num 4 \
                        --mode train > ../logs/modality_replacement_SMORE_toys_moe4_qwen_text.log 2>&1 &