CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--model SLMRec \
--dataset douyin \
--mode train > ../logs/train_SLMRec_douyin_128.log 2>&1 &