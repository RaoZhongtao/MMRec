CUDA_VISIBLE_DEVICES=3 nohup python main.py \
                        --model LGMRec \
                        --dataset toys \
                        --mode train > ../logs/train_LGMRec_toys.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
                        --model LGMRec \
                        --dataset sports \
                        --mode train > ../logs/train_LGMRec_sports.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python main.py \
                        --model LGMRec \
                        --dataset beauty \
                        --mode train > ../logs/train_LGMRec_beauty.log 2>&1 &