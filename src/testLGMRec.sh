# INFO 	Parameters: ['seed', 'dropout', 'reg_weight']=(999, 0.9, 0.0),
# ---learning_rate=0.001


# INFO 	Parameters: ['seed', 'dropout', 'reg_weight']=(999, 0.8, 0.001),
nohup python main.py \
--model LGMRec \
--dataset sports \
--mode test \
--ckpt_dir '../checkpoints/LGMRec_sports_best.pth' > ../logs/test_LGMRec_sports.log 2>&1 &

# INFO 	Parameters: ['seed', 'dropout', 'reg_weight']=(999, 0.8, 0.0),
nohup python main.py \
--model LGMRec \
--dataset toys \
--mode test \
--ckpt_dir '../checkpoints/LGMRec_toys_best.pth' > ../logs/test_LGMRec_toys.log 2>&1 &

# INFO 	Parameters: ['seed', 'dropout', 'reg_weight']=(999, 0.8, 0.001)
nohup python main.py \
--model LGMRec \
--dataset beauty \
--mode test \
--ckpt_dir '../checkpoints/LGMRec_beauty_best.pth' > ../logs/test_LGMRec_beauty.log 2>&1 &