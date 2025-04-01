# INFO 	Parameters: ['seed', 'n_layers', 'reg_weight', 'dropout']=(999, 1, 0.01, 0.5)
# ---learning_rate=0.001
nohup python main.py \
--model BM3 \
--dataset clothing \
--mode test \
--ckpt_dir '../checkpoints/BM3_clothing_best.pth' > ./testlogs/test_BM3_clothing_neg_29.log 2>&1 &

# INFO 	Parameters: ['seed', 'n_layers', 'reg_weight', 'dropout']=(999, 1, 0.1, 0.5),
nohup python main.py \
--model BM3 \
--dataset sports \
--mode test \
--ckpt_dir '../checkpoints/BM3_sports_best.pth' > ./testlogs/test_BM3_sports_neg_29.log 2>&1 &

# INFO 	Parameters: ['seed', 'n_layers', 'reg_weight', 'dropout']=(999, 1, 0.1, 0.5)
nohup python main.py \
--model BM3 \
--dataset toys \
--mode test \
--ckpt_dir '../checkpoints/BM3_toys_best.pth' > ./testlogs/test_BM3_toys_neg_29.log 2>&1 &

# INFO 	Parameters: ['seed', 'n_layers', 'reg_weight', 'dropout']=(999, 1, 0.01, 0.3)
nohup python main.py \
--model BM3 \
--dataset beauty \
--mode test \
--ckpt_dir '../checkpoints/BM3_beauty_best.pth' > ./testlogs/test_BM3_beauty_neg_29.log 2>&1 &