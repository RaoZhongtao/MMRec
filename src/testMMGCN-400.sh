
# INFO 	Parameters: ['seed', 'reg_weight', 'learning_rate']=(999, 0, 0.0001)
nohup python main.py \
--model MMGCN \
--dataset clothing \
--mode test \
--ckpt_dir '../checkpoints/MMGCN_clothing_best.pth' > ./testlogs/test_MMGCN_400_clothing.log 2>&1 &

# INFO 	Parameters: ['seed', 'reg_weight', 'learning_rate']=(999, 0.01, 0.0005)
nohup python main.py \
--model MMGCN \
--dataset sports \
--mode test \
--ckpt_dir '../checkpoints/MMGCN_sports_best.pth' > ./testlogs/test_MMGCN_400_sports.log 2>&1 &

# INFO 	Parameters: ['seed', 'reg_weight', 'learning_rate']=(999, 1e-05, 0.001)
nohup python main.py \
--model MMGCN \
--dataset toys \
--mode test \
--ckpt_dir '../checkpoints/MMGCN_toys_best.pth' > ./testlogs/test_MMGCN_400_toys.log 2>&1 &

# INFO 	Parameters: ['seed', 'reg_weight', 'learning_rate']=(999, 0.1, 0.001)
nohup python main.py \
--model MMGCN \
--dataset beauty \
--mode test \
--ckpt_dir '../checkpoints/MMGCN_beauty_best.pth' > ./testlogs/test_MMGCN_400_beauty.log 2>&1 &