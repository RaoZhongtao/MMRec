# INFO 	Parameters: ['seed', 'dropout', 'reg_weight']=(999, 0.9, 0.0),
# ---learning_rate=0.001
nohup python main.py \
--model FREEDOM \
--dataset clothing \
--mode test \
--ckpt_dir '../checkpoints/FREEDOM_clothing_best.pth' > ./testlogs/test_FREEDOM_400_clothing.log 2>&1 &

# INFO 	Parameters: ['seed', 'dropout', 'reg_weight']=(999, 0.9, 0.0),
nohup python main.py \
--model FREEDOM \
--dataset sports \
--mode test \
--ckpt_dir '../checkpoints/FREEDOM_sports_best.pth' > ./testlogs/test_FREEDOM_400_sports.log 2>&1 &

# INFO 	Parameters: ['seed', 'dropout', 'reg_weight']=(999, 0.8, 0.0),
nohup python main.py \
--model FREEDOM \
--dataset toys \
--mode test \
--ckpt_dir '../checkpoints/FREEDOM_toys_best.pth' > ./testlogs/test_FREEDOM_400_toys.log 2>&1 &

# INFO 	Parameters: ['seed', 'dropout', 'reg_weight']=(999, 0.9, 0.0001)
nohup python main.py \
--model FREEDOM \
--dataset beauty \
--mode test \
--ckpt_dir '../checkpoints/FREEDOM_beauty_best.pth' > ./testlogs/test_FREEDOM_400_beauty.log 2>&1 &