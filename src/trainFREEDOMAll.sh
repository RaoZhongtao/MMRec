nohup python main.py \
--model FREEDOM \
--dataset beauty \
--mode train > train_freedom_beauty.log 2>&1 &

python main.py \
--model FREEDOM \
--dataset clothing \
--mode train > train_freedom_clothing.log 2>&1 &

python main.py \
--model FREEDOM \
--dataset sports \
--mode train > train_freedom_sports.log 2>&1 &

python main.py \
--model FREEDOM \
--dataset toys \
--mode train > train_freedom_toys.log 2>&1 &