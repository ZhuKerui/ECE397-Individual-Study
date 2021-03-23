python -m embeddings.train --config experiments/pair2vec_train.json --save_path data/result_3/ --gpu 1
echo "Task is done." | mail -s "Task done" -A data/result_3/stdout.log keruiz2@illinois.edu