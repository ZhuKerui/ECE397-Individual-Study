import numpy as np
from embeddings.play import Play
import io

# -------------------------------- Extract Rel -------------------------------- #
# data = np.vstack([np.load('data/train_data'+str(i)+'.npy') for i in range(28)])
# rel = data[:, 12:18]
# rel_uni, rel_cnt = np.unique(rel, return_counts=True, axis=0)
# rel_list = np.hstack((rel_uni, rel_cnt.reshape(-1,1)))
# rel_list = sorted(rel_list, key=lambda x: x[-1], reverse=True)
# np.save('data/rel.npy', rel_list)

# -------------------------------- Filter Rel -------------------------------- #
# rel_list = np.load('data/rel.npy')
# filtered_mask = rel_list[:, -1] >= 20
# np.save('data/rel_20.npy', rel_list[filtered_mask])

# -------------------------------- Translate Rel -------------------------------- #
# rel = np.load('data/rel_20.npy')
# rel_uni, rel_cnt = rel[:, :-1], rel[:, -1]
# p = Play('data/result_3/best_model.pt', 'data/result_3/saved_config.json')
# rel_text = [' '.join(p.itos(arr)) for arr in rel_uni]
# with io.open('data/rel_20.txt', 'w', encoding='utf-8') as rel_out:
#     rel_out.write('\n'.join(rel_text))

# -------------------------------- Extract Arg -------------------------------- #
# data = np.vstack([np.load('data/train_data'+str(i)+'.npy') for i in range(10)])
# arg1 = data[:, : 6]
# arg2 = data[:, 6 : 12]
# arg = np.vstack((arg1, arg2))
# arg_uni, arg_cnt = np.unique(arg, return_counts=True, axis=0)
# arg_list = np.hstack((arg_uni, arg_cnt.reshape(-1,1)))
# arg_list = sorted(arg_list, key=lambda x: x[-1], reverse=True)
# np.save('data/arg.npy', arg_list)

# -------------------------------- Filter Arg -------------------------------- #
# arg_list = np.load('data/arg.npy')
# filtered_mask = arg_list[:, -1] >= 10
# np.save('data/arg_10.npy', arg_list[filtered_mask])

# -------------------------------- Translate Arg -------------------------------- #
# arg = np.load('data/arg_10.npy')
# arg_uni, arg_cnt = arg[:, :-1], arg[:, -1]
# p = Play('data/result/best_model.pt', 'data/result/saved_config.json')
# arg_text = [' '.join(p.itos(arr)) for arr in arg_uni]
# with io.open('data/arg_10.txt', 'w', encoding='utf-8') as arg_out:
#     arg_out.write('\n'.join(arg_text))