import os
import os.path as osp

root_path = osp.dirname(osp.realpath(__file__))
data_use_only_pocket = True
atomic_distance_cutoff = 4.0

# Change this line is necessary
data_path = osp.join(root_path, 'data')
decoy_path = osp.join(data_path, 'decoys_docking')
# ----------------
index_dir_path = osp.join(data_path, 'index')
indexes_path = {'general': osp.join(index_dir_path, 'INDEX_general_PL_data.2020'),
                'refined': osp.join(index_dir_path, 'INDEX_refined_data.2020'),
                'core': osp.join(index_dir_path, 'CoreSet_2016.dat'),
                'core_2013': osp.join(index_dir_path, '2013_core_data.lst')}
experiments_path = osp.join(root_path, 'experiments')
if not osp.isdir(experiments_path):
    os.mkdir(experiments_path)

# The following lines may be change

hidden_channels_pa = 128
hidden_channels_la = 128
num_layers = 4
num_timesteps = 4
p_dropout = 0.05
heads = 1
hetero_aggr = 'sum'

input_size = hidden_channels_pa + hidden_channels_la
mlp_channels = [input_size, input_size*2, input_size*3, 1]

learning_rate = 1e-4
weight_decay = 1e-4

nb_epochs = 150
batch_size = 64
datamodule_num_worker = 80
