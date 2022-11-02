import os
import os.path as osp

root_path = osp.dirname(osp.realpath(__file__))
root_path = osp.dirname(root_path)

data_use_only_pocket = True

model_parameters_path = osp.join(root_path, 'model_parameters.yaml')
data_path = osp.join(root_path, 'data')
decoy_path = osp.join(data_path, 'decoys_docking')

index_dir_path = osp.join(data_path, 'index')
indexes_path = {'general': osp.join(index_dir_path, 'INDEX_general_PL_data.2020'),
                'refined': osp.join(index_dir_path, 'INDEX_refined_data.2020'),
                'core': osp.join(index_dir_path, 'CoreSet_2016.dat'),
                'core_2013': osp.join(index_dir_path, '2013_core_data.lst')}
experiments_path = osp.join(root_path, 'experiments')
if not osp.isdir(experiments_path):
    os.mkdir(experiments_path)
