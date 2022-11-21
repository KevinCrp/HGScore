# /!\ All low and high limit are included in ranges

# Num Layers for Atomic part
num_layers_atm_low = 3
num_layers_atm_high = 5

# First hidden channel size
layer_1st_pa_low = 24
layer_1st_pa_high = 48

# First hidden channel size
layer_1st_la_low = 24
layer_1st_la_high = 48

# Layer size factor ( Size_i+1 = Size_i * factor)
layer_size_factor_low_pa = 0.5
layer_size_factor_high_pa = 1.5

layer_size_factor_low_la = 0.5
layer_size_factor_high_la = 1.5

# Heads number for all GAT
heads_low = 1
heads_high = 8

# Learning rate (1e-pow)
learning_rate_pow_low = 3
learning_rate_pow_high = 3

# nb_molecular_embedding for Molecular part
nb_molecular_embedding_size_low = 2
nb_molecular_embedding_size_high = 8

# Nb MLP layers
nb_mlp_layer_low = 1
nb_mlp_layer_high = 5

# Dropout (step = 0.01)
dropout_low = 0.00
dropout_high = 0.25

# Hetero Aggr
hetero_aggr = ["sum", "mean", "max"]

# Weight Decay  (1e-pow)
weight_decay_pow_low = 4
weight_decay_pow_high = 4

# InterAtomic Cutoff
intermol_atomic_cutoff = [4.0]

# ----------------

# Number of trial per run
nb_trials = 2

# Number of run per hyperparameters set
# Number of effective training will be nb_trials*nb_runs
nb_runs = 1

# Batch size
batch_size = 64

# NB Epochs
nb_epochs = 1