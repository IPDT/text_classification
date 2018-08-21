class CNNRNNConfig(object):
    forced_sequence_length = 1000
    batch_size = 128
    dropout_keep_prob = 0.5
    embedding_dim = 300
    evaluate_every = 200
    filter_sizes = "3,4,5"
    hidden_unit = 300
    l2_reg_lambda = 0.2
    max_pool_size = 2
    non_static = False
    num_filters = 32
    num_epochs = 5
