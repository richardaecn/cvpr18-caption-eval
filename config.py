class Config(object):
    """ Model Configuration
    [use_img_feat] options control how we use image features
        None                  not using image features (default)
        "concat_bf_lstm"      concatenate image feature before LSTM
        "concat_af_lstm"      concatenate image feature after LSTM
        "only_img"            image feature only
    [combine_typ]  how do we combine context feature with candidate feature
        "bilinpool"  using binlinear pooling (default)
        "concat"      concatenate features directly
    [cls_hidden]   number of hidden layers for classifer (all with size 256)
    """
    learning_rate = 0.001
    learning_rate_decay = 0.9
    max_epoch = 30
    grad_clip = 1.0
    num_layers = 1
    num_steps = 15
    hidden_size = 512
    dropout_prob = 0.5
    batch_size = 100
    vocab_size = 10004
    embedding_size = 300
    num_input = 2
    use_lstm  = True

    # How to use Image Feature :
    #   None | 'concat_bf_lstm' | 'concat_af_lstm' | 'only_img'
    use_img_feat= 'concat_af_lstm'

    # How to combine context feature:
    #   'bilinpool' | 'concat'
    combine_typ = 'concat'

    # 0 for basic linear classifier
    cls_hidden = 0
    use_residual         = False # Whether use residual connection in LSTM
    use_random_human     = True  # Whether using random human captions transformations
    use_random_word      = True  # Whether using random word replacement
    use_word_permutation = True  # Whehter using random word permutations
    use_mc_samples       = True  # Whether using Monte Carlo Sampled Captions

def set_no_da(config):
    # not using data augmentation durng training.
    config.use_random_human = False
    config.use_random_word = False
    config.use_word_permutation = False
    config.use_mc_samples = False
    return config

def config_model_coco(config, model_architecture):
    config.num_layers = 1 # using 1 LSTM layer
    # Linear models
    if model_architecture == 'concat_no_img_1_512_0':
        config.use_img_feat = None
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'concat_img_1_512_0':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'concat_only_img_1_512_0':
        config.use_img_feat = 'only_img'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'concat_img_1_512_0_noda':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 0
        config = set_no_da(config)
    # Non-linear models with Compact Bilinear Pooling
    elif model_architecture == 'bilinear_img_1_512_0':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'bilinear_no_img_1_512_0':
        config.use_img_feat = None
        config.combine_typ = 'bilinpool'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'bilinear_only_img_1_512_0':
        config.use_img_feat = 'only_img'
        config.combine_typ = 'bilinpool'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'bilinear_img_1_512_0_noda':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.hidden_size = 512
        config.cls_hidden = 0
        config = set_no_da(config)
    # Non-linear models with MLP
    elif model_architecture == 'mlp_1_img_1_512_0':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 1
    elif model_architecture == 'mlp_1_no_img_1_512_0':
        config.use_img_feat = None
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 1
    elif model_architecture == 'mlp_1_only_img_1_512_0':
        config.use_img_feat = 'only_img'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 1
    elif model_architecture == 'mlp_1_img_1_512_0_noda':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 1
        config = set_no_da(config)
    else:
        raise Exception("Invalid architecture name:%s"%model_architecture)
    return config


def config_model_flickr(config, model_architecture):
    config.use_random_human     = True
    config.use_random_word      = False
    config.use_word_permutation = False
    config.use_mc_samples       = False
    config.batch_size = 50
    config.max_epoch = 100
    config.learning_rate_decay = 0.98
    config.learning_rate = 0.001

    config.batch_size = 100
    config.test_batch_size = 15000
    config.vocab_size = 3441 # Without lemmatization

    if model_architecture == 'baseline':
        return config
    if model_architecture == 'baseline_mlp':
        config.use_img_feat = None
        config.combine_typ = 'concat'
        config.cls_hidden = 1
        return config
    if model_architecture == 'bilinear':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 1
        config.cls_hidden = 0
        return config
    if model_architecture == 'bilinear_moreLSTM':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 0
        return config
    if model_architecture == 'bilinear_clf_moreLSTM':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 1
        return config
    if model_architecture == 'bilinear_sm':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 1
        config.cls_hidden = 0
        config.hidden_size = 128
        return config
    if model_architecture == 'bilinear_moreLSTM_sm':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 0
        config.hidden_size = 128
        return config
    if model_architecture == 'bilinear_clf_moreLSTM':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 1
        config.hidden_size = 128
        return config
    if model_architecture == 'bilinear_clf_moreLSTM_sm':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 1
        config.hidden_size = 128
        return config
    if model_architecture == 'bilinear_bilinear':
        config.use_img_feat = 'bilinpool'
        config.combine_typ = 'bilinpool'
        config.num_layers = 1
        config.cls_hidden = 0
        config.hidden_size = 128
        return config

    # Different Dropout
    if model_architecture == 'bilinear_clf_moreLSTM_dropout0.3':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 1
        config.hidden_size = 128
        config.dropout_prob = 0.3
        return config
    if model_architecture == 'bilinear_clf_moreLSTM_dropout0.1':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 1
        config.hidden_size = 128
        config.dropout_prob = 0.3
        return config
    # Turn the learning rate
    if model_architecture == 'bilinear_clf_moreLSTM_lr0.001':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 1
        config.hidden_size = 128
        config.learning_rate = 0.001
        return config

    if model_architecture == 'bilinear_clf_moreLSTM_lr0.002':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 1
        config.hidden_size = 128
        config.learning_rate = 0.002
        return config

    if model_architecture == 'bilinear_clf_moreLSTM_lr0.0008':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 1
        config.hidden_size = 128
        config.learning_rate = 0.0008
        return config

    if model_architecture == 'bilinear_clf_moreLSTM_lr0.0005':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 1
        config.hidden_size = 128
        config.learning_rate = 0.0005
        return config

    if model_architecture == 'bilinear_clf_moreLSTM_lr0.0002':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 1
        config.hidden_size = 128
        config.learning_rate = 0.0002
        return config

    if model_architecture == 'bilinear_clf_moreLSTM_lr0.0001':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.num_layers = 2
        config.cls_hidden = 1
        config.hidden_size = 128
        config.learning_rate = 0.0001
        return config

    raise Exception("%s not found"%model_architecture)
