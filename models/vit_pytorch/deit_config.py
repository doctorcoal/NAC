import ml_collections

def get_deit_base_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    """ DeiT configuration"""
    config.patch_size   = 16
    config.embed_dim    = 768
    config.depth        = 12
    config.num_heads    = 12
    config.mlp_ratio    = 4
    return config

def get_deit_small_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1536
    config.transformer.num_heads = 6
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None


    """ DeiT configuration"""
    config.patch_size   = 16
    config.embed_dim    = 384
    config.depth        = 12
    config.num_heads    = 6
    config.mlp_ratio    = 4
    return config

def get_deit_tiny_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 3
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    """ DeiT configuration"""
    config.patch_size   = 16
    config.embed_dim    = 192
    config.depth        = 12
    config.num_heads    = 3
    config.mlp_ratio    = 4
    return config

def get_config(scale: str):
    if scale == 'tiny':
        return get_deit_tiny_config()
    elif scale == 'small':
        return get_deit_small_config()
    elif scale == 'base':
        return get_deit_base_config()