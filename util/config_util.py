import json

def load_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

def save_config(filename, config):
    with open(filename, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

def load_train_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

def save_train_config(filename, train_config):
    with open(filename, 'w') as f:
        json.dump(train_config, f, sort_keys=True, indent=4)
