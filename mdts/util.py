import pickle
from tensorflow.keras import models


def save_tree(mdts_tree, filename):
    if mdts_tree.use_XP:
        mdts_tree.XP.model.save("XP_network_model")
        mdts_tree.XP.model=None
    pickle.dump(mdts_tree, open(filename,'wb'))


def load_tree(filename):
    try:
        with open(filename, 'rb') as f:
            pickle_data = pickle.load(f)
            if pickle_data.use_XP:
                pickle_data.XP.model=models.load_model("XP_network_model")
    except UnicodeDecodeError as e:
        with open(filename, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
            if pickle_data.use_XP:
                pickle_data.XP.model = models.load_model("XP_network_model")
    except Exception as e:
        print('Unable to load data ', filename, ':', e)
        raise
    return pickle_data