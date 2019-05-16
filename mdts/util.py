import pickle


def save_tree(mdts_tree, filename):
    pickle.dump(mdts_tree, open(filename,'wb'))


def load_tree(filename):
    try:
        with open(filename, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(filename, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', filename, ':', e)
        raise
    return pickle_data