import cPickle as pickle


def save_tree(mdts_tree, filename):
    pickle.dump(mdts_tree, open(filename,'wb'))


def load_tree(filename):
    return pickle.load(open(filename, 'rb' ))