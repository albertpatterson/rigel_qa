import os
import pickle


def write_data(data, filename):
    full_path = os.path.join("data", filename)
    if not os.path.exists("data"):
        os.makedirs("data")

    pickle.dump(data, open(full_path, "wb"))


def load_data(filename):
    path = os.path.join("data", filename)
    data = pickle.load(open(path, "rb"))
    return data
