import os
import pickle


def write_data(data, filename):
    full_path = os.path.join("data", filename)
    if not os.path.exists("data"):
        os.makedirs("data")

    pickle.dump(data, open(full_path, "wb"))
