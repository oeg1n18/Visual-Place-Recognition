import pickle

datasets = ["CosPlace_dataset.pkl", "HOG_dataset.pkl", "MixVPR_dataset.pkl", "NetVLAD_dataset.pkl"]

for ds in datasets:
    with open(ds, 'rb') as f:
        data = pickle.load(f)

    print(ds, type(data), len(data))
