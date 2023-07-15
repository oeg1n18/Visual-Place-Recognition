import pandas as pd
import config
import matplotlib as plt
import numpy as np
from PIL import Image

df = pd.read_csv(config.root_dir + '/vpr/vpr_techniques/techniques/selectCNN/data/f1_dataset.csv')

methods = [[], [], [], []]


for i in range(len(df.index)):
    select = np.argmax(df.iloc[i].to_numpy()[2:])
    methods[select].append(df["query_images"].to_numpy()[i])

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig, axes = plt.subplots(ncols=7, nrows=4, constrained_layout=True)

for i in range(5):
    for j in range(7):
        try:
            axes[i][j].imshow(np.array(Image.open(methods[i][j])))
        except:
            continue
        axes[i][j].set_xticklabels('')
        axes[i][j].set_yticklabels('')

plt.show()

plt.show()

