import pandas as pd
import config
import matplotlib as plt
import numpy as np
from PIL import Image
import random

df = pd.read_csv(config.root_dir + '/vpr/vpr_techniques/techniques/selectCNN/data/nordlands_f1_train.csv')


methods_correct = [[], [], [], []]

for i in range(len(df.index)):
    select = np.argmax(df.iloc[i].to_numpy()[2:])
    if np.max(df.iloc[i].to_numpy()[2:] > 0.7):
        methods_correct[select].append(df["query_images"].to_numpy()[i])
    if select == 1:
        methods_correct[select].append(df["query_images"].to_numpy()[i])
    else:
        continue


for method in methods_correct:
    random.shuffle(method)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.figure(figsize = (9,4))
gs1 = gridspec.GridSpec(4, 9)
gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes.

print("Methods successes: ", len(methods_correct[0]), len(methods_correct[1]), len(methods_correct[2]), len(methods_correct[3]))

for i in range(4):
    for j in range(9):
   # i = i + 1 # grid spec indexes from 0
        ax1 = plt.subplot(gs1[i,j])
        try:
            ax1.imshow(np.array(Image.open(methods_correct[i][j]).resize((224, 224))))
        except:
            continue
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')

plt.show()


methods_only_correct = [[], [], [], []]

a = [1, 0, 0, 0]
b = [0, 1, 0, 0]
c = [0, 0, 1, 0]
d = [0, 0, 0, 1]

for i in range(len(df.index)):
    gt = list(df.iloc[i].to_numpy()[2:])
    if gt == a:
        methods_only_correct[0].append(df["query_images"].to_numpy()[i])
    if gt == b:
        methods_only_correct[1].append(df["query_images"].to_numpy()[i])
    if gt == c:
        methods_only_correct[2].append(df["query_images"].to_numpy()[i])
    if gt == d:
        methods_only_correct[3].append(df["query_images"].to_numpy()[i])

for method in methods_only_correct:
    random.shuffle(method)




