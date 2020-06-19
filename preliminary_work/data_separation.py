import pandas as pd
import os

data = pd.read_csv("../maestro-v2.0.0/maestro-v2.0.0.csv")

print("Datos leídos")

maestro_path = "../maestro-v2.0.0/"

idx_train = idx_valid = idx_test = 0

for i in range(len(data)):
    title = data.iloc[i,4]
    split = data.iloc[i,2]
    if split == 'train':
        idx = idx_train
        idx_train += 1
    elif split == 'validation':
        idx = idx_valid
        idx_valid +=1
    else:
        idx = idx_test
        idx_test += 1

    midi = open(maestro_path + title, 'r')
    path = open("C:\Users\Pau\Documents\uni\maestro-v2.0.0\\" + split, 'w')
    path.write(midi)
    os.rename(maestro_path + split + title, maestro_path + split + str(idx) + '.mid')
    print("funciona")
