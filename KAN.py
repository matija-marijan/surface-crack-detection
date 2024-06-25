import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from kan import *

# ------------------------------------------------------------------------------------------------------------------
# Ucitavanje i prikaz obelezja

# positive_all = pd.read_csv('C:/Users/Matija/Desktop/ETF/5. GODINA/3 - MASTER RAD/Obelezja/positive_norm_all.csv', header=None)
# negative_all = pd.read_csv('C:/Users/Matija/Desktop/ETF/5. GODINA/3 - MASTER RAD/Obelezja/negative_norm_all.csv', header=None)

positive_all = pd.read_csv('C:/Users/Matija/Desktop/ETF/5. GODINA/3 - MASTER RAD/Obelezja/positive_PCA.csv', header=None)
negative_all = pd.read_csv('C:/Users/Matija/Desktop/ETF/5. GODINA/3 - MASTER RAD/Obelezja/negative_PCA.csv', header=None)

# positive_all = pd.read_csv('C:/Users/Matija/Desktop/ETF/5. GODINA/3 - MASTER RAD/Obelezja/positive_LDA.csv', header=None)
# negative_all = pd.read_csv('C:/Users/Matija/Desktop/ETF/5. GODINA/3 - MASTER RAD/Obelezja/negative_LDA.csv', header=None)

positive_all = np.array(positive_all)
negative_all = np.array(negative_all)

N = len(negative_all)

if negative_all.shape[0] > 3:
    ind_ob = [1, 6, 15]     # 2, 7, 16
else:
    ind_ob = [0, 1, 2]

positive_obelezja = positive_all[ind_ob, :]
negative_obelezja = negative_all[ind_ob, :]

ob1_min = min(np.min(negative_obelezja[0, :]), np.min(positive_obelezja[0, :]))
ob2_min = min(np.min(negative_obelezja[1, :]), np.min(positive_obelezja[1, :]))
ob3_min = min(np.min(negative_obelezja[2, :]), np.min(positive_obelezja[2, :]))

ob1_max = max(np.max(negative_obelezja[0, :]), np.max(positive_obelezja[0, :]))
ob2_max = max(np.max(negative_obelezja[1, :]), np.max(positive_obelezja[1, :]))
ob3_max = max(np.max(negative_obelezja[2, :]), np.max(positive_obelezja[2, :]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(negative_obelezja[0, :], negative_obelezja[1, :], negative_obelezja[2, :], c='b', marker='.', label='Bez pukotine')
ax.scatter(positive_obelezja[0, :], positive_obelezja[1, :], positive_obelezja[2, :], c='r', marker='.', label='Sa pukotinom')
ax.set_xlabel(f'Obelezje {ind_ob[0] + 1}')
ax.set_ylabel(f'Obelezje {ind_ob[1] + 1}')
ax.set_zlabel(f'Obelezje {ind_ob[2] + 1}')
ax.set_title('Prikaz obelezja u prostoru')
ax.legend()
plt.show()

# ------------------------------------------------------------------------------------------------------------------
# Podela na skupove

pos_train, pos_test = train_test_split(positive_obelezja.T, test_size=0.2, random_state=42)
neg_train, neg_test = train_test_split(negative_obelezja.T, test_size=0.2, random_state=42)
pos_train = pos_train.T
pos_test = pos_test.T
neg_train = neg_train.T
neg_test = neg_test.T

# ------------------------------------------------------------------------------------------------------------------
# Kolmogorov-Arnold Network

# Dataset creation
dataset = {}
