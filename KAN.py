import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from kan import *

# ------------------------------------------------------------------------------------------------------------------
# Ucitavanje i prikaz obelezja

positive_all = pd.read_csv('Obelezja/positive_norm_all.csv', header=None)
negative_all = pd.read_csv('Obelezja/negative_norm_all.csv', header=None)

# positive_all = pd.read_csv('Obelezja/positive_PCA.csv', header=None)
# negative_all = pd.read_csv('Obelezja/negative_PCA.csv', header=None)

# positive_all = pd.read_csv('Obelezja/positive_LDA.csv', header=None)
# negative_all = pd.read_csv('Obelezja/negative_LDA.csv', header=None)

positive_all = np.array(positive_all)
negative_all = np.array(negative_all)

N = negative_all.shape[1]

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

# Function to randomly sample 5% of the data for visualization
def sample_data(data, percentage=0.05):
    num_samples = int(data.shape[1] * percentage)
    indices = np.random.choice(data.shape[1], num_samples, replace=False)
    return data[:, indices]

# Sample 5% of the data
negative_sampled = sample_data(negative_obelezja)
positive_sampled = sample_data(positive_obelezja)

# Plot feature distribution in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(negative_obelezja[0, :], negative_obelezja[1, :], negative_obelezja[2, :], c='b', marker='.', label='Bez pukotine')
# ax.scatter(positive_obelezja[0, :], positive_obelezja[1, :], positive_obelezja[2, :], c='r', marker='.', label='Sa pukotinom')
ax.scatter(negative_sampled[0, :], negative_sampled[1, :], negative_sampled[2, :], c='b', marker='.', label='Bez pukotine')
ax.scatter(positive_sampled[0, :], positive_sampled[1, :], positive_sampled[2, :], c='r', marker='.', label='Sa pukotinom')
ax.set_xlabel(f'Obelezje {ind_ob[0] + 1}')
ax.set_ylabel(f'Obelezje {ind_ob[1] + 1}')
ax.set_zlabel(f'Obelezje {ind_ob[2] + 1}')
ax.set_title('Prikaz obelezja u prostoru')
ax.legend()
plt.show()

# ------------------------------------------------------------------------------------------------------------------
# Train/test split

N_train = np.round(0.8 * N).astype(int)
N_test = N - N_train

pos_train, pos_test = train_test_split(positive_obelezja.T, test_size=0.2, random_state=42)
neg_train, neg_test = train_test_split(negative_obelezja.T, test_size=0.2, random_state=42)
pos_train = pos_train.T
pos_test = pos_test.T
neg_train = neg_train.T
neg_test = neg_test.T

# ------------------------------------------------------------------------------------------------------------------
# Dataset creation
X_train = np.concatenate((pos_train, neg_train), axis = 1).T
Y_train = np.concatenate((np.zeros(N_train), np.ones(N_train)))

X_test = np.concatenate((pos_test, neg_test), axis = 1).T
Y_test = np.concatenate((np.zeros(N_test), np.ones(N_test)))

dataset = {}
dataset['train_input'] = torch.from_numpy(X_train)
dataset['test_input'] = torch.from_numpy(X_test)
dataset['train_label'] = torch.from_numpy(Y_train).long()
dataset['test_label'] = torch.from_numpy(Y_test).long()

X_plot = dataset['train_input']
Y_plot = dataset['train_label']

# Test dataset creation by plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_plot[:,0], X_plot[:,1], X_plot[:,2], c=Y_plot[:,0])
# plt.show()

# ------------------------------------------------------------------------------------------------------------------
# Kolmogorov-Arnold Network

model = KAN(width=[3, 2], grid=3, k=3)

def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

# start_time = time.time()

results = model.train(dataset, opt="LBFGS", steps=25, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss())

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Training time: {elapsed_time} [s]")

print(results['train_acc'][-1])
print(results['test_acc'][-1])

# ------------------------------------------------------------------------------------------------------------------
# Predictions on train and test sets and saving for MATLAB post-processing

prediction_train = model(dataset['train_input']).detach().numpy()
prediction_train = np.argmax(prediction_train, axis=1)
np.savetxt('prediction_train.csv', prediction_train, delimiter=',')

prediction_test = model(dataset['test_input']).detach().numpy()
prediction_test = np.argmax(prediction_test, axis=1)
np.savetxt('prediction_test.csv', prediction_test, delimiter=',')

d = 0.15
x, y, z = np.meshgrid(np.arange(ob1_min - 0.5, ob1_max + 0.5, d),
                      np.arange(ob2_min - 0.5, ob2_max + 0.5, d),
                      np.arange(ob3_min - 0.5, ob3_max + 0.5, d))
xGrid = np.c_[x.ravel(), y.ravel(), z.ravel()]
xGrid = torch.from_numpy(xGrid)

probs = model(xGrid)
f = probs[:, 1].reshape(x.shape)
f = f.detach().numpy()
f_2d = f.reshape(-1, f.shape[-1])
np.savetxt('probs.csv', f_2d, delimiter=',')
