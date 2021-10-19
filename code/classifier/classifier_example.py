import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# 0) Prepare data
# df = pd.read_csv('../results/data_s/F_total.csv', header = None)
df2 = pd.read_csv('../results/data0-99/F_total.csv', header = None)
df3 = pd.read_csv('../results/data100-199/F_total.csv', header = None)
df4 = pd.read_csv('../results/data200-299/F_total.csv', header = None)

# print(df)
print("data loaded")
# X = np.array(df.loc[:, 1:2]) #features vectors
# y = np.array(df.loc[:, 3])   #class labels: 1 = concave 0 = convex

X2 = np.array(df2.loc[:, 1:2]) #features vectors
y2 = np.array(df2.loc[:, 3])   #class labels: 1 = concave 0 = convex

X3 = np.array(df3.loc[:, 1:2]) #features vectors
y3 = np.array(df3.loc[:, 3])   #class labels: 1 = concave 0 = convex

X4 = np.array(df4.loc[:, 1:2]) #features vectors
y4 = np.array(df4.loc[:, 3])   #class labels: 1 = concave 0 = convex

X = np.vstack((X2,X3,X4))
y = np.vstack((y2,y3,y4))

# X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) Model
# Linear model f = wx + b , sigmoid at the end
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_features)

# 2) Loss and optimizer
num_epochs = 500
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
PATH = 'Model.pth'
torch.save(model.state_dict(), PATH)
print("saved")


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')