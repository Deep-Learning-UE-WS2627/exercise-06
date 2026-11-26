import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop("Name", axis=1)
df = df.drop("PassengerId", axis=1)

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
def make_numeric(df):
  df["Sex"] = pd.factorize(df["Sex"])[0]
  df["Cabin"] = pd.factorize(df["Cabin"])[0]
  df["Ticket"] = pd.factorize(df["Ticket"])[0]
  df["Embarked"] = pd.factorize(df["Embarked"])[0]
  return df
df = make_numeric(df)

# 3. Remove all rows that contain missing values
df = df.dropna()

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
y = df["Survived"]
x = df.drop("Survived", axis=1)

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.1)

# Scale from training data only, the network input has one value per feature
print("Input shape:", x_train.shape[1])
scaler = StandardScaler()
x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
train_data = TensorDataset(x_train, torch.tensor(y_train.to_numpy(), dtype=torch.long))
loader = DataLoader(train_data, batch_size=32, shuffle=True)
torch.manual_seed(0)

def evaluate(model, epochs=100, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            nn.functional.nll_loss(model(xb).clamp_min(1e-8).log(), yb).backward()
            optimizer.step()
    with torch.no_grad():
        predictions = model(x_test).argmax(dim=1).numpy()
    print("precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(
        precision_score(y_test, predictions), recall_score(y_test, predictions),
        f1_score(y_test, predictions)))


# Step 4.1: 10-unit hidden softmax layer, softmax output
print("Network 1: hidden layer 10 (softmax)")
evaluate(nn.Sequential(nn.Linear(x_train.shape[1], 10), nn.Softmax(dim=1),
                       nn.Linear(10, 2), nn.Softmax(dim=1)))

# Step 4.2: 20-unit sigmoid and 10-unit ReLU hidden layers, softmax output
print("Network 2: hidden layers 20 (sigmoid), 10 (relu)")
evaluate(nn.Sequential(nn.Linear(x_train.shape[1], 20), nn.Sigmoid(),
                       nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 2),
                       nn.Softmax(dim=1)))

# Step 4.3: Experiments with other layer settings
print("Experiment 1: hidden layers 32 and 16 (relu)")
evaluate(nn.Sequential(nn.Linear(x_train.shape[1], 32), nn.ReLU(),
                       nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2),
                       nn.Softmax(dim=1)))
print("Experiment 2: hidden layers 100 and 200 (relu)")
evaluate(nn.Sequential(nn.Linear(x_train.shape[1], 100), nn.ReLU(),
                       nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 2),
                       nn.Softmax(dim=1)))
print("Experiment 3: hidden layers 50 (sigmoid) and 25 (relu)")
evaluate(nn.Sequential(nn.Linear(x_train.shape[1], 50), nn.Sigmoid(),
                       nn.Linear(50, 25), nn.ReLU(), nn.Linear(25, 2),
                       nn.Softmax(dim=1)))