import pandas as pd

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

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.1)

# 3. Train feed-forward neural networks with PyTorch.
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Fit the scaler on training data only to avoid data leakage.
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
Y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)

input_size = X_train.shape[1]
print("Input shape:", x.shape)
print("Train/test:", X_train.shape, X_test.shape)

def make_model(architecture):
    if architecture == 1:
        return nn.Sequential(nn.Linear(input_size, 10), nn.Softmax(dim=1),
                             nn.Linear(10, 2), nn.Softmax(dim=1))
    return nn.Sequential(nn.Linear(input_size, 20), nn.Sigmoid(),
                         nn.Linear(20, 10), nn.ReLU(),
                         nn.Linear(10, 2), nn.Softmax(dim=1))


def train_and_evaluate(architecture, epochs=300, lr=0.01):
    torch.manual_seed(0)
    model = make_model(architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # NLLLoss expects log probabilities, not softmax probabilities.
    criterion = nn.NLLLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        probabilities = model(X_train)
        loss = criterion(probabilities.clamp_min(1e-12).log(), Y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1).numpy()
    print(f"Model {architecture}, epochs={epochs}, lr={lr}: "
          f"precision={precision_score(y_test, y_pred, zero_division=0):.4f}, "
          f"recall={recall_score(y_test, y_pred, zero_division=0):.4f}, "
          f"f1={f1_score(y_test, y_pred, zero_division=0):.4f}")


# Limit CPU threading for this small dataset.
torch.set_num_threads(1)
train_and_evaluate(1)
train_and_evaluate(2)
# Step 4.3: Compare learning rates and training duration.
train_and_evaluate(1, epochs=600, lr=0.005)
train_and_evaluate(2, epochs=600, lr=0.005)
train_and_evaluate(2, epochs=100, lr=0.01)
