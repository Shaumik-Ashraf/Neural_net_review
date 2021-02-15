# Iris FFNN
# Shaumik Ashraf

import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import pandas as pd;
#import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split;
from sklearn.datasets import load_iris;
from sklearn.metrics import f1_score;


print("Initializing...");

# setup dataset
iris = load_iris();
X = iris.data;
y = iris.target;

#note: if train_test_split random_state = 42, then accuracy is 100%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2);
X_train = torch.tensor(X_train, dtype=torch.float);
X_test = torch.tensor(X_test, dtype=torch.float);
y_train = torch.tensor(y_train, dtype=torch.long);
y_test = torch.tensor(y_test, dtype=torch.long);


# define model+optimizer
class MyNN(nn.Module):
    def __init__(self):
        super().__init__();
        self.fc1 = nn.Linear(in_features=4, out_features=16);
        self.fc2 = nn.Linear(in_features=16, out_features=12);
        self.output = nn.Linear(in_features=12, out_features=3);
        
    def forward(self, x):
        x = F.relu(self.fc1(x));
        x = F.relu(self.fc2(x));
        x = self.output(x);
        return x;

model = MyNN();
criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(model.parameters(), lr=0.01);


# training
epochs = 100;
loss_arr = [];

for i in range(epochs):
    y_hat = model.forward(X_train);
    loss = criterion(y_hat, y_train);
    loss_arr.append(loss);
    
    if( i%10 == 0 ):
        print(f"Epoch: {i} Loss: {loss}");

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();


# testing
# parameters were trained from back propogation
preds = [];

with torch.no_grad():
    for x in X_test:
        y_hat = model.forward(x);
        preds.append( y_hat.argmax().item() );
        
df = pd.DataFrame({'Y': y_test, 'YHat': preds});
df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])];
#print(df.head(5));

# evaluate
f1 = f1_score(y_test, preds, average='micro');
print(f"F1 Score: {f1}");
print("Done.");