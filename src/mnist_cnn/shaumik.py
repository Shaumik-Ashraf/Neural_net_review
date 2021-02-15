# MNIST CNN
# Shaumik Ashraf
"""
Notes:
    num_workers set to zero because windows
    
"""

import torch;
import torch.nn as nn;
import torch.nn.functional as F;
from torchvision import transforms;
from torchvision.datasets import MNIST;
from sklearn.metrics import f1_score;


model_path = "./shaumik_mnist_cnn.pth"
data_path = "./data"


print("Initializing...");
xform = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize(0.5,0.5)]);

train_data = MNIST(root=data_path, train=True, download=True, transform=xform);
test_data = MNIST(root=data_path, train=False, download=True, transform=xform);

train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, 
                                           shuffle=True, num_workers=0);
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4,
                                          shuffle=False, num_workers=0);

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9');



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__();
        
        #first convolution layer taking 1 input channel, 32 output channels, and 3x3 kernal
        self.conv1 = nn.Conv2d(1, 32, 3, 1);
        #2nd convolution layer taking 32 input channel, 64 output channels, and 3x3 kernal
        self.conv2 = nn.Conv2d(32, 64, 3, 1);
        
        #dropouts
        self.dropout1 = nn.Dropout2d(0.25);
        self.dropout2 = nn.Dropout2d(0.5);
        
        #first fully connected layer
        self.fc1 = nn.Linear(9216, 128); #9216 = 32 * 32 * 3 * 3 = image x kernal
        #second fully connected layer
        self.fc2 = nn.Linear(128, 10);
        
    def forward(self, x):
        x = F.relu(self.conv1(x));
        x = F.relu(self.conv2(x));
        x = F.max_pool2d(x, 2); #max pooling 
        x = self.dropout1(x);
        x = torch.flatten(x,1); #flatten x with dim=1
        x = F.relu(self.fc1(x));
        x = self.dropout2(x);
        x = F.log_softmax(self.fc2(x), dim=1); #last layer gets softmaxed
        return x;
    
cnn = CNN();
criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum = 0.9);



# This code is for training model - do not run if saved trained model exists
print("Training", cnn);
for epoch in range(2):
    running_loss = 0.0;
    for i, data in enumerate(train_loader, 0):
        x_train, y_train = data;
        
        optimizer.zero_grad();
        
        y_hat = cnn(x_train);
        loss = criterion(y_hat, y_train);
        loss.backward();
        optimizer.step();
        
        #print stats every 2000 mini-batches
        running_loss += loss.item();
        if( i%2000 == 1999 ):
            print("[%d %5d] loss: %0.3f" % (epoch, i+1, running_loss/2000));
            running_loss = 0.0;

print("Training done.");
torch.save(cnn.state_dict(), model_path);


"""
# This code is for loading a saved trained model
print("Loading trained parameters");
trained_state = torch.load( model_path );
cnn.load_state_dict(trained_state);
print("Loading done.")
"""


print("Testing");

with torch.no_grad():
    for i, (x_test, y_test) in enumerate(test_loader, 0):
        y_pred = cnn(x_test);
        print( 'True:', y_test, "|Pred:", y_pred );
        if( i==10 ):
            break;

print("Done.");