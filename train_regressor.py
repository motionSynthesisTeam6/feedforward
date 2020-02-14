import os
import sys
import torch
from Feedforward import *
from network_new import *
from data import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")



def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)  


def decoder(autoencoder, Upsilon_T):
    x = autoencoder.unpool1d(Upsilon_T)
    x = autoencoder.dropout2(x)
    x = autoencoder.conv2(x)
    return x


save_dir = 'model_dir'


if __name__=='__main__':
    print('=====>Prepare dataloader ...')
    data = DATA(mode='train')
    X = torch.utils.data.DataLoader(data,batch_size=1,shuffle=True)
    train_loader = torch.utils.data.DataLoader(data[:,-7:],batch_size=1,shuffle=True)



    print('=====>Prepare model ...')
    autoencoder = AutoEncoder().to(computing_device)
    autoencoder.load_state_dict(torch.load("./model_100.pth.tar",map_location=torch.device('cpu')))
    autoencoder.eval()

    feedforward = Feedforward().to(computing_device)

    omega = [64,128,45,25,15,7]

    #Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
    optimizer = optim.Adam(feedforward.parameters(),lr = 0.001)
    writer = SummaryWriter(os.path.join(save_dir, 'train_info'))

    alpha = 0.1
    epoch_num = 200
    iters = 0

    for epoch in range(epoch_num):
        train_loss = 0
        
        for idx, (Upsilon_T, data_X) in enumerate(zip(train_loader, X)):
            iters += 1
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch+1, idx+1, len(train_loader))
            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()
            
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
    #         Upsilon_T, encoded_X = Upsilon_T.to(computing_device), encoded_X.to(computing_device)

            outputs_T = feedforward(Upsilon_T)
            criterion = nn.MSELoss()
            decoded_T = decoder(autoencoder, outputs_T)

            loss = criterion(data_X, decoded_T) + alpha*np.sum(np.abs(omega))
        
            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()

            # Update the weights
            optimizer.step()    
            # Add this iteration's loss to the total_loss
            
            train_loss += loss
            
            if iters%1000 == 0:    
                    writer.add_scalar('loss', train_loss.data.cpu().numpy(), iters)
                    train_info += ' loss: {:.4f}'.format(train_loss.data.cpu().numpy())
                    sys.stdout.write('\r')
                    sys.stdout.write(train_info)
                    sys.stdout.flush()
        print('=====>Save model ...')
        save_model(feedforward, os.path.join(save_dir, 'model_feedforward{}.pth.tar'.format(epoch)))










