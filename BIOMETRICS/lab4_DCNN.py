# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:33:04 2022

@author: afaca
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from os import listdir
from os.path import isfile, join
import copy

torch.zeros(1).cuda()

cudnn.benchmark = True
plt.ion()   # interactive mode

data_dir_test='fvc2000_final_test'
test_list = [f for f in listdir('fvc2000_final_test') if isfile(join('fvc2000_final_test', f))]
data_dir = 'fvc2000/'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomAffine(5,translate=(0.2,0.2),scale=(0.9,1.1))
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomAffine(5,translate=(0.2,0.2),scale=(0.9,1.1))
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}



dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

model = torchvision.models.resnet50()
# model_ft = models.resnet50(pretrained=True)
weights = model.fc.weight

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)
model.to(device)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.01)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    weights = model.fc.weight
                    #tf.norm(tensor, ord='euclidean', axis=None, keepdims=None, name=None)
                    #tf.sqrt(tf.reduce_sum(tf.square(w)))
                    Wfc=torch.norm(weights, p='fro', dim=None, keepdim=False, out=None, dtype=None)
                    l = 0.1
                    loss = criterion(outputs, labels) + l*Wfc

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


################################################################################################################
def test_model(model, criterion, optimizer, scheduler, num_epochs=len(test_list)):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

            # Iterate over data.
        for inputs, labels in data_dir_test[epoch]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(test_list[epoch])
        epoch_acc = running_corrects.double() / len(test_list[epoch])

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

################################################################################################################


# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=2)

visualize_model(model_ft)
PATH = 'model.pt'
torch.save(model_ft, PATH)

model_tested = test_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=len(test_list))

model_tested = model_tested.unsqueeze(0)

###################################################################################


preds = []
with torch.no_grad():
   for val in test_list:
       y_hat = model_tested.forward(val)
       preds.append(y_hat.argmax().item())


df = pd.DataFrame({'Y': test_list, 'YHat': preds})
df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]




#############################################################################
# Disable grad
# with torch.no_grad():
    
#     # Retrieve item
#     index = 3
#     item = test_list[index]
#     image = item[0]
#     true_target = item[1]
    
#     # Loading the saved model
#     save_path = './mlp.pth'
#     mlp = MLP()
#     mlp.load_state_dict(torch.load(save_path))
#     mlp.eval()
    
#     # Generate prediction
#     prediction = mlp(image)
    
#     # Predicted class value using argmax
#     predicted_class = np.argmax(prediction)
    
#     # Reshape image
#     image = image.reshape(28, 28, 1)
    
#     # Show result
#     plt.imshow(image, cmap='gray')
#     plt.title(f'Prediction: {predicted_class} - Actual target: {true_target}')
#     plt.show()

