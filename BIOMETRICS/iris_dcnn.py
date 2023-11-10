import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import transforms as T
from torchvision import models
from sklearn.decomposition import PCA
from sklearn import svm, metrics
import numpy as np

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

TRAIN_PERC = 0.75


def main(path):
    # input transformations
    image_transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    # data loader
    mmu_data = ImageFolder(root=path, transform=image_transforms)

    # split data into train and test datasets
    no_of_samples = len(mmu_data.targets)
    train_len = int(TRAIN_PERC * no_of_samples)
    test_len = no_of_samples - train_len
    train_dataset, test_dataset = random_split(mmu_data, (train_len, test_len))

    # dataset loaders
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

    # TODO: load VGG with pretrained weights
    print("--- Original VGG-16:")
    vgg = models.vgg16(pretrained=True)
    #print(vgg)

    # TODO: change the network classifier to the nn.Flatten layer
    print("--- Modified VGG-16:")
    #m = nn.Flatten(1,-1)
    #vgg_mod = m(vgg)
    #print(vgg_mod)

    # extract features using VGG-16 in eval mode - save them in a list
    vgg.eval()
    train_features = list()
    train_labels = list()
    test_features = list()
    test_labels = list()
    with torch.no_grad():
        for img, label in train_loader:
            # TODO: extract train features and append them and labels to the list           
            train_features.append(vgg(img).squeeze().cpu().detach().numpy())
            train_labels.append(label.squeeze().cpu().detach().numpy())

        for img, label in test_loader:
            # TODO: extract test features and append them and labels to the list
            test_features.append(vgg(img).squeeze().cpu().detach().numpy())
            test_labels.append(label.squeeze().cpu().detach().numpy())


    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)

    # TODO: PCA decomposition
    #min(n_samples, n_features)
    pca = PCA(n_components = 15)

    pca_train_features = []

    for feature in train_features:
        pca_train_features.append(pca.fit_transform([feature]))

    pca_test_features = []

    for feature in test_dataset :
        pca_test_features.append(pca.transform([feature]))

    # TODO: train svm on train features and test on test features
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    clf.fit(pca_train_features, train_labels)

    Y_predict = clf.predict(pca_test_features)
    # print metrics
    print(metrics.classification_report(test_labels, Y_predict))

        # Return results
    sucess = 0
    total_labels = 0
    acc = []
    for i in range(len(test_labels)) :
        total_labels += 1
        if test_labels[i] == Y_predict[i] :
            sucess = sucess + 1

    acc.append(sucess/total_labels)
    
    print("Best nb_components is: ",acc.index(np.max(acc)))
    print("Accuracy =  ",np.max(acc)*100,"%")


if __name__ == "__main__":
    path_mmu = "mmu_iris"# TODO: insert path to the mmu dataset
    main(path_mmu)
