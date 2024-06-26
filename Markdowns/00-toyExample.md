
---
title: "ToyExample"
author: "Kyrillos Ishak"
date: "`r Sys.Date()`"
output: 
  html_notebook:
    pandoc_args: [
      "--number-offset=1,0"
      ]
    fig_caption: true
    number_sections: yes
    toc: yes
    toc_depth: 3
---

:::{.cell}
# Toy Example
:::
:::{.cell}
## **1. Introduction**

Machine learning pipelines, like any code, are susceptible to errors. One such error, data leakage between training and testing data, can inflate a model's apparent accuracy during evaluation. This can lead to deploying poor-performing models in production. Leakage often occurs unintentionally due to poor practices, and detecting it manually can be difficult. While encountering duplicate data in training and testing sets might seem like an obvious oversight, it's surprisingly common. [Ref](https://www.cs.cmu.edu/~ckaestne/pdf/ase22.pdf)
:::

:::{.cell}
✨ The objective of this notebook is :

* Understand the concept of data leakage, specifically focusing on duplicate data leakage.
* Create synthetic data with intentional duplicates to illustrate the concept.
* Detect and mitigate duplicate data leakage in a real-world dataset (CIFAR-100).
* Train machine learning models and evaluate their performance before and after removing duplicates.
* Critically analyze the impact of duplicate data leakage on model performance.
:::

:::{.cell}
🔍 In this notebook, we will explore:
1. A toy example with 🔴synthetic data🔴 to illustrate duplicate data leakage.
2. A real-world example using the 🔴CIFAR-100🔴 dataset.
:::

:::{.cell}
## **2. A toy example with  synthetic data  to illustrate duplicate data leakage**.
:::

:::{.cell}
🕵‍♀ Duplicates data leakage: The most obvious case is when test data is directly fed into training or hyperparameter tuning. A more subtle form, called overlap leakage, arises during data augmentation or oversampling so we will present this kind of leakages

1. Real duplicates
2. Duplicates made because of oversampling
:::

:::{.cell}
### 2.1 Real duplicates
:::

:::{.cell}
**Example :**


```python
dataset1 = {
    'feature1': [1, 2, 3, 4],
    'feature2': ['A', 'B', 'C', 'D'],
    'target': [0, 1, 0, 1]
}

dataset2 = {
    'feature1': [5, 6, 7, 4],
    'feature2': ['E', 'F', 'G', 'D'],
    'target': [0, 1, 0, 1]
}
df1 = pd.DataFrame(dataset1)
df2 = pd.DataFrame(dataset2)

df = pd.concat([df1, df2], ignore_index=True)

# Separate features and target
X = df[['feature1', 'feature2']]
y = df['target']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

```
:::

:::{.cell}
##### 🤔 Think Does this has any issue?
:::

:::{.cell}
YES, Data leakage can occur when `Frankenstein datasets` are used in machine learning scenarios. In the provided example, combining `dataset1` and `dataset2` into `df` introduces overlap in `feature1` and `feature2` values. This overlap means that the model might inadvertently learn patterns from data that it will later encounter in the test set, leading to overly optimistic performance metrics during evaluation.

Some practitioners intentionally create Frankenstein datasets to augment training data, believing it enhances model robustness by exposing it to diverse examples from multiple sources. This approach aims to mitigate bias and improve generalization, especially when individual datasets may be limited in scope or quality. However, careful handling is crucial to prevent unintended data leakage, ensuring that models learn and generalize effectively across varied but independent data points.

- Check COVIDx dataset for [example](https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-021-00307-0/MediaObjects/42256_2021_307_MOESM1_ESM.pdf).

:::

:::{.cell}
#### 💡 Solution :
:::

:::{.cell}
Before concatenating `dataset1` and `dataset2`, we should ensure that there are no overlapping data points between them. This can be done by checking for duplicates based on the features that we consider important (`feature1` and `feature2`).



```python
dataset1 = {
    'feature1': [1, 2, 3, 4],
    'feature2': ['A', 'B', 'C', 'D'],
    'target': [0, 1, 0, 1]
}

dataset2 = {
    'feature1': [5, 6, 7, 4],
    'feature2': ['E', 'F', 'G', 'D'],
    'target': [0, 1, 0, 1]
}
df1 = pd.DataFrame(dataset1)
df2 = pd.DataFrame(dataset2)

df = pd.concat([df1, df2], ignore_index=True)

# Dropping duplicates
df = df.drop_duplicates(subset=['feature1', 'feature2'])

# Separate features and target
X = df[['feature1', 'feature2']]
y = df['target']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

We can use many techniques if we want to find duplicates in :

1. **Photos** :
  *   Pixel-by-pixel cosine similarity
  *   Md5 hashing
  *   Comparing image embeddings

2. **Classical ML data**:
  *   Hashing
  *   `drop_duplicates` function in dataframe

3. **Texts** :
  * Identifying duplicates in text can be complex and varies based on the application. [Read more about duplicate detection in texts](https://arxiv.org/pdf/2107.06499).

:::

:::{.cell}
### 2.2 Duplicates made because of oversampling
:::
:::{.cell}
**Example :**


``` python
X_new , y_new = SMOTE().fit_resample (X , y )
X_train , X_test , y_train , y_test = train_test_split ( X_new , y_new , test_size =0.2 , random_state =42)
rf = RandomForestClassifier().fit ( X_train , y_train )
rf . predict ( X_test )
```
:::

:::{.cell}
##### 🤔 Think Does this has any issue?
:::

:::{.cell}
YES, Oversampling with SMOTE is performed **before** the train/test split. This creates a problem because the synthetic data generated by SMOTE are based on the original dataset's samples. When you split the data afterward, there is a high chance that the synthetic data in the training set will have very similar counterparts (or even the same ones) in the test set. This causes data leakage, where information from the training set influences the test set, leading to overly optimistic performance estimates.
:::

:::{.cell}
#### 💡 Solution :
:::

:::{.cell}
To avoid this issue, you should perform the train/test split before applying SMOTE. This ensures that the synthetic data generation only affects the training set, keeping the test set independent and truly representative of unseen data.


We can edit the code to :

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
rf = RandomForestClassifier().fit(X_train_resampled, y_train_resampled)
predictions = rf.predict(X_test)

```
:::

:::{.cell}
### 2.3 Excersise
:::
:::{.cell}

Identify the data leakage that caused by duplicates
``` python
import pandas as pd
from sklearn . feature_selection import SelectPercentile, chi2
from sklearn . model_selection import LinearRegression ,Ridge

X_0 , y = load_data ()

select = SelectPercentile( chi2 , percentile =50)
select.fit (X_0)
X = select.transform ( X_0 )

X_train , y_train , X_test , y_test = train_test_split (X ,y)
lr = LinearRegression ()
lr.fit ( X_train , y_train )
lr_score = lr.score ( X_test , y_test )

ridge = Ridge ()
ridge.fit (X , y)
ridge_score = ridge.score (X_test , y_test)
final_model = lr if lr_score > ridge_score else ridge
```
:::
:::{.cell}
## **3. A real-world example using the CIFAR-100 dataset**
:::
:::{.cell}
The CIFAR-100 dataset consists of 60,000 32x32 color images in 100 classes, with 600 images per class. There are 50,000 training images and 10,000 test images. It has been [discovered](https://arxiv.org/pdf/1902.00423) that CIFAR-100 dataset has 3 kinds of duplicates We will use CIFAR-100 to demonstrate how duplicates in the dataset can lead to data leakage and affect model performance.

The [paper](https://arxiv.org/pdf/1902.00423) introduces a fixed version of the CIFAR dataset called the “fair CIFAR” (ciFAIR) dataset, which classifies images into the following categories:

1. **Exact Duplicate** Almost all pixels in the two images are
approximately identical.
2. **Near-Duplicate** The content of the images is exactly the
same, i.e., both originated from the same camera shot.
However, different post-processing might have been
applied to this original scene, e.g., color shifts, translations, scaling etc.
3. **Very Similar** The contents of the two images are different,
but highly similar, so that the difference can only be
spotted at the second glance.
4. **Different** The pair does not belong to any other category.

We will use PyTorch for this tutorial. Not familiar with PyTorch [check](https://www.tutorialspoint.com/pytorch/index.htm)

---
:::

:::{.cell}
We will start by importing the required modules
:::


:::{.cell .code}
```
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from torchvision.models import resnet18
import warnings
warnings.filterwarnings("ignore")
```
:::

:::{.cell}
Then we will download the original `CIFAR-100` dataset from `torchvision.datasets` and download the `ciFAIR` dataset from the official github repo.
:::

:::{.cell .code}
```
# We will create a class for the ciFAIR dataset
class ciFAIR100(torchvision.datasets.CIFAR100):
    base_folder = 'ciFAIR-100'
    url = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-100.zip'
    filename = 'ciFAIR-100.zip'
    tgz_md5 = 'ddc236ab4b12eeb8b20b952614861a33'
    test_list = [
        ['test', '8130dae8d6fc6a436437f0ebdb801df1'],
    ]

# Data augmentation and normalization for training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
])


# # Downloading CIFAR-100 dataset to the 'data' directory and loading it
train_data = CIFAR100(download=True, root="./data", transform=transform_train)
test_data = CIFAR100(root="./data", train=False, transform=transform_test)

# # Downloading ciFAIR-100 dataset to the 'data_fixed' directory and loading it
train_data_fixed = ciFAIR100(download=True, root="./data_fixed", transform=transform_train)
test_data_fixed = ciFAIR100(root="./data_fixed", train=False, transform=transform_test)
```
:::

:::{.cell}
Exploring the duplicate images
:::

:::{.cell .code}
```
# Downloading duplicate image information from the ciFAIR repository
!wget https://github.com/cvjena/cifair/raw/master/meta/duplicates_cifar100.csv
!wget https://github.com/cvjena/cifair/raw/master/meta/duplicates_cifar100_test.csv

# Reading the CSV file containing duplicate information for train/test splits and test split
train_test_duplicates = pd.read_csv('duplicates_cifar100.csv')
# Reading the CSV file containing duplicate information for test splits itself (test split have duplicates)
test_duplicates = pd.read_csv('duplicates_cifar100_test.csv')
```
:::

:::{.cell .code}
```
# run train_test_duplicates and see the similarity between each pair that classified as duplicates
```
:::

:::{.cell .code}
```
# Choosing random samples from the duplicates for visualization
l = random.sample(range(0, 890), 5)

# plotting the images
for x in l:
  if train_test_duplicates['Judgment'][x] == 0:
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(test_data.data[train_test_duplicates['TestID'][x]])
    axes[1].imshow(train_data.data[train_test_duplicates['TrainID'][x]])
    fig.suptitle('Real Duplicates')
    plt.show()
  elif train_test_duplicates['Judgment'][x] == 1:
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(test_data.data[train_test_duplicates['TestID'][x]])
    axes[1].imshow(train_data.data[train_test_duplicates['TrainID'][x]])
    fig.suptitle('Near Duplicate')
    plt.show()
  elif train_test_duplicates['Judgment'][x] == 2:
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(test_data.data[train_test_duplicates['TestID'][x]])
    axes[1].imshow(train_data.data[train_test_duplicates['TrainID'][x]])
    fig.suptitle('Very similar')
    plt.show()
```
:::

:::{.cell .code}
```
# Choosing random samples from the duplicates for visualization
l = random.sample(range(0, 103), 5)

# plotting the images
for x in l:
  if test_duplicates['Judgment'][x] == 0:
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(test_data.data[test_duplicates['TestID'][x]])
    axes[1].imshow(test_data.data[test_duplicates['TrainID'][x]])
    fig.suptitle('Real Duplicates')
    plt.show()
  elif test_duplicates['Judgment'][x] == 1:
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(test_data.data[test_duplicates['TestID'][x]])
    axes[1].imshow(test_data.data[test_duplicates['TrainID'][x]])
    fig.suptitle('Near Duplicate')
    plt.show()
  elif test_duplicates['Judgment'][x] == 2:
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(test_data.data[test_duplicates['TestID'][x]])
    axes[1].imshow(test_data.data[test_duplicates['TrainID'][x]])
    fig.suptitle('Very similar')
    plt.show()
```
:::

:::{.cell}
Then we will start to train a ResNet CNN model using pyTorch on the original CIFAR dataset and asses its performance on the test split of the original dataset and the ciFAIR dataset test splits

Note : both CIFAR-100 and ciFAIR 🔴have the same🔴 train data split the only difference is in the test set
:::

:::{.cell}
Currently will use this hyperparameters
:::

:::{.cell .code}
```
# Hyperparameters
batch_size = 128
num_epochs = 50
learning_rate = 0.01
momentum = 0.9
weight_decay = 5e-4
```
:::

:::{.cell}
First we will use ResNet model without any pre-training
:::

:::{.cell .code}
```
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet model and modify the final layer for CIFAR-100
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 100)  # CIFAR-100 has 100 classes
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
# This helps in fine-tuning the learning rate as training progresses, often leading to better performance.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) 
```
:::

:::{.cell}
Then we will define the train/test functions
:::

:::{.cell .code}
```
# Training function
def train(epoch, trainloader):
    # Sets the model to training mode. IMP for layers (dropout and batchnorm) that behave differently in train/eval mode.
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # Move the input data and labels to the specified device (CPU or GPU).
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

# Testing function
def test(testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
```
:::

:::{.cell}
Load `PyTorch` data loader  
:::

:::{.cell .code}
```
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

testloader_fixed = DataLoader(test_data_fixed, batch_size=batch_size, shuffle=False, num_workers=2)
```
:::

:::{.cell .code}
```
# Training loop
for epoch in range(num_epochs):
    train(epoch,trainloader)
    scheduler.step()
print('Finished Training')
# Testing
print('Testing with original CIFAR testset')
test(testloader)
print('Testing with ciFAIR testset')
test(testloader_fixed)
```
:::

:::{.cell}
#### ❔ Why do you think the test set of the original split performs better than the test set of the ciFAIR dataset?
:::

:::{.cell}
The model was trained on images in the original test set, so it performs very well on images it has seen before. However, it performs poorly on the ciFAIR test set, which contains images it has not seen before.
:::
