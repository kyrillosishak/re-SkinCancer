
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

::: {.cell .markdown}

# Toy Example
:::

::: {.cell .markdown}
## **1. Introduction**

Machine learning pipelines, like any code, are susceptible to errors. One such error, data leakage between training and testing data, can inflate a model's apparent accuracy during evaluation. This can lead to deploying poor-performing models in production. Leakage often occurs unintentionally due to poor practices, and detecting it manually can be difficult. While encountering duplicate data in training and testing sets might seem like an obvious oversight, it's surprisingly common. [Reference](https://www.cs.cmu.edu/~ckaestne/pdf/ase22.pdf)

✨ The objective of this notebook is :

* Understand the concept of data leakage, specifically focusing on duplicate data leakage.
* Create synthetic data with intentional duplicates to illustrate the concept.
* Detect and mitigate duplicate data leakage in a real-world dataset (CIFAR-100).
* Train machine learning models and evaluate their performance before and after removing duplicates.
* Critically analyze the impact of duplicate data leakage on model performance.

🔍 In this notebook, we will explore:
1. A toy example with <span style="color:#973131">synthetic data</span> to illustrate duplicate data leakage.
2. A real-world example using the <span style="color:#973131">CIFAR-100</span> dataset.

---

**Duplicates :** Duplicates in a dataset refer to instances that are either exactly the same or very similar to each other. They can arise due to various reasons during data collection and preprocessing. Duplicates can lead to overly optimistic evaluations of a machine learning model's performance. This happens because the model might end up training on and testing against highly similar or identical instances, giving a false sense of its generalization capability.

<img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/3kindOfDuplicates.png">

*In the context of image data, duplicates can be categorized into three types:*

- <span style="color:#80AF81">Exact Duplicate</span> Almost all pixels in the two images are
approximately identical.
- <span style="color:#80AF81">Near-Duplicate</span> The content of the images is exactly the
same, i.e., both originated from the same camera shot.
However, different post-processing might have been
applied to this original scene, e.g., color shifts, translations, scaling etc.
- <span style="color:#80AF81">Very Similar</span> The contents of the two images are different,
but highly similar, so that the difference can only be
spotted at the second glance.

---
**How Duplicate Samples Might End Up in Data?**



1.   <span style="color:#E68369">Reasons specific to the data and how it was collected</span> :
     -   *Scenario 1* : When training an email classifier for an academic department. In this scenario, data is collected from all students within the department. Since these students are part of the same academic environment, they receive a significant number of common emails, such as departmental announcements, course notifications, and event reminders.
     -   *Scenario 2* : When training a chatbot for customer support, data is often gathered from various interactions with customers. Many customers might ask similar questions or encounter the same issues, resulting in repeated dialogue patterns within the dataset.
     -   *Scenario 3* : When training a model to classify news articles, data might be sourced from various news outlets. Major news stories are often covered by multiple sources, and syndicated articles or press releases can appear across different outlets, leading to duplicate samples.
     -    *Scenario 4* : Speech recognition datasets often include recordings of common phrases or sentences. If data is collected from multiple participants who are asked to repeat specific phrases, duplicates are inevitable.

2.  <span style="color:#E68369">Frankenstein datasets</span> :

    <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/covidx.png" />

     A "Frankenstein dataset" refers to a dataset composed of multiple other public datasets. Some practitioners intentionally create Frankenstein datasets to augment training data, as it enhances model robustness by exposing it to diverse examples from multiple sources. This approach aims to mitigate bias and improve generalization, especially when individual datasets may be limited in scope or quality. However, unintentionally, duplicates can arise within these datasets when one dataset includes information that duplicates or overlaps with data already included from another source.
    
      *For example, consider the COVIDx dataset, which incorporates three main datasets: COHEN, RSNA, and CHOWDHURY. CHOWDHURY itself includes the COHEN dataset, this overlap introduces duplicates within the Frankenstein dataset.*


3.  <span style="color:#E68369">Using LLM to generate data for training</span> :

       <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/generateDataWithLLM.png" width="200" height="200" />
    
      * The process of creating a dataset using large language models (LLMs) can be formalized within a unified framework, leveraging the model's ability to generate text based on given conditions. Assume we have an LLM, ( 𝜧 ), which is pre-trained to generate a sequence \( x = [x_1, x_2, …, x_n] \) by recursively sampling tokens conditioned on previous tokens. The goal is to create samples \((x, y)\) where \( y \) is a label from a defined label space \( Y \). This can involve using label-descriptive prompts \( W_y \), in-domain unlabeled examples \( x_u \) from \( D_U \), or a small number of labeled examples \((x_l, y_l)\) from \( D_L \) along with their explanations. Two broad strategies are used: employing the LLM as a labeler to assign labels to unlabeled data, or using the LLM as a generator to produce data samples conditioned on labels. The LLM can be prompted with few-shot examples or descriptive prompts to create coherent and diverse datasets.

     * Duplicates can arise in datasets created using LLMs due to several factors. First, limited prompt variety can lead the model to generate similar or identical outputs, especially if the same prompt is used repeatedly. Additionally, the training data of the LLM may contain repetitive patterns, causing the model to reproduce these during generation. Randomness in the text generation process, particularly with low diversity settings, can also result in repetitive sequences. Furthermore, using in-domain unlabeled examples or few-shot examples that are not diverse enough may limit the variability of the generated samples. To mitigate duplicates, it is essential to employ strategies such as diverse prompt design, careful sampling, and post-generation deduplication techniques.
  
     Want to know more? [Read the full paper](https://arxiv.org/pdf/2310.20111)


4. <span style="color:#E68369">Data augmentation or oversampling before split</span> :

      Data augmentation involves creating modified versions of existing data to increase the dataset size and diversity, while oversampling involves replicating data points to balance class distributions. If these techniques are applied before the dataset is split, the augmented or replicated samples can be distributed across the different splits. As a result, the same data point, or its augmented version, might appear in both the training and validation or test sets. This overlap can cause the model to have an unfair advantage, as it may encounter the same or very similar data during both training and evaluation phases. This can inflate performance metrics, giving a false sense of the model's generalization capabilities.

:::

:::{.cell .markdown}
## **2. A toy example with  synthetic data  to illustrate duplicate data leakage**.
:::

:::{.cell .markdown}

🕵‍♀ Duplicates data leakage: The most obvious case is when test data is directly fed into training or hyperparameter tuning. A more subtle form, called overlap leakage, arises during data augmentation or oversampling so we will present this kind of leakages

*  Duplicates made because of oversampling

:::

:::{.cell .markdown}
### 2.1 Example 1

Illustration of the example: Here we demonstrate that splitting after oversampling can lead to duplicate data leakage, which can bias the evaluation of our model.
:::

:::{.cell .code}
```python
# Importing necessary modules
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```
:::

:::{.cell .code}
```python
def generate_data(n_samples=100, n_features=1, noise_level=0.2):
    """
    Generates synthetic data and casts it to the known distribution.
    """
    # Generate random data
    X = np.random.rand(n_samples, n_features)
    # Known coefficients
    coef = np.random.randn(n_features)
    # Generate target variable
    y = X @ coef + np.random.randn(n_samples) * noise_level
    return X, y, coef

def sample_with_replacement(X, y, n_samples):
    """
    Make Oversampling to the data to make duplicates data leakage
    """
    # Sample with replacement
    indices = np.random.choice(np.arange(len(X)), size=n_samples, replace=True)
    return X[indices], y[indices], indices

def add_noise(X, noise_level=0.1):
    """
    Adds noise to the features.
    """
    return X + np.random.randn(*X.shape) * noise_level

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and returns the MSE.
    """
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)
```
:::

:::{.cell .code}
```python
# Generate data with noise
n_samples_initial = 100
n_samples_total = 150
noise_level = 0.1  # Change this value to experiment with different noise levels

# Generate initial data
X, y, coef = generate_data(n_samples=n_samples_initial, noise_level=noise_level)
# Apply Oversampling
X_sampled, y_sampled, indices = sample_with_replacement(X, y, n_samples=n_samples_total)
num_duplicates = len(indices) - len(np.unique(indices))

# Add noise to features
X_noisy = add_noise(X_sampled, noise_level=noise_level)

# Split into training and test sets after oversampling
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y_sampled, test_size=0.2)
# Train model
model = LinearRegression().fit(X_train, y_train)


# Evaluate on bad test set
mse_bad_test = evaluate_model(model, X_test, y_test)

# Generate a new "clean" test set (for comparison)
X_clean_test, y_clean_test, dup = generate_data(n_samples=len(y_test), noise_level=noise_level)

# Evaluate on the "clean" test set
mse_clean_test = evaluate_model(model, X_clean_test, y_clean_test)

# Print results
print(f"Number of duplicates in sampled data: {num_duplicates}")
print(f"MSE on 'bad' test set: {mse_bad_test}")
print(f"MSE on 'clean' test set: {mse_clean_test}")
```
:::

:::{.cell .markdown}
#### 🤔 Why MSE on `bad` test set is `always` lower than MSE on `clean` test set?
:::

:::{.cell .markdown}

Because the 'bad' test set has data leakage which can cause overly optimistic results.

:::

:::{.cell .markdown}

### 2.2 Example 2
:::

:::{.cell .markdown}

**Example :**


``` python
X_new , y_new = SMOTE().fit_resample(X ,y)
X_train , X_test , y_train , y_test = train_test_split( X_new , y_new , test_size =0.2 , random_state =42)
rf = RandomForestClassifier().fit( X_train , y_train )
rf.predict( X_test )
```
:::

:::{.cell .markdown}

##### 🤔 Think Does this has any issue?
:::

:::{.cell .markdown}

YES, Oversampling with SMOTE is performed **before** the train/test split. This creates a problem because the synthetic data generated by SMOTE are based on the original dataset's samples. When you split the data afterward, there is a high chance that the synthetic data in the training set will have very similar counterparts (or even the same ones) in the test set. This causes data leakage, where information from the training set influences the test set, leading to overly optimistic performance estimates.

:::

:::{.cell .markdown}
#### 💡 Solution :
:::

:::{.cell .markdown}

To avoid this issue, you should perform the train/test split before applying SMOTE. This ensures that the synthetic data generation only affects the training set, keeping the test set independent and truly representative of unseen data.


We can edit the code to :

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
rf = RandomForestClassifier().fit(X_train_resampled, y_train_resampled)
predictions = rf.predict(X_test)

```

:::

:::{.cell .markdown}
### 2.3 Excersise
:::

:::{.cell .markdown}


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

:::{.cell .markdown}

## **3. A real-world example using the CIFAR-100 dataset**

:::

:::{.cell .markdown}

The CIFAR-100 dataset consists of 60,000 32x32 color images in 100 classes, with 600 images per class. There are 50,000 training images and 10,000 test images. It has been [discovered](https://arxiv.org/pdf/1902.00423) that CIFAR-100 dataset has 3 kinds of duplicates We will use CIFAR-100 to demonstrate how duplicates in the dataset can lead to data leakage and affect model performance.

*We will use PyTorch for this tutorial. Not familiar with PyTorch [check](https://www.tutorialspoint.com/pytorch/index.htm)*

---

:::

:::{.cell .markdown}

We will start by importing the required modules

:::

:::{.cell .code}
```python
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

:::{.cell .markdown}

Then we will download the original `CIFAR-100` dataset from `torchvision.datasets` and download the `ciFAIR` dataset from the official github repo.

:::

:::{.cell .code}
```python
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

:::{.cell .markdown}

Exploring the duplicate images

:::

:::{.cell .code}
```python
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
```python
# Number of duplicates in each class :
label = {
    0: 'Real-duplicates',
    1: 'Near-duplicates',
    2: 'very similar'
}
print("duplicates in train/test splits\n")
labels, counts = np.unique(train_test_duplicates['Judgment'], return_counts=True)
for l, count in zip(labels, counts):
    print(f"Label: {label[l]}, Count: {count}")

print("\nduplicates in test split\n")
labels, counts = np.unique(test_duplicates['Judgment'], return_counts=True)
for l, count in zip(labels, counts):
    print(f"Label: {label[l]}, Count: {count}")
```
:::

:::{.cell .code}
```python
# run train_test_duplicates and see the similarity between each pair that classified as duplicates
```
:::

:::{.cell .code}
```python
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
```python
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

:::{.cell .markdown}

Then we will start to train a ResNet CNN model using pyTorch on the original CIFAR dataset and asses its performance on the test split of the original dataset and the ciFAIR dataset test splits

Note : both CIFAR-100 and ciFAIR <span style="color:red">have the same</span> train data split the only difference is in the test set

:::

:::{.cell .markdown}

Currently will use this hyperparameters

:::

:::{.cell .code}
```python
# Hyperparameters
batch_size = 128
num_epochs = 50
learning_rate = 0.01
momentum = 0.9
weight_decay = 5e-4
```
:::

:::{.cell .markdown}

First we will use ResNet model without any pre-training

:::

:::{.cell .code}
```python
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

:::{.cell .markdown}

Then we will define the train/test functions

:::

:::{.cell .code}
```python

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

:::{.cell .markdown}

Load `PyTorch` data loader  

:::

:::{.cell .code}
```python
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

testloader_fixed = DataLoader(test_data_fixed, batch_size=batch_size, shuffle=False, num_workers=2)
```
:::

:::{.cell .code}
```python
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

:::{.cell .markdown}

#### ❔ Why do you think the test set of the original split performs better than the test set of the ciFAIR dataset?

:::

:::{.cell .markdown}

The model was trained on images in the original test set, so it performs very well on images it has seen before. However, it performs poorly on the ciFAIR test set, which contains images it has not seen before.

:::

:::{.cell .markdown}

## **4. How to measure duplicates?**

In this section, we will discuss how to find duplicates in a dataset based on the following types:

1.   Images
2.   Texts
3.   General data

---

### **1. How to mitigate duplicates in Images:**
*Finding duplicate images can be challenging due to variations in size, format, and slight alterations. Here are some common methods:*

  * <span style="color:#96C9F4">Hashing</span> : Compute hash values for images and compare them. Techniques like MD5, SHA-1, or perceptual hashing (such as pHash) can identify identical or visually similar images. A `perceptual hash` is a fingerprint of a multimedia file derived from various features from its content. Unlike cryptographic hash functions which rely on the avalanche effect of small changes in input leading to drastic changes in the output, perceptual hashes are "close" to one another if the features are similar. You can use [this tool](https://github.com/coenm/ImageHash).
    
    <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/similarityWithHashing.png">
  * <span style="color:#96C9F4">Pixel-by-pixel cosine similarity</span> : It is a method used to detect duplicate or near-duplicate images by comparing the pixel values of two images. This technique measures the cosine of the angle between two vectors, which in this context are the pixel value arrays of the images. A threshold value can be set to determine if the images are considered duplicates. For example, a threshold of 0.95 might be used to account for minor variations while still recognizing duplicates. If you want to learn more use this [example](https://proceedings.mlr.press/v184/silva22a/silva22a.pdf) that uses pixel-by-pixel similarity to detect similarities in COVIDx dataset.
  ```python
  def cosine_similarity(vec1, vec2):
      dot_product = np.dot(vec1, vec2)
      magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
      if not magnitude:
          return 0
      return dot_product / magnitude

  similarity = cosine_similarity(img1.flatten(), img2.flatten())
  ```

  * <span style="color:#96C9F4">Similarities of image embeddings</span> : Using image embeddings to detect duplicates involves leveraging deep learning models to extract high-level features from images, converting them into dense vectors (embeddings). First, a pre-trained convolutional neural network (CNN) such as VGG, ResNet, or Inception is used to generate embeddings for each image. These models, trained on large datasets. Next, the cosine similarity or Euclidean distance between the embeddings of different images is calculated. Cosine similarity measures the cosine of the angle between two vectors, indicating their similarity. Images with high cosine similarity are considered duplicates or near-duplicates. [Reference](https://proceedings.mlr.press/v184/silva22a/silva22a.pdf)

---

### **2. How to mitigate duplicates in Texts:**

*Duplicate detection in text data involves identifying identical or nearly identical pieces of text. Some methods include:*
    <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/textduplicates.png">

  * <span style="color:#96C9F4">Exact Substring Duplication</span> : Due to the diversity of human language, it is uncommon for the same idea to be expressed identically in multiple documents unless one is derived from the other or both quote a shared source. When two text examples share a sufficiently long substring. Based on statistical analyses, a minimum matching substring length of 50 tokens is selected.
    * **Suffix Arrays**: Exact substring matching is computationally prohibitive with naive (quadratic) all-pair matching. To improve efficiency, all examples in the dataset are concatenated into a giant sequence, from which a Suffix Array is constructed. A suffix array is a representation of a suffix tree that can be constructed in linear time and allows efficient computation of many substring queries. 
    
        *For example, the suffixes of the sequence “banana” are (“banana”, “anana”, “nana” “ana”, “na”, “a”) and so the suffix array is the sequence (6 4 2 1 5 3).*
    *  **Substring matching**: Identify Duplicates by scanning the suffix array, repeated sequences can be identified as adjacent indices in the array. If two sequences share a common prefix of at least the threshold length, they are recorded as duplicates.
  *  <span style="color:#96C9F4">Approximate Matching with MinHash</span> : Exact substring matching may not be sufficient for all cases, especially with web crawl text where documents might be identical except for minor variations. For such cases, approximate deduplication using MinHash is effective. MinHash is an approximate matching algorithm widely used in large-scale deduplication tasks. MinHash approximates the Jaccard Index, which measures the similarity between two sets of n-grams derived from documents. The algorithm uses hash functions to create document signatures by sorting n-grams and keeping only the smallest hashed n-grams.

      * Generate N-grams
      * Use hash functions to generate MinHash signatures by selecting the smallest hashed n-grams.
      * Calculate the probability that two documents are potential matches based on their MinHash signatures.


*Want to learn more about deduplication? check [Reference](https://arxiv.org/pdf/2107.06499)*

---
  
### **3. How to mitigate duplicates in General Data:**

*For general data, which can include structured data in tables, finding duplicates often involves comparing multiple columns.*

* Comparing entire records for exact matches.
* Clustering grouping similar records together based on features.
* Assigning probabilities to different fields and calculating an overall similarity score.


:::
