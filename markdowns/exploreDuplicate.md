
::: {.cell .markdown}

# Toy Example
:::

::: {.cell .markdown}
## Introduction
:::

::: {.cell .markdown}

Machine learning pipelines, like any code, are susceptible to errors. One such error, data leakage between training and testing data, can inflate a model's apparent accuracy during evaluation. This can lead to deploying poor-performing models in production. Leakage often occurs unintentionally due to poor practices, and detecting it manually can be difficult. While encountering duplicate data in training and testing sets might seem like an obvious oversight, it's surprisingly common [1].

✨ The objective of this notebook is :

* Understand the concept of data leakage, specifically focusing on duplicate data leakage.
* Create synthetic data with intentional duplicates to illustrate the concept.
* Detect and mitigate duplicate data leakage in a real-world dataset (CIFAR-100).
* Train machine learning models and evaluate their performance before and after removing duplicates.
* Critically analyze the impact of duplicate data leakage on model performance.

🔍 In this notebook, we will explore:

1. A toy example with **synthetic data** to illustrate duplicate data leakage.
2. A real-world example using the **CIFAR-100** dataset.

---

**Duplicates :** Duplicates in a dataset refer to instances that are either exactly the same or very similar to each other. They can arise due to various reasons during data collection and preprocessing. Duplicates can lead to overly optimistic evaluations of a machine learning model's performance. This happens because the model might end up training on and testing against highly similar or identical instances, giving a false sense of its generalization capability.

<figure>
    <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/3kindOfDuplicates.png" alt="duplicates">
</figure>


This image from [2]. 

*In the context of image data, the paper "Do we train on test data? Purging CIFAR of near-duplicates" [2] describes three types of duplication:*

> - **Exact Duplicate** Almost all pixels in the two images are approximately identical.
> - **Near-Duplicate** The content of the images is exactly the same, i.e., both originated from the same camera shot. However, different post-processing might have been applied to this original scene, e.g., color shifts, translations, scaling etc.
> - **Very Similar** The contents of the two images are different, but highly similar, so that the difference can only be spotted at the second glance.

_In many situations, the distinction between `near-duplicate` and `very similar` isn't crucial. **Near-duplicate** often serves as a more general term encompassing a wide range of image similarities.  This category can include images derived from the same source with minor edits, but also extends to pictures of the same scene or object captured from different angles, cameras, or even screenshots._


---
**How Duplicate Samples Might End Up in Data?**

:::

:::{.cell .markdown}

**1. Reasons specific to the data and how it was collected** :
     
-   *Scenario 1* : When training an email classifier for an academic department. In this scenario, data is collected from all students within the department. Since these students are part of the same academic environment, they receive a significant number of common emails, such as departmental announcements, course notifications, and event reminders.
-   *Scenario 2* : When training a chatbot for customer support, data is often gathered from various interactions with customers. Many customers might ask similar questions or encounter the same issues, resulting in repeated dialogue patterns within the dataset.
-   *Scenario 3* : When training a model to classify news articles, data might be sourced from various news outlets. Major news stories are often covered by multiple sources, and syndicated articles or press releases can appear across different outlets, leading to duplicate samples.
-   *Scenario 4* : Speech recognition datasets often include recordings of common phrases or sentences. If data is collected from multiple participants who are asked to repeat specific phrases, duplicates are inevitable.
-   *Scenario 5* : Frankenstein datasets (A "Frankenstein dataset" refers to a dataset composed of multiple other public datasets. Some scientists intentionally create Frankenstein datasets to augment training data, as it enhances model robustness by exposing it to diverse examples from multiple sources. This approach aims to mitigate bias and improve generalization, especially when individual datasets may be limited in scope or quality. However, unintentionally, duplicates can arise within these datasets when one dataset includes information that duplicates or overlaps with data already included from another source.)

      <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/covidx.png" />
        
      *This image from survey [3].*	  
      
      *For example, consider the COVIDx dataset, which incorporates three main datasets: COHEN, RSNA, and CHOWDHURY. CHOWDHURY itself includes the COHEN dataset, this overlap might cause duplicates within the Frankenstein dataset [3].*
				  
:::


:::{.cell .markdown}

**2. Using LLM to generate data for training** :

Duplicates can arise in datasets created using LLMs due to several factors. First, limited prompt variety can lead the model to generate similar or identical outputs, especially if the same prompt is used repeatedly. Additionally, the training data of the LLM may contain repetitive patterns, causing the model to reproduce these during generation. Randomness in the text generation process, particularly with low diversity settings, can also result in repetitive sequences. Furthermore, using in-domain unlabeled examples or few-shot examples that are not diverse enough may limit the variability of the generated samples. To mitigate duplicates, it is essential to employ strategies such as diverse prompt design, careful sampling, and post-generation deduplication techniques.

*This example shows how using LLM for generating dataset for specific task (sentiment analysis in our case) can cause duplicates:* 

:::

:::{.cell .code}
```python
!pip -q install git+https://github.com/huggingface/transformers # need to install from github
!pip install -q datasets loralib sentencepiece
!pip -q install bitsandbytes accelerate xformers einops
```
:::


:::{.cell .code}
```python
import torch
import textwrap
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
def wrap_text(text, width=90): #preserve_newlines
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def generate(input_text, system_prompt="",max_length=512):
    prompt = f"""<s>[INST]{input_text}[/INST]"""
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    outputs = model.generate(**inputs,
                             eos_token_id = 2,
                             max_new_tokens=max_length,
                             pad_token_id=1,
                             temperature=0.1,
                             do_sample=True)
    text = tokenizer.batch_decode(outputs)[0]
    wrapped_text = wrap_text(text)
    print(wrapped_text)
    return wrapped_text
```
:::

:::{.cell .code}
```python
torch.set_default_device('cuda')
model = AutoModelForCausalLM.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", torch_dtype="auto")
```
:::

:::{.cell .code}
```python
generate('Generate a sentences expressing positive sentiment:', max_length=512)
```
:::

:::{.cell .markdown} 

**3. Data augmentation or oversampling before split** :

   Data augmentation involves creating modified versions of existing data to increase the dataset size and diversity, while oversampling involves replicating data points to balance class distributions. If these techniques are applied before the dataset is split, the augmented or replicated samples can be distributed across the different splits. As a result, the same data point, or its augmented version, might appear in both the training and validation or test sets. This overlap can cause the model to have an unfair advantage, as it may encounter the same or very similar data during both training and evaluation phases. This can inflate performance metrics, giving a false sense of the model's generalization capabilities.
    
   *In this example, we show how using augmentation before splitting can cause duplicate data leakage:*

:::

:::{.cell .code}
```python
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=2)

# Sample 100 images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Define augmentation
augment = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Convert to PIL images for augmentation
images_pil = [transforms.ToPILImage()(img) for img in images]

# Augment all of the images
num_augment = len(images_pil)
augmented_images = [(augment(img),idx) for idx, img in enumerate(images_pil[:num_augment])]
normal_images =  [(transforms.ToTensor()(img),idx) for idx, img in enumerate(images_pil)]
# Store indices with images
augmented_images_with_indices =  augmented_images + normal_images
train_images_with_indices, test_images_with_indices = train_test_split(augmented_images_with_indices, test_size=0.2, random_state=seed)

# Find duplicates between train and test sets
train_indices = set(idx for _, idx in train_images_with_indices)
test_indices = set(idx for _, idx in test_images_with_indices)
duplicates = train_indices.intersection(test_indices)

print(f"Duplicate indices between train and test sets: {duplicates}")

# Show duplicate images
fig, axes = plt.subplots(len(duplicates), 2, figsize=(10, 5 * len(duplicates)))
for i, idx in enumerate(duplicates):
    train_img = next(img for img, id_ in train_images_with_indices if id_ == idx)
    test_img = next(img for img, id_ in test_images_with_indices if id_ == idx)
    axes[i, 0].imshow(np.transpose(train_img.numpy(), (1, 2, 0)))
    axes[i, 0].set_title(f'Train Image Index: {idx}')
    axes[i, 0].axis('off')
    axes[i, 1].imshow(np.transpose(test_img.numpy(), (1, 2, 0)))
    axes[i, 1].set_title(f'Test Image Index: {idx}')
    axes[i, 1].axis('off')

plt.show()

```
:::

:::{.cell .markdown}
## A toy example with  synthetic data  to illustrate duplicate data leakage.
:::



:::{.cell .markdown}
### Example with accidental overlap between training and test set

:::

::: {.cell .markdown}

Illustration of the example: Here we demonstrate wrong data preprocessing that causes duplicate data leakage, which can bias the evaluation of our model.

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
```
:::

:::{.cell .code}
```python
# Generate data with noise
n_samples_initial = 100
n_samples_total = 150
noise_level = 0.3  # Change this value to experiment with different noise levels
np.random.seed(42)

# Generate initial data
X, y, coef = generate_data(n_samples=n_samples_initial, noise_level=noise_level)

# Sample with replacement
indices = np.random.choice(np.arange(len(X)), size=n_samples_total, replace=True)
X_sampled, y_sampled, indices = X[indices], y[indices], indices
num_duplicates = len(indices) - len(np.unique(indices))

# Add noise to features
X_noisy = X_sampled + np.random.randn(*X_sampled.shape) * noise_level

# Split into training and test sets according to the indices
# Use 80% for training and 20% for testing
train_size = int(0.8 * n_samples)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, y_train = X_noisy[:train_size], y_sampled[:train_size]
X_test, y_test = X_noisy[train_size:], y_sampled[train_size:]

# Calculate number of duplicates in test set that are also in train set
duplicates_in_test = len(set(train_indices) & set(test_indices))

# Train model
model = LinearRegression().fit(X_train, y_train)

# Evaluate on "bad" test set
y_pred = model.predict(X_test)
mse_bad_test = mean_squared_error(y_test, y_pred)

# Generate a new "clean" test set (for comparison)
X_clean_test, y_clean_test, _ = generate_data(n_samples=len(y_test), noise_level=noise_level)

# Evaluate on the "clean" test set
y_pred_clean = model.predict(X_clean_test)
mse_clean_test = mean_squared_error(y_clean_test, y_pred_clean)

# Print results
print(f"Number of duplicates in sampled data: {num_duplicates}")
print(f"Number of duplicates in test set that are also in train set: {duplicates_in_test}")
print(f"MSE on 'bad' test set: {mse_bad_test}")
print(f"MSE on 'clean' test set: {mse_clean_test}")
```
:::

:::{.cell .markdown}
#### 🤔 Why MSE on `bad` test set is lower than MSE on `clean` test set?
:::

:::{.cell .markdown}

Because the 'bad' test set has data leakage which can cause overly optimistic results.

*Note : sometimes the MSE on `bad` test data perform worse (higher MSE) than `clean` test data that can be as both the noisy and clean datasets are generated with some randomness, leading to variability in the MSE. Even though the clean set doesn't have duplicates, the particular noise added could make it slightly easier or harder to predict accurately compared to the noisy dataset.*


:::

:::{.cell .markdown}

### Example with incorrect oversampling
:::

:::{.cell .markdown}

**Example :**

:::

:::{.cell .code}
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Add noise to the data
noise = np.random.normal(0, 1, X.shape)
X_noisy = X + noise

# Convert to DataFrame for better visualization (optional)
df = pd.DataFrame(X_noisy)
df['target'] = y
```
:::

:::{.cell .code}
```python
# Incorrect implementation with data leakage
X_new, y_new = SMOTE().fit_resample(X_noisy, y)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

rf_incorrect = RandomForestClassifier(random_state=42).fit(X_train, y_train)
predictions_incorrect = rf_incorrect.predict(X_test)

accuracy_incorrect = accuracy_score(y_test, predictions_incorrect)
print(f"Incorrect Implementation Accuracy: {accuracy_incorrect:.4f}")
```
:::

:::{.cell .markdown}

##### 🤔 Think why this is a bad implementation?
:::

:::{.cell .markdown}

Oversampling with SMOTE is performed **before** the train/test split. This creates a problem because the synthetic data generated by SMOTE are based on the original dataset's samples. When you split the data afterward, there is a high chance that the synthetic data in the training set will have very similar counterparts (or even the same ones) in the test set. This causes data leakage, where information from the training set influences the test set, leading to overly optimistic performance estimates.

:::

:::{.cell .markdown}
#### 💡 Solution :
:::

:::{.cell .markdown}

To avoid this issue, you should perform the train/test split before applying SMOTE. This ensures that the synthetic data generation only affects the training set, keeping the test set independent and truly representative of unseen data.


We can edit the code to :

:::

:::{.cell .code}
```python
# Correct implementation without data leakage
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)

rf_correct = RandomForestClassifier(random_state=42).fit(X_train_resampled, y_train_resampled)
predictions_correct = rf_correct.predict(X_test)

accuracy_correct = accuracy_score(y_test, predictions_correct)
print(f"Correct Implementation Accuracy: {accuracy_correct:.4f}")
```
:::

:::{.cell .markdown}
### Exercise
:::

:::{.cell .markdown}

Identify the data leakage that caused by duplicates
``` python
import pandas as pd
from sklearn . feature_selection import SelectPercentile, chi2
from sklearn . model_selection import LinearRegression ,Ridge

X_0 , y = load_data ()

select = SelectPercentile( chi2 , percentile =50)
select.fit(X_0)
X = select.transform( X_0 )

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
### Exercise

Correct the data leakage that caused by duplicates
:::

:::{.cell .code}
``` python
import pandas as pd
from sklearn . feature_selection import SelectPercentile, chi2
from sklearn . model_selection import LinearRegression ,Ridge

X_0 , y = load_data ()

## Write your code here ##

# Use the same approach as above

#########################
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

## A real-world example using the CIFAR-100 dataset

:::

:::{.cell .markdown}

The CIFAR-100 dataset consists of 60,000 32x32 color images in 100 classes, with 600 images per class. There are 50,000 training images and 10,000 test images. It has been discovered [2] that CIFAR-100 dataset has 3 kinds of duplicates We will use CIFAR-100 to demonstrate how duplicates in the dataset can lead to data leakage and affect model performance.

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
# Data
categories = ['Dup.', 'Near-Dup.', 'Similar']
datasets = ['CIFAR-100', 'CIFAR-100', 'CIFAR-100']
training_counts = [39, 582, 270]
test_counts = [2, 72, 30]

# Positions of the bars
bar_width = 0.5
r1 = np.arange(len(categories))

# Create the figure and increase the figure size
plt.figure(figsize=(12, 7))

# Create the bars
plt.bar(r1, training_counts, color='#1f77b4', width=bar_width, label='duplicates in trainset appeared in testset')
plt.bar(r1, test_counts, color='#ff7f0e', bottom=training_counts, width=bar_width, label='duplicates in testset itself')

# Adding the text labels and titles
plt.xlabel('')
plt.ylabel('')
plt.xticks(r1, categories)
plt.legend()

# Add custom x-axis labels
positions = [(r1[0] + r1[2]) / 2]
dataset_labels = ['CIFAR-100']
for pos, label in zip(positions, dataset_labels):
    plt.text(pos, -50, label, ha='center')

plt.grid(axis='y')

# Setting x-ticks positions and labels
plt.xticks(r1, categories, rotation=0)

# Show the plot
plt.show()
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

Then we will start to train a ResNet CNN model using pyTorch on the original CIFAR dataset and assess its performance on the test split of the original dataset and the ciFAIR dataset test splits

Note : both CIFAR-100 and ciFAIR **have the same** train data split the only difference is in the test set

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

## How to measure duplicates?
:::

::: {.cell .markdown}

In this section, we will discuss how to find duplicates in a dataset based on the following types:

1.   Images
2.   Texts
3.   General data

---

:::

:::{.cell .markdown}

### How to mitigate duplicates in Images:

:::

::: {.cell .markdown}

*Finding duplicate images can be challenging due to variations in size, format, and slight alterations. Here are some common methods:*

* **Hashing** : Compute hash values for images and compare them. Techniques like MD5, SHA-1. This techniques can't detect Near-duplicate it can detect Exact-duplicates only.
   *Example :* 
    
   ```python
   import hashlib
   def calculate_sha256_hash(image_path):
       with open(image_path, "rb") as f:
           image_data = f.read()
           return hashlib.sha256(image_data).hexdigest()
   ```
        
  <table>
    <tr>
    <td>
      <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/dog.jpg" width="400" height="400"/>

    </td>
    <td>
      <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/dog_cropped.jpg" width="400" height="400" />

    </td>
    </tr>
    <tr>
    <td> Hash : 795d4344cdc8fa578b86d30622a8d935e237bcff406c4e5a1d6d17b568c73bfa</td>
    <td>Hash : 553f7d8da6736af2cc237f9abad4c1f35b2d26705e1b87527f04c62c48f018f4</td>
    </tr>
  </table>

  *As we can see hash cannot detect if the image is augmented (cropped in our case) but can detect exact duplicates.*
* **Pixel-by-pixel cosine similarity** : It is a method used to detect duplicate or near-duplicate images by comparing the pixel values of two images. This technique measures the cosine of the angle between two vectors, which in this context are the pixel value arrays of the images. A threshold value can be set to determine if the images are considered duplicates. For example, a threshold of 0.95 might be used to account for minor variations while still recognizing duplicates.
    
  ```python
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    
    similarity = cosine_similarity(img1.flatten(), img2.flatten())
    ```
  
* **Similarities of image embeddings** : Using image embeddings to detect duplicates involves leveraging deep learning models to extract high-level features from images, converting them into dense vectors (embeddings). First, a pre-trained convolutional neural network (CNN) such as VGG, ResNet, or Inception is used to generate embeddings for each image. These models, trained on large datasets. Next, the cosine similarity or Euclidean distance between the embeddings of different images is calculated. Cosine similarity measures the cosine of the angle between two vectors, indicating their similarity. Images with high cosine similarity are considered duplicates or near-duplicates. *Example showing comparing cosine similarities:*
  ```python
  import torch
  import torch.nn as nn
  import torchvision.transforms as transforms
  import torchvision.models as models
  import numpy as np
  from PIL import Image
  from sklearn.metrics.pairwise import cosine_similarity
  
  # Load pre-trained ResNet50 model
  model = models.resnet50(pretrained=True)
  model = nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
  
  # Set model to evaluation mode
  model.eval()
  
  # Paths to the images (adjust these paths as per your directory structure)
  img1_path = "dog.jpg"
  img2_path = "dog_cropped.jpg"
  img3_path = "another_dog.jpg"
  
  image1 = Image.open(img1_path).convert('RGB')
  image2 = Image.open(img2_path).convert('RGB')
  image3 = Image.open(img3_path).convert('RGB')
  preprocess = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  
  # Load and preprocess images
  image1_tensor = preprocess(image1).unsqueeze(0)
  image2_tensor = preprocess(image2).unsqueeze(0)
  image3_tensor = preprocess(image3).unsqueeze(0)
  
  # Compute image embeddings using ResNet50
  with torch.no_grad():
      model.eval()
      embedding1 = model(image1_tensor).squeeze().cpu().numpy()
      embedding2 = model(image2_tensor).squeeze().cpu().numpy()
      embedding3 = model(image3_tensor).squeeze().cpu().numpy()
  
  # Compute cosine similarities
  similarity_dog_augmented = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
  similarity_unrelated_dog = cosine_similarity(embedding1.reshape(1, -1), embedding3.reshape(1, -1))[0][0]
  
  # Print results
  print(f"Similarity between original dog and augmented dog: {similarity_dog_augmented:.4f}")
  print(f"Similarity between original dog and unrelated dog: {similarity_unrelated_dog:.4f}")
  
  ```
  
  <table>
   <tr>
     <td>
       <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/dog.jpg" width="400" height="400"/>
     </td>
     <td>
         <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/dog_cropped.jpg" width="400" height="400" />
     </td>
   </tr>
  </table>
  
  _Cosine similarity between 2 images : 0.8782_
  <table>
   <tr>
     <td>
       <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/dog.jpg" width="400" height="400"/>
     </td>
     <td>
         <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/another_dog.jpg" width="400" height="400" />
     </td>
   </tr>
  </table>
  
  _Cosine similarity between 2 images : 0.4228_
  
    *As observed, the cosine similarity between the original dog photo and the cropped dog photo is significantly higher compared to the similarity between the original dog photo and the unrelated dog photo. This method enables us to effectively identify duplicates within a dataset.  This images from [5].* 

--- 

:::

:::{.cell .markdown}

### How to mitigate duplicates in Texts:

*Duplicate detection in text data involves identifying identical or nearly identical pieces of text. In the context of image data, the paper “Deduplicating Training Data Makes Language Models Better" [4] describes two ways of detecting duplicates:*
  
  <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/textduplicates.png">

* **Exact Substring Duplication** : Due to the diversity of human language, it is uncommon for the same idea to be expressed identically in multiple documents unless one is derived from the other or both quote a shared source. When two text examples share a sufficiently long substring. Based on statistical analyses, a minimum matching substring length of 50 tokens is selected.
  * **Suffix Arrays**: Exact substring matching is computationally prohibitive with naive (quadratic) all-pair matching. To improve efficiency, all examples in the dataset are concatenated into a giant sequence, from which a Suffix Array is constructed. A suffix array is a representation of a suffix tree that can be constructed in linear time and allows efficient computation of many substring queries. 
    
      *For example, the suffixes of the sequence “banana” are (“banana”, “anana”, “nana” “ana”, “na”, “a”) and so the suffix array is the sequence (6 4 2 1 5 3).*
  *  **Substring matching**: Identify Duplicates by scanning the suffix array, repeated sequences can be identified as adjacent indices in the array. If two sequences share a common prefix of at least the threshold length, they are recorded as duplicates.

*This example illustrate the first approach :* 

:::

:::{.cell .code}
```python
# Input text
text = "the quick brown fox jumps over the lazy dog jumps over the quick brown fox."

# Step 1: Build suffix array
suffixes = [(text[i:], i) for i in range(len(text))]  # Create list of suffixes with their indices
suffixes.sort()  # Sort the suffixes lexicographically
suffix_array = [suffix[1] for suffix in suffixes]  # Extract indices from sorted suffixes
print("Suffix Array:", suffix_array)

# Step 2: Find duplicates using the suffix array
min_length = 7
duplicates = []
n = len(suffix_array)

for i in range(n - 1):
    lcp_length = 0  # Initialize the length of the longest common prefix (LCP)
    
    # Calculate the LCP between consecutive suffixes
    while (suffix_array[i] + lcp_length < len(text) and
           suffix_array[i+1] + lcp_length < len(text) and
           text[suffix_array[i] + lcp_length] == text[suffix_array[i+1] + lcp_length]):
        lcp_length += 1
    
    # Check if the LCP length is greater than or equal to the minimum length
    if lcp_length >= min_length:
        duplicates.append((suffix_array[i], suffix_array[i+1], lcp_length))

print("Duplicates:", duplicates)

# Step 3: Print duplicate substrings
for start1, start2, length in duplicates:
    duplicate_substring = text[start1:start1+length]
    print(f"Duplicate substring: '{duplicate_substring}' found at indices {start1} and {start2}")
``` 
:::

:::{.cell .markdown}
  * **Approximate Matching with MinHash** : Exact substring matching may not be sufficient for all cases, especially with web crawl text where documents might be identical except for minor variations. For such cases, approximate deduplication using MinHash is effective. MinHash is an approximate matching algorithm widely used in large-scale deduplication tasks. MinHash approximates the Jaccard Index, which measures the similarity between two sets of n-grams derived from documents. The algorithm uses hash functions to create document signatures by sorting n-grams and keeping only the smallest hashed n-grams.

      * Generate N-grams
      * Use hash functions to generate MinHash signatures by selecting the smallest hashed n-grams.
      * Calculate the probability that two documents are potential matches based on their MinHash signatures.

  *This example illustrate the second approach :* 

:::

:::{.cell .code}
```python
import hashlib

# Input texts
text1 = "the quick brown fox"
text2 = "the quick brown fox jumps over the lazy dog"
text3 = "I love ML"
n = 3

# Step 1: Generate n-grams
ngrams1 = [text1[i:i+n] for i in range(len(text1)-n+1)]
ngrams2 = [text2[i:i+n] for i in range(len(text2)-n+1)]
ngrams3 = [text3[i:i+n] for i in range(len(text3)-n+1)]
print("N-grams 1:", ngrams1)
print("N-grams 2:", ngrams2)
print("N-grams 3:", ngrams3)

# Step 2: Compute MinHash signatures
num_hashes = 100
signature1 = [min([int(hashlib.md5((str(seed) + ngram).encode()).hexdigest(), 16)
                  for ngram in ngrams1])
              for seed in range(num_hashes)]
signature2 = [min([int(hashlib.md5((str(seed) + ngram).encode()).hexdigest(), 16)
                  for ngram in ngrams2])
              for seed in range(num_hashes)]
signature3 = [min([int(hashlib.md5((str(seed) + ngram).encode()).hexdigest(), 16)
                  for ngram in ngrams3])
              for seed in range(num_hashes)]
print("MinHash Signature 1:", signature1)
print("MinHash Signature 2:", signature2)
print("MinHash Signature 3:", signature3)

# Step 3: Calculate Jaccard and MinHash similarities
jaccard_sim1_2 = len(set(ngrams1).intersection(set(ngrams2))) / len(set(ngrams1).union(set(ngrams2)))
jaccard_sim1_3 = len(set(ngrams1).intersection(set(ngrams3))) / len(set(ngrams1).union(set(ngrams3)))
minhash_sim1_2 = sum(1 for i in range(num_hashes) if signature1[i] == signature2[i]) / num_hashes
minhash_sim1_3 = sum(1 for i in range(num_hashes) if signature1[i] == signature3[i]) / num_hashes
print("Jaccard Similarity between text1 and text2:", jaccard_sim1_2)
print("Jaccard Similarity between text1 and text3:", jaccard_sim1_3)
print("MinHash Similarity between text1 and text2:", minhash_sim1_2)
print("MinHash Similarity between text1 and text3:", minhash_sim1_3)
```
:::

:::{.cell .markdown}

*As observed in this example, text1 has similarities with text2 but has 0 similarity with text3.*

---

:::

:::{.cell .markdown}

### How to mitigate duplicates in General Data:

:::

::: {.cell .markdown}

*For general data, which can include structured data in tables, finding duplicates often involves comparing multiple columns.*

* Comparing entire records for exact matches.
* Clustering grouping similar records together based on features.
* Assigning probabilities to different fields and calculating an overall similarity score.


:::

:::{.cell .markdown}

## References

:::

::: {.cell .markdown}

[1] [Leakage and the reproducibility crisis in machine learning-based science](https://arxiv.org/abs/2207.07048)

[2] [Do we train on test data? Purging CIFAR of near-duplicates](https://arxiv.org/pdf/1902.00423)

[3] [Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans](https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-021-00307-0/MediaObjects/42256_2021_307_MOESM1_ESM.pdf)

[4] [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/pdf/2107.06499)

[5] [National geographic](https://www.nationalgeographic.com/animals/mammals/facts/domestic-dog)

:::
