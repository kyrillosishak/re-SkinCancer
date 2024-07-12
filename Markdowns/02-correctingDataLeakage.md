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

:::{.cell .markdown}

# Mitigating Data Leakage in Skin Cancer Classification with Transfer Learning

`Skin Cancer Classification with Transfer Learning` paper that we are discussing uses HAM_10000 dataset.

The HAM_10000 dataset is a widely used dataset for dermatological image classification, containing thousands of images of pigmented lesions. While it has been a valuable resource for training machine learning models, recent studies have highlighted significant issues with data leakage due to duplicate images within the dataset. This data leakage can lead to overly optimistic performance estimates and undermine the validity of research findings [2].

In this notebook, we tackle the data leakage problem in the HAM_10000 dataset by addressing the duplicates within the dataset. Notably, insights from the survey paper "Leakage and the Reproducibility Crisis in ML-based Science" highlighted the issue of data leakage and its detrimental effects on reproducibility in machine learning. This survey also references the study "Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets," which provided evidence of duplicate images in dermatological datasets, including `HAM_10000`[1].

Our approach involves two main notebooks: the first reproduces the original paper, and the second focuses on identifying and addressing data leakage in the HAM_10000 dataset. By showing image similarity and identifying duplicates, we will clean the validation dataset and subsequently evaluate the impact on model accuracy and confusion metrics.

---
**🔍 In this notebook, we will:**

1. Identify Duplicates in `HAM_10000`4
2. Clean the Validation Dataset.
3. Evaluate Model Performance on new clean validation data.
4. Discuss Implications.

:::

:::{.cell .markdown}

## **1. Identify Duplicates in HAM_10000**

**`HAM_10000` have severe flaw :**

> A caveat of HAM10000, despite its rather large size, is that it contains multiple images of the
same lesion captured either from different viewing angles or at different magnification levels

> the number of lesions with unique lesion IDs (HAM_xxx) is smaller than the
number of images with unique image IDs (ISIC_xxx).

> observe that the 10,015 images are in fact derived from only 7,470 unique lesions, and 1,956
of these lesion IDs (∼26.18%) contains 2 or more images: 1,423 lesions have 2 images, 490
lesions have 3 images, 34 lesions have 4 images, 5 lesions have 5 images, and 6 lesions have 4
images each.

<img src = "https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/Near-duplicate_HAM10000.png" height = 150>
<img src = "https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/Near-duplicate2_HAM10000.png" height = 150>


_This images from [2]._

This results in near-duplicates within the dataset, which can compromise the integrity of any machine learning model trained on it. The presence of near-duplicate images can lead to data leakage, where the model learns to recognize specific lesions rather than generalizing to new, unseen lesions.

*The duplicates were identified using a tool called [FastDup](https://github.com/visual-layer/fastdup) in the repository.*

:::

:::{.cell .code}
```python
# Download duplicates file data
!wget https://github.com/kakumarabhishek/Corrected-Skin-Image-Datasets/raw/main/DermaMNIST/HAM10000_DuplicateConfirmation/fastdup_outputs/duplicates_1000.csv
```
:::

:::{.cell .code}
```python
# Download the dataset to dir data
!curl -L -O -J -H "X-Dataverse-key:$API_TOKEN" https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/DBW86T
!unzip -q dataverse_files.zip
!mkdir data
%cd data
!unzip -q ../HAM10000_images_part_1.zip
!unzip -q ../HAM10000_images_part_2.zip
%cd ..
```
:::

:::{.cell .code}
```python
import pandas as pd
duplicates = pd.read_csv("duplicates_1000.csv")
duplicates.head()
```
:::

:::{.cell .code}
```python
# Read the CSV file containing metadata for the HAM10000 dataset
df = pd.read_csv('HAM10000_metadata')

# Group the image filenames by their class labels ('dx') and convert the groups to a dictionary
# The dictionary keys are class labels and the values are lists of image filenames belonging to each class
class_files = df.groupby('dx')['image_id'].apply(list).to_dict()

# Create a mapping from class names to integer indices
# This is useful for converting class labels to numeric format for machine learning tasks
label_map = {class_name: idx for idx, class_name in enumerate(class_files.keys())}

# Create a list of class names
class_names = [class_name for class_name in class_files.keys()]

print("Class names: ",class_names)
print("Label map: ",label_map)
```
:::

:::{.cell .markdown}

## **2. Clean the validation dataset**

In this section, we will use a previously training dataset that was loaded from the previous experiment we have done. We will clean the validation data, and then validate the model again.

:::

:::{.cell .code}
```python
!wget https://huggingface.co/KyrillosIshak/Re-SkinCancer/resolve/main/Experiments/exp1/train.pt
!wget https://huggingface.co/KyrillosIshak/Re-SkinCancer/resolve/main/Experiments/exp1/val.pt
!wget https://huggingface.co/KyrillosIshak/Re-SkinCancer/resolve/main/Experiments/exp1/train_loader.pt
!wget https://huggingface.co/KyrillosIshak/Re-SkinCancer/resolve/main/Experiments/exp1/val_loader.pt
```
:::

:::{.cell .code}
```python
import os
import copy
import math
import torch
import random
import PIL.Image
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```
:::

:::{.cell .code}
```python
class SkinLesionDataset(Dataset):
    """
    A custom dataset class for loading and transforming images of skin lesions along with their labels.

    Args:
        image_list (list of str): List of image filenames.
        labels (list of int): List of labels corresponding to each image.
        transform (callable, optional): Optional transform to be applied on an image sample.
    """
    def __init__(self, image_list, labels, transform=None):
        self.image_list = image_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img_path = "./data/"+img_path+".jpg"
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
```
:::

:::{.cell .code}
```python
def get_duplicates(duplicates,train_dataset,val_dataset):
    """
    Identifies and returns duplicate images that are present in different datasets (train and val).
    
    Args:
        duplicates (pd.DataFrame): DataFrame containing columns 'from_img' and 'to_img' with image filenames.
        train_dataset (Dataset): Dataset object containing an attribute image_list with image filenames in the training set.
        val_dataset (Dataset): Dataset object containing an attribute image_list with image filenames in the validation set.

    Returns:
        list of dict: A list of dictionaries where each dictionary contains information about duplicate images and their locations.
                      Each dictionary has keys 'from_img', 'to_img', 'from_img_location', and 'to_img_location'.
    """

    # Remove the '.jpg' extension from the filenames in the duplicates DataFrame
    duplicates['from_img'] = duplicates['from_img'].str.replace('.jpg', '')
    duplicates['to_img'] = duplicates['to_img'].str.replace('.jpg', '')

    # Initialize list to store results
    duplicate_results = []

    # Loop through the duplicates dataframe and check for presence in datasets
    for index, row in duplicates.iterrows():
        from_img = row['from_img']
        to_img = row['to_img']
        
        # Check if the images are in the train or val dataset
        from_img_in_train = from_img in train_dataset.image_list
        from_img_in_val = from_img in val_dataset.image_list
        to_img_in_train = to_img in train_dataset.image_list
        to_img_in_val = to_img in val_dataset.image_list
        
        # If one image is in train and the other is in val, record the locations
        if (from_img_in_train and to_img_in_val) or (from_img_in_val and to_img_in_train):
            location = {
                'from_img': from_img,
                'to_img': to_img,
                'from_img_location': 'train' if from_img_in_train else 'val',
                'to_img_location': 'val' if to_img_in_val else 'train'
            }
            duplicate_results.append(location)
    return duplicate_results
```
:::

:::{.cell .code}
```python
def find_replacement_image(train_list, val_list, class_files_dict,image):
    """
    Finds a replacement image that is not present in the training or validation datasets.

    Args:
        train_list (list): List of image filenames in the training dataset.
        val_list (list): List of image filenames in the validation dataset.
        class_files_dict (dict): Dictionary where keys are class labels and values are lists of image filenames belonging to that class.
        image (str): The image filename to avoid selecting as a replacement.

    Returns:
        str: A replacement image filename or an empty string if no suitable image is found.
    """

    # Filter out images that are already in the training or validation datasets, and the given image to avoid
    available_images = [img for img in class_files_dict if img not in train_list and img not in val_list and not image]
    if available_images : 
        return random.choice(available_images)
    else:
        return ""

def correct_duplicates(duplicate_results,train_dataset,val_dataset,class_files):
    """
    Corrects duplicate images by replacing them with new images from the same class.

    Args:
        duplicate_results (list): List of dictionaries containing information about duplicate images and their locations.
        train_dataset (Dataset): Dataset object containing an attribute image_list with image filenames in the training set.
        val_dataset (Dataset): Dataset object containing an attribute image_list with image filenames in the validation set.
        class_files (dict): Dictionary where keys are class labels and values are lists of image filenames belonging to that class.

    Returns:
        Dataset: The updated validation dataset with duplicates corrected.
    """

    # Loop through each duplicate record
    for dup in duplicate_results:
        # If the duplicate image is in the validation set, find and replace it
        if dup['from_img_location'] == 'val' and dup['from_img'] in val_dataset.image_list:
            # Remove the duplicate image from the validation dataset
            val_dataset.image_list.remove(dup['from_img'])
            # Identify the class label of the duplicate image
            class_label = next(key for key, value in class_files.items() if dup['from_img'] in value)
            # Find a replacement image from the same class
            replacement_img = find_replacement_image(train_dataset.image_list, val_dataset.image_list, class_files[class_label], dup['to_img'])
            # Add the replacement image to the validation dataset if found
            if replacement_img != "":
                val_dataset.image_list.append(replacement_img)
        
        # Repeat the process for the 'to_img' if it's in the validation set
        if dup['to_img_location'] == 'val' and dup['to_img'] in val_dataset.image_list:
            # Remove the duplicate image from the validation dataset
            val_dataset.image_list.remove(dup['to_img'])
            # Identify the class label of the duplicate image
            class_label = next(key for key, value in class_files.items() if dup['to_img'] in value)
            # Find a replacement image from the same class
            replacement_img = find_replacement_image(train_dataset.image_list, val_dataset.image_list, class_files[class_label], dup['to_img'])
            # Add the replacement image to the validation dataset if found
            if replacement_img != "":
                val_dataset.image_list.append(replacement_img)
    
    return val_dataset
```
:::

:::{.cell .markdown}

First we will load the previous experiment's checkpoint (the data we used in the experiment)

:::

:::{.cell .code}
```python
val_dataset = torch.load('val.pt')
val_loader = torch.load('val_loader.pt')
train_dataset = torch.load("train.pt")

```
:::

:::{.cell .markdown}

Second we will compute the number of duplicates in training/validation

:::

:::{.cell .code}
```python
duplicate_results = get_duplicates(duplicates,train_dataset,val_dataset)
print(len(duplicate_results))
```
:::


:::{.cell .markdown}

Third we will fix the duplicates in validation dataset by removing duplicates and replacing every duplicate img by a random img from the same class

:::

:::{.cell .code}
```python
val_dataset_corrected = correct_duplicates(duplicate_results,train_dataset,val_dataset,class_files)
val_loader_corrected = DataLoader(val_dataset_corrected, batch_size=10, shuffle=False)
```
:::

:::{.cell .markdown}

Recheck if we have any duplicates in our new validation set

:::

:::{.cell .code}
```python
duplicate_results = get_duplicates(duplicates,train_dataset,val_dataset_corrected)
print(len(duplicate_results))
```
:::



:::{.cell .markdown}

## **3. Evaluate Model Performance on new clean validation data**

:::

:::{.cell .code}
```python
!wget https://huggingface.co/KyrillosIshak/Re-SkinCancer/resolve/main/Experiments/exp1/3500_81_33.pt
```
:::

:::{.cell .code}
```python
class ModifiedInceptionResNetV2(nn.Module):
    """ModifiedInceptionResNetV2 class for transfer learning with custom classifier.

      This class implements a modified version of the Inception ResNet V2 model for image classification tasks.
      It leverages transfer learning by freezing the pre-trained feature extraction layers from a
      provided Inception ResNet V2 model and adding a custom classifier on top.

      Args:
          original_model (torchvision.models.InceptionV3): A pre-trained Inception ResNet V2 model
              (typically loaded with `pretrained=True`).
          num_classes (int, optional): The number of output classes for the classification task.
              Defaults to 7.

      Attributes:
          features (nn.Sequential): A sequential container holding all layers from the original model
              except the final classifier (Softmax layer).
          classifier (nn.Sequential): A custom classifier consisting of:
              - nn.Flatten(): Flattens the input from the feature extractor.
              - nn.Linear(1536, 64): First fully-connected layer with 64 units and ReLU activation.
              - nn.Linear(64, num_classes): Second fully-connected layer with 'num_classes' units
                and Softmax activation for probability distribution of the classes.
    """
    def __init__(self, original_model, num_classes=7):
        super(ModifiedInceptionResNetV2, self).__init__()

        # Retain all layers except the final classifier(Softmax)
        self.features = nn.Sequential(*list(original_model.children())[:-1])

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 1536 output from the last layer after removing the classifier
            nn.Linear(1536, 512),  # First fully connected layer
            nn.ReLU(),
            nn.Linear(512, num_classes),  # Second fully connected layer
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```
:::

:::{.cell .code}
```python
def get_predictions_and_labels(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)
```
:::

:::{.cell .code}
```python
!pip -q install timm
```
:::

:::{.cell .code}
```python
from timm import create_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model('inception_resnet_v2', pretrained=True,num_classes=7)
modified_model = ModifiedInceptionResNetV2(model, num_classes=7)
modified_model.load_state_dict(torch.load('3500_81_33.pt')['model'])

modified_model.to(device)
print("Loaded model successfully.")
```
:::

:::{.cell .code}
```python
val_preds_duplicates, val_labels_duplicates = get_predictions_and_labels(modified_model, val_loader, device)
val_preds_clean, val_labels_clean = get_predictions_and_labels(modified_model, val_loader_corrected, device)
```
:::

:::{.cell .code}
```python
accuracy_duplicates = np.mean(val_preds_duplicates == val_labels_duplicates)*100
accuracy_clean = np.mean(val_preds_clean == val_labels_clean)*100
print(f"Validation Accuracy of the duplicated set: {accuracy_duplicates:.4f}")
print(f"Validation Accuracy of the clean set: {accuracy_clean:.4f}")
```
:::

:::{.cell .code}
```python
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
```
:::

:::{.cell .code}
```python
# Compute confusion matrix
cm1 = confusion_matrix(val_labels_duplicates, val_preds_duplicates)
cm2 = confusion_matrix(val_labels_clean, val_preds_clean)
```
:::

:::{.cell .code}
```python
# Plot confusion matrix
plot_confusion_matrix(cm1, class_names)
```
:::

:::{.cell .code}
```python
# Plot confusion matrix
plot_confusion_matrix(cm2, class_names)
```
:::

:::{.cell .markdown}

## **Refrences**

1. [Leakage and the Reproducibility Crisis in ML-based Science](https://arxiv.org/abs/2207.07048)

2. [Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Dataset](https://arxiv.org/pdf/2401.14497)

:::
