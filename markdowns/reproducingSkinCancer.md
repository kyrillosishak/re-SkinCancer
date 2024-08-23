
:::{.cell .markdown}
# Skin Cancer Classification using Inception Network and Transfer Learning

:::

:::{.cell .markdown}
In this series of notebooks, we aim to reproduce the results of [Skin Cancer Classification using Inception Network and Transfer Learning](https://arxiv.org/pdf/2111.02402v1). The paper explores the application of advanced neural network architectures, specifically Inception Networks, in conjunction with transfer learning techniques for skin cancer classification. Our objective is to rigorously validate the findings of the paper, ensuring transparency and reproducibility in our approach. Additionally, we will investigate and mitigate potential sources of data leakage within the dataset. Data leakage can lead to overly optimistic results by inadvertently incorporating information from the validation or test sets into the training process. By identifying and addressing these issues, we aim to demonstrate how such leakage can impact model performance and interpretation of results.

**🏆 Objectives of this notebooks:**

1. Identify the specific claims related to classification accuracy, generalization across different skin lesion types, and the efficacy of the proposed methodology.
2. Assess the methodology, dataset (HAM_10000), and the problem statement presented.
3. Define specific experiments needed to validate each identified claim.

4. Obtain the HAM_10000 dataset and perform data cleaning and preprocessing steps as mentioned in the paper.

5. Choose the same or similar models used in the paper for skin cancer classification and train the models using the preprocessed data and evaluate them using the same metrics reported in the paper.
6. Compare the results obtained from your reproduced models with the results reported in the original paper and analyze any differences or discrepancies in performance and identify potential reasons for variations.
7. Identify potential sources of data leakage within the HAM_10000 dataset.
8. Re-train models after mitigating identified data leakage sources and compare performance metrics before and after leakage removal.


**🔍 In this notebook, we will:**
1. Identify the specific claims.
2. Define specific experiments.
3. Obtain the HAM_10000 dataset and perform data preprocessing.
4. Use the same model used in the paper for skin cancer using the preprocessed data.
5. Compare the results obtained from your reproduced models with the original paper's results.
---

**🗣️ Claims :**

1. The model achieved a validation accuracy of 73.4% after the first training phase and 78.9% after the second training phase.
2. The results showed that the model could classify six out of seven categories with a true positive rate higher than 75%, even for classes with limited samples.
3. The entire training process required less than 20 GB of RAM and was completed in under four hours using a Google Colab GPU.

**🧪Experiment from the paper:**

> Images are loaded and resized from 450×600 to 299×299 in order
to be correctly processed by the network. After a normalization step on RGB arrays, we split the dataset into a training and validation set with 80:20 ratio.

> In order to re-balance the dataset, we chose to shrink the amount of images for each class to an equal maximum dimension of 450 samples. This significant decrease of available images is then mitigated by applying a step of data augmentation. Training set expansion is made by altering images with small transformations to reproduce some variations, such as horizontal flips, vertical flips, translations, rotations and shearing transformations.


> We decided to take advantage of transfer learning, utilizing `Inception-ResNet-v2` pre-trained on ImageNet.

> The original `Inception-ResNet-v2` architecture has a stem block consisting of the concatenation of multiple convolutional and pooling layers, while Inception-ResNet blocks (A, B and C) contain a set of convolutional filters with an average pooling layer. *This structure has
been extended with a final module consisting of a flattening step, two fully-connected layers of 64 units each, and the softmax classifier.*

>  In this work we used a
stochastic gradient descent optimizer (SGD), with learning rate set to 0.0006 and usage of momentum and Nesterov Accelerated Gradient in order to adapt updates to the slope of the loss function (categorical cross entropy) and speed up the training process.

>  The total number of epochs was set to 100, using a small
batch size of 10. A maximum patience of 15 epochs was set to the early stopping callback in order to mitigate the overfitting.

> In order to improve classification performance, specially on minority classes, we loaded the best model obtained in the first round to extend the training phase and explore other potential local minimum points of the loss function, by using an additional amount of 20 epochs.

:::

:::{.cell .markdown}

## Data Loading and preprocessing

:::

:::{.cell .markdown}

`HAM10000`(Human Against Machine with 10000 images) contains 10,015 dermoscopic images of pigmented skin lesions collected from patients at two study sites in Australia and Austria, with their diagnoses confirmed by either histopathology, confocal microscopy, clinical follow-up visits, or expert consensus. The 7 disease labels in the dataset cover 95% of the lesions encountered in clinical practice. Because of these meritorious properties, `HAM10000` is a good candidate dataset for dermatological analysis. However the resulting `HAM10000` from serious flaws.

:::

:::{.cell .markdown}

### Downloading the data

:::

:::{.cell .markdown}

To download the data you should :

1. Signup https://dataverse.harvard.edu/
2. Create API token from [this](https://dataverse.harvard.edu/dataverseuser.xhtml?selectTab=apiTokenTab).

:::

:::{.cell .code}
```python
!export API_TOKEN=#[put your API token here without space between '=' and token]
```
:::

:::{.cell .code}
```python
# Downloading the data from official Harvard dataverse website
!curl -L -O -J -H "X-Dataverse-key:$(API_TOKEN)" https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/DBW86T
```
:::

:::{.cell .code}
```python
!unzip -q dataverse_files.zip
```
:::

:::{.cell .code}
```python
# Unzipping all images in directory data
!mkdir data
%cd data
!unzip -q ../HAM10000_images_part_1.zip
!unzip -q ../HAM10000_images_part_2.zip
```
:::

:::{.cell .markdown}

### Exploring the data

:::

:::{.cell .code}
```python
import pandas as pd
# Loading metadata
df = pd.read_csv('../HAM10000_metadata')
class_files = df.groupby('dx')['image_id'].apply(list).to_dict()
print("This is the classes dictionary :",class_files.keys())
print("The number of images in class df    =",len(class_files['df']))
print("The number of images in class vasc  =",len(class_files['vasc']))
print("The number of images in class akiec =",len(class_files['akiec']))
print("The number of images in class bcc   =",len(class_files['bcc']))
print("The number of images in class bkl   =",len(class_files['bkl']))
print("The number of images in class mel   =",len(class_files['mel']))
print("The number of images in class nv    =",len(class_files['nv']))
```
:::

:::{.cell .markdown}

As we can see here, the data is heavily unbalanced. To achieve a balanced dataset, each class should have an average of 
10015/7 ≈ 1430 images. However, 6 out of the 7 classes fall below this number.

Additionally, the authors made a mistake by not thoroughly exploring the data to identify that it contains duplicates. This oversight can lead to data leakage, negatively impacting model performance and evaluation. In the next notebook, we will identify this problem and take steps to address it.

:::


:::{.cell .code}
```python
import random
from PIL import Image
from glob import glob
# Choosing a random image
list_of_images_paths = glob('../data/*')
random_index = random.randint(0, len(list_of_images_paths) - 1)
random_image_path = list_of_images_paths[random_index]
# Load the image
image = Image.open(random_image_path)
print("Size of the image is",image.size)
image
```

:::

:::{.cell .markdown}

### Data Augmentation 

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
import seaborn as sns
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
```
:::

:::{.cell .markdown}

#### Augmentation

:::

:::{.cell .markdown}

> Images are loaded and resized from 450×600 to 299×299

> Training set expansion is made by altering images with small transformations to reproduce some variations, such as horizontal flips, vertical flips, translations, rotations and shearing transformations.

**If you are not fimiliar with torchvision transforms :**

- `transforms.Compose`: This function takes a list of transformations and applies them sequentially to an image. 
- `transforms.RandomHorizontalFlip()` and `transforms.RandomVerticalFlip()` : This transformation randomly flips the image horizontally/vertically with a probability of 50%.
- `transforms.RandomAffine()` : 
    
    * Randomly rotates the image within the range of -30 to +30 degrees.
    * Randomly translates the image horizontally and vertically by up to 10% of the image size.
    * Randomly applies a shear transformation within the range of -10 to +10 degrees.
 
- `transforms.ToTensor()`: This transformation converts a PIL Image or a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

:::

:::{.cell .code}
```python
train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(
        degrees=30,
        translate=(0.1, 0.1),
        scale=None,
        shear=10
    ),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])
```

:::

:::{.cell .markdown}

#### Defining Dataset

:::
:::{.cell .markdown}

If you are not familiar with PyTorch Dataset & Dataloader [visit](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).

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
        img_path = "../data/"+img_path+".jpg"
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
```
:::

:::{.cell .markdown}

#### Splitting data & creating data loaders

:::
:::{.cell .markdown}

> we split the dataset into a training and validation set with 80:20 ratio.

> In order to re-balance the dataset, we chose to shrink the amount of images for each class to an equal maximum dimension of 450 samples. This significant decrease of available images is then mitigated by applying a step of data augmentation. Training set expansion is made by altering images with small transformations to reproduce some variations, such as horizontal flips, vertical flips, translations, rotations and shearing transformations.

**The functions do the following :** 

* it combines the image_list and label_list into pairs and shuffles them to ensure randomness. It then limits the size of the dataset to 450 images if the original list exceeds this number. The function splits the class size into training and validation sets, with the validation set comprising 20% of the total images.

* the function focuses on augmenting the training data to increase the dataset size up to a target size (default is 300). It repeatedly adds images from the training set until the desired target size is reached.

* the function creates and returns two dataset objects, train_dataset and val_dataset, using the SkinLesionDataset class, applying the specified transformations for training and validation respectively. This function is intended to be called for each of the seven classes individually, thus preparing the data in well-augmented manner.



:::

:::{.cell .code}
```python
def augment_and_split_data(image_list, label_list, train_transform, val_transform, target_size=300):
    """
    Augments and splits a dataset of images and labels into training and validation sets.

    Args:
        image_list (list of str): List of image filenames.
        label_list (list of int): List of labels corresponding to each image.
        train_transform (callable): Transformations to be applied to training images.
        val_transform (callable): Transformations to be applied to validation images.
        target_size (int, optional): Desired number of training images after augmentation. Default is 300.

    Returns:
        tuple: A tuple containing the training dataset and validation dataset.
    """
    # Combine images and labels, and shuffle them
    combined = list(zip(image_list, label_list))
    random.shuffle(combined)
    image_list[:], label_list[:] = zip(*combined)

    # Limit the dataset size to 450 images if it exceeds this number
    if len(image_list) > 450:
        image_list = image_list[:450]
        label_list = label_list[:450]

    # Calculate the size of the validation set (20% of the total dataset)
    val_size = math.ceil(0.2 * len(image_list))

    # Split the dataset into training and validation sets
    val_images = image_list[:val_size]
    val_labels = label_list[:val_size]
    train_images = image_list[val_size:]
    train_labels = label_list[val_size:]


    # Augment the training set to reach the target size
    augmented_images = []
    augmented_labels = []
    while len(augmented_images) < target_size:
        for img, label in zip(train_images, train_labels):
            augmented_images.append(img)
            augmented_labels.append(label)
            if len(augmented_images) >= target_size:
                break
    train_dataset = SkinLesionDataset(augmented_images, augmented_labels, transform=train_transform)
    val_dataset = SkinLesionDataset(val_images, val_labels, transform=val_transform)

    return train_dataset, val_dataset
```
:::

:::{.cell .code}
```python
def process_train_val_loader(target_size, train_transform, val_transform):
    """
    Processes and prepares the training and validation data loaders.

    Args:
        target_size (int): The desired number of training images after augmentation.
        train_transform (callable): Transformations to be applied to training images.
        val_transform (callable): Transformations to be applied to validation images.

    Returns:
        tuple: A tuple containing the training dataset, validation dataset, training data loader, and validation data loader.
    """
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    # Create a label map to convert class names to numeric labels
    label_map = {class_name: idx for idx, class_name in enumerate(class_files.keys())}

    # Process each class in the class_files dictionary
    for class_name, image_list in class_files.items():
        # Generate labels for the current class
        labels = [label_map[class_name]] * len(image_list)
        # Augment and split the data into training and validation sets
        train_dataset, val_dataset = augment_and_split_data(image_list, labels, train_transform, val_transform, target_size)
        train_images.extend(train_dataset.image_list)
        train_labels.extend(train_dataset.labels)
        val_images.extend(val_dataset.image_list)
        val_labels.extend(val_dataset.labels)

    # Create datasets and loaders
    train_dataset = SkinLesionDataset(train_images, train_labels, transform=train_transform)
    val_dataset = SkinLesionDataset(val_images, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader
```
:::

:::{.cell .markdown}

## Model creation and Transfer Learning

:::
:::{.cell .markdown}

>  The original Inception-ResNet-v2 architecture has a stem block consisting of the concatenation of multiple convolutional and pooling layers, while Inception-ResNet blocks (A, B and C) contain a set of convolutional filters with an average pooling layer. This structure has been extended with a final module consisting of a flattening step, two fully-connected layers of 64 units each, and the softmax classifier.

`Inception-ResNet-v2` is a deep convolutional neural network architecture that combines the strengths of two powerful designs: Inception and Residual Networks (ResNet). Inception modules aim to capture multi-scale features by performing convolutions of different sizes (e.g., 1x1, 3x3, 5x5) in parallel, followed by concatenation of the results. This design helps in efficiently capturing spatial hierarchies and reduces computational cost. Inspired by ResNet, Inception-ResNet-v2 incorporates skip (residual) connections which addresses the vanishing gradient problem and enabling the training of much deeper networks.


<table>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/Original_Inception-ResNet-v2.png" />
    </td>
    <td>
<img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/Modified_Inception-ResNet-v2.png" width="199" height="469" />
    </td>
  </tr>

  <tr>
    <td>Original model</td>
    <td>Modified model</td>
  </tr>
</table>


**Transfer Learning :**

* we'll leverage transfer learning to create a custom image classifier. We'll be using the Inception ResNet V2 model pre-trained on ImageNet, a massive dataset with thousands of image classes. Transfer learning allows us to reuse the knowledge this model has learned from ImageNet, even if our own dataset has different categories.
* We'll use the timm library from Hugging Face to load the Inception ResNet V2 model pre-trained on ImageNet. This pre-trained model has already learned effective ways to extract features from images.
* We'll remove the final classification layer of the pre-trained model and add our own custom classifier.

:::

:::{.cell .markdown}

### Load original model

:::

:::{.cell .code}
```python
# if you are in colab install timm 
!pip -q install timm
```
:::

:::{.cell .code}
```python
# if you are in Kaggle notebook install torchsummary 
!pip -q install torchsummary
```
:::

:::{.cell .code}
```python
from torchsummary import summary
from timm import create_model
model = create_model('inception_resnet_v2', pretrained=True,num_classes=7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Inspect model parameters and layers
summary(model, input_size=(3, 299, 299))
```
:::

:::{.cell .markdown}

As we see these are the last 2 layers :
```
Dropout-1151                 [-1, 1536]               0
Linear-1152                    [-1, 7]             10,759
```
we will remove the last layer only(the classifier) 
:::

:::{.cell .markdown}

### Creating modified model

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
            nn.Linear(1536, 64),  # First fully connected layer 
            nn.ReLU(),
            nn.Linear(64, num_classes),  # Second fully connected layer
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
modified_model = ModifiedInceptionResNetV2(model, num_classes=7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modified_model.to(device)
# Inspect the modified model
summary(modified_model, input_size=(3, 299, 299))
```
:::

:::{.cell .markdown}

## Training

:::
:::{.cell .markdown}

*The problem we counter while reproducing the paper that isn't specified how much they augmented the Train data.*

>  In this work we used a
stochastic gradient descent optimizer (SGD), with learning rate set to 0.0006 and usage of momentum and Nesterov Accelerated Gradient in order to adapt updates to the slope of the loss function (categorical cross entropy) and speed up the training process.

>  The total number of epochs was set to 100, using a small
batch size of 10. A maximum patience of 15 epochs was set to the early stopping callback in order to mitigate the overfitting.

> In order to improve classification performance, specially on minority classes, we loaded the best model obtained in the first round to extend the training phase and explore other potential local minimum points of the loss function, by using an additional amount of 20 epochs.

:::

:::{.cell .code}
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(modified_model.parameters(), lr=0.0006,momentum=0.9, nesterov=True)
```
:::

:::{.cell .code}
```python
!mkdir ../experiments
```
:::

:::{.cell .code}
```python
def train(num_epochs, train_loader, val_loader, model, optimizer, criterion, patience, early_stopping=True):
    """Trains a deep learning model for image classification with early stopping.

    This function trains a provided model (`model`) on a given dataset (`train_loader`) 
    for a specified number of epochs (`num_epochs`). It also performs validation 
    on a separate dataset (`val_loader`) to monitor performance and potentially apply 
    early stopping to prevent overfitting.

    Args:
        num_epochs (int): The number of training epochs.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        val_loader (torch.utils.data.DataLoader): The data loader for validation data.
        model (torch.nn.Module): The deep learning model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
        criterion (torch.nn.Module): The loss function used for calculating training loss.
        patience (int): The number of epochs to wait for improvement in validation loss 
            before triggering early stopping (if enabled).
        early_stopping (bool, optional): A flag to enable early stopping (default: True).

    Returns:
        torch.nn.Module: The trained model with the best weights found during validation.
    """
    # Track best validation loss and patience counter for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    
    for epoch in tqdm(range(num_epochs)):
        # Set model to training mode
        model.train()

        # Initialize running loss for the epoch
        running_loss = 0.0

        for images, labels in train_loader:
            # Transfer images and labels to the device
            images, labels = images.to(device), labels.to(device)
            # Clear gradients from the previous iteration
            optimizer.zero_grad()
            # Forward pass: predict on the images
            outputs = model(images)
            # Calculate the loss based on predictions and labels
            loss = criterion(outputs, labels)
            # Backpropagate the loss to update model weights
            loss.backward()
            # Update model parameters using the optimizer
            optimizer.step()
            # Accumulate the training loss for the epoch
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Early stopping
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load the best model weights
    model.load_state_dict(best_model_weights)
    model_name = f'{number}'
    experiment_dir = '../experiments'
    model_directory =  os.path.join(experiment_dir, f'{model_name}.pt')
    torch.save({
        'model': model.state_dict()
    }, model_directory)

    print(f"Model saved to checkpoint: {model_directory} as f'{model_name}.pt")
    print("Training and validation completed.")
    return model
```
:::

:::{.cell .code}
```python
number = 500
num_epochs = 100
patience = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset, val_dataset, train_loader, val_loader = process_train_val_loader(number, train_transform, val_transform)
print("Size of trainset : " + str(len(train_dataset.image_list)))
print("Size of validationset : " + str(len(val_dataset.image_list)))
trained_model = train(num_epochs, train_loader, val_loader, modified_model, optimizer, criterion, patience, early_stopping=True)
```
:::

:::{.cell .markdown}
## Validation
:::

:::{.cell .code}
```python
def get_predictions_and_labels(model, data_loader, device):
    # Set the model to evaluation mode
    model.eval()
    all_preds = []
    all_labels = []
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, labels in data_loader:
            # Move images and labels to the specified device
            images, labels = images.to(device), labels.to(device)
            
            # Get model outputs
            outputs = model(images)
            
            # Get the index of the highest probability class
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    return np.array(all_preds), np.array(all_labels)
```
:::

:::{.cell .code}
```python
# Get predictions and labels for the validation dataset
val_preds, val_labels = get_predictions_and_labels(trained_model, val_loader, device)

# Create a mapping from class names to indices
label_map = {class_name: idx for idx, class_name in enumerate(class_files.keys())}

# Get class names in the correct order
class_names = [class_name for class_name in class_files.keys()]
```
:::

:::{.cell .code}
```python
# Compute and plot the confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Compute confusion matrix
cm = confusion_matrix(val_labels, val_preds)

# Plot confusion matrix
plot_confusion_matrix(cm, class_names)
```
:::

:::{.cell .markdown}
## Comparing our results to paper's results

:::
:::{.cell .markdown}

<table>
  <tr>
    <td></td>
    <td>Original results</td>
    <td>Our results</td>
  </tr>
  <tr>
    <td>
        Accuracy
    </td>
    <td>
        78.9%
   </td>
    <td>
        78.6%
    </td>
  </tr> 
  <tr>
    <td>
        Number of epochs
    </td>
    <td>
        Approx. 42 epochs
    </td>
    <td>
        40 epochs
    </td>
  </tr>
  <tr>
    <td>
        Training size
    </td>
    <td>
        Unknown
    </td>
    <td>
        7000 samples
    </td>
  </tr>
  <tr>
    <td>
        Validation size
    </td>
    <td>
        478 samples
    </td>
    <td>
        478 samples
    </td>
  </tr>
  <tr>
    <td>
        Confusion martix
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/paper's_results.jpeg" />
    </td>
    <td>
        <img src="https://raw.githubusercontent.com/kyrillosishak/re-SkinCancer/main/assets/Our_results.jpeg" />
    </td>
  </tr>
  
</table>

:::


:::{.cell .markdown}
*for this experiment you can download the trained model parameters and the data used from https://huggingface.co/KyrillosIshak/Re-SkinCancer/resolve/main/Experiments/exp3/*
:::
