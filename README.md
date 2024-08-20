# re-SkinCancer

In this series of notebooks, we will explore a result from 

> Benedetti, P., Perri, D., Simonetti, M., Gervasi, O., Reali, G., Femminella, M. (2020). Skin Cancer Classification Using Inception Network and Transfer Learning. In: Gervasi, O., et al. Computational Science and Its Applications – ICCSA 2020. ICCSA 2020. Lecture Notes in Computer Science(), vol 12249. Springer, Cham. https://doi.org/10.1007/978-3-030-58799-4_39 [arXiv](https://arxiv.org/pdf/2111.02402v1)

In this paper, the authors use transfer learning on a pretrained convolutional neural network to classify skin lesions in dermatoscopic images from HAM10000 (Human Against Machine with 10000 training images) dataset. The paper reports a final accuracy of 78.9% on the validation set.

The first notebook in this sequence reproduces their result and achieves approximately 78.87% accuracy on the validation set:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyrillosishak/re-SkinCancer/blob/main/notebooks/reproducingSkinCancer.ipynb) [Reproducing "Skin Cancer Classification Using Inception Network and Transfer Learning"](https://github.com/kyrillosishak/re-SkinCancer/blob/main/Notebooks/reproducingSkinCancer.ipynb)

However, although this seems like a good result, we do not expect the model to achieve such a high accuracy when used to classify new lesions. The original result was trained and evaluated with *data leakage* - where there is contamination between the training set and validation or test set, in a way that will not be present when the model is used in production. Specifically, this example has a kind of data leakage where there are duplicate images of the same lesion in both the training set and the validation set that is used to evaluate model performance. This leads to an *overly optimistic evaluation* on the validation set - we were validating the model using an "easier" task than we had intended: classify lesions already seen in traning.

The second notebook in this sequence explores this type of data leakage using some synthetic data and some smaller datasts:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyrillosishak/re-SkinCancer/blob/main/notebooks/exploreDuplicate.ipynb) [Exploring the problem of duplicate samples in training and validation/test](https://github.com/kyrillosishak/re-SkinCancer/blob/main/notebooks/exploreDuplicate.ipynb)

Finally, we will repeat the original work, but with a correct evaluation - making sure that there are no duplicate images in the training and validation sets. (We will also correct another data leakage problem in the original: using the validation set for both early stopping and for evaluation of model performance.) We will see that the final accuracy is lower when the model is evaluated correctly.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyrillosishak/re-SkinCancer/blob/main/notebooks/correctingSkinCancer.ipynb) [Repeating "Skin Cancer Classification Using Inception Network and Transfer Learning" without data leakage](https://github.com/kyrillosishak/re-SkinCancer/blob/main/notebooks/correctingSkinCancer.ipynb)

---

This resource may be executed on Google Colab or on [Chameleon](https://chameleoncloud.org/). The buttons above will open the materials on Colab. If you are using Chameleon, start by running the [reserve.ipynb](https://github.com/kyrillosishak/re-SkinCancer/blob/main/reserve.ipynb) notebook inside the Chameleon Jupyter environment.


---

This project was part of the 2024 Summer of Reproducibility organized by the [UC Santa Cruz Open Source Program Office](https://ucsc-ospo.github.io/).

* Contributor: [Kyrillos Ishak](https://github.com/kyrillosishak)
* Mentors: [Fraida Fund](https://github.com/ffund), [Mohamed Saeed](https://github.com/mohammed183)