# MNIST CNN Classifier in PyTorch
Based on this [NextJournal Article](https://nextjournal.com/gkoehler/pytorch-mnist) by Gregor Koehler.

This example is broken down into several distinct sections:
1. [Definition of Variables](#definition-of-variables)
2. [Loading Dataset](#loading-datasets)
3. [Analysing the Dataset](#analysing-the-dataset)
4. [Building the Model](#building-the-model)

## Definition of Variables
- Trivially, import the libraries that we need:
  ```python
  import torch, torchvision
  ```
- This example starts off by defining some variables that we'll use later:
  ```python
  n_epochs = 3  # Number of times to loop over the training dataset.
  batch_size_train = 64
  batch_size_test = 1000
  learning_rate = 0.01 # Hyperparameter for optimiser
  momentum = 0.5       # Hyperparameter for optimiser
  log_interval = 10
  ```

  | Variable Name      | Description                                        |
  | ------------------ | -------------------------------------------------- |
  | `n_epochs`         | Number of times to loop over the training datasets |
  | `batch_train_size` | Number of samples to use for training              |
  | `batch_test_size`  | Number of samples to use for testing               |
  | `learning_rate`    | Hyperparameter for optimiser (to be used later)    |
  | `momentum`         | parameter for optimiser (to be used later)         |
  | `log_interval`     | --                                                 |

- In this section, we also `seed` the random number generator of PyTorch so that the experiments are repeatable.
- We will also use the `random_seed` variable for `random` and `numpy` later.
  ```python
  random_seed = 1
  torch.manual_seed(random_seed)
  ```
- We also disable cuDNN using non-deterministic algorithms using the following line
  ```python
  torch.backends.cudnn.enabled = False
  ```

## Loading Datasets
- Here, we use **TorchVision** to load the datasets.
  - Use `batch_train_size=64` for training and `batch_test_size=1000` for testing
  - The values `0.1307` and `0.3081` are global mean and standard deviation of the MNIST dataset.
  ```python
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('/files/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('/files/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_test, shuffle=True)
  ```
- When run, this section downloads the MNIST dataset into the `/files/` subdirectory.


## Analysing the Dataset
- Using the `test_loader` defined before, we can analyse the dataset.
  ```python
  example = enumerate(test_loader)
  batch_idx, (example_data, example_targets) = next(example)
  ```
- The `example_data` is a variable that has shape `torch.Size([1000, 1, 28, 28])`
  - That means that we have 1000 examples
  - The `1` signifies that there's one colour channel (black and white); no RGB
  - The images are `28x28`
---
- We can use `matplotlib` to plot the MNIST images:
```python
import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2, 3, i+1) #row, column, subplot number / slot
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
```

![MNIST_MatPlotLib_Samples](/assets/MNIST_MatPlotLib_Samples.svg)

## Building the Model
