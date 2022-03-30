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
- For this example, we're using a Convolutional Neural Network (CNN) with:
  - Two 2D convolution layers
  - Two fully-connected (linear) layers.
  - `Activation Function` ReLU (Rectified Linear Units). The ReLU function can be represented by:
    $$ R(x)=\left\{\begin{matrix}y,&&x>=0\\0,&&x<0\end{matrix} \right. $$
  - Use two `dropout layers` for regularisation
---
- To represent this model in PyTorch, we'll create a class:
  ```python
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim

  class Net(nn.module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
  ```
### Breakdown of Class Definition
- The nn.[Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) is a method that creates a 2D convolution layer.
  ```python
  # nn.Conv2d(in_channels, out_channels, kernel_size)
  self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
  self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
  ```
  | Variable                       | Variable Description                  |
  | ------------------------------ | ------------------------------------- |
  | `in_channels`(int)             | # of channels in the input range      |
  | `out_channels`(int)            | # of channels produced by convolution |
  | `kernel_size` (int, int x int) | Size of convolving kernel             |

- The `torch.nn.*` layers are layers that contain trainable parameters whilst the `torch.nn.functional` layers are purely functional layers.
- The `forward(...)` function defines the way that the output is computed using the given layers and functions.

### Initialise Network and Optimiser
```python
network = Net()
optimiser = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
```

### Build the Training Loop
- Workflow for training CNN
  1. Ensure that the model is in `training mode`
  2. Iterate over training data once per `epoch`. (Loading of individual batches of data is handled by the `DataLoader`)
     - Manually set the gradients to zero using `optimizer.zero_grad()`
  3. Prodduce output of network (forward pass) then compute `negative log-likelihood loss` between output and griound truth.
  4. Call `backward()` method to backpropagate new set of gradients into each of the network's parameters using `optimizer.step()`

  ```python
  # Keep track off some statistics for easier debugging later.
  train_losses = []
  train_counter = []
  test_losses = []
  test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

  def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = network(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
          (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        torch.save(network.state_dict(), '/results/model.pth')
        torch.save(optimizer.state_dict(), '/results/optimizer.pth')
  ```
