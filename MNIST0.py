"""
File created by Matt Choy on 30-Mar-22 10:01AM
This file is my attempt at recreating this MNIST CNN detector based on the
NextJournal article by Gregor Koehler (Read MNIST0.md for more details).
"""
import torch, torchvision

""" ========== Hyperparameters ========== """
n_epochs = 3  # Number of times to loop over the training dataset.
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01 # Hyperparameter for optimiser
momentum = 0.5       # Hyperparameter for optimiser
log_interval = 10

# We set a random seed so that it's deterministic / repeatable.
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

""" ========== Loading Dataset ========== """
# Here, we use TorchVision to load the dataset.
# 0.1307 -- Global mean of MNIST dataset
# 0.3081 -- Standard Deviation of MNIST dataset
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
