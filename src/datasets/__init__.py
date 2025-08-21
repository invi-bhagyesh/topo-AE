"""Datasets."""
from .manifolds import Spheres
from .mnist import MNIST
from .fashion_mnist import FashionMNIST
from .cifar10 import CIFAR
from .bloodmnist import BloodMNIST
from .PBMC import PBMC
from .emnist import EMNIST
from .syn import SYN
__all__ = ['Spheres', 'MNIST', 'FashionMNIST', 'CIFAR','BloodMNIST', 'PBMC', 'EMNIST', 'SYN']
