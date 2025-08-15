"""Datasets."""
from .manifolds import Spheres
from .mnist import MNIST
from .fashion_mnist import FashionMNIST
from .cifar10 import CIFAR
from .blood import BloodMNIST
__all__ = ['Spheres', 'MNIST', 'FashionMNIST', 'CIFAR','BloodMNIST' ]
