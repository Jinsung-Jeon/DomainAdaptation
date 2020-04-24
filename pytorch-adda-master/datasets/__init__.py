from .mnist import get_mnist, get_mnist_set
from .usps import get_usps
from .svhn import get_svhn, get_svhn_set
from .stl10 import get_stl10
from .cifar10 import get_cifar10

__all__ = (get_usps, get_mnist, get_svhn, get_stl10, get_cifar10, get_svhn_set, get_mnist_set)
