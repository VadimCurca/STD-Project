import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import random as rnd
from torch.utils.data import Dataset

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


def evaluate_accuracy(net, data_iter, loss, device):
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode

    total_loss = 0
    total_hits = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            total_loss += float(l)
            total_hits += sum(net(X).argmax(axis=1).type(y.dtype) == y)
            total_samples += y.numel()
    return float(total_loss) / len(data_iter), float(total_hits) / total_samples  * 100


class DistilationLoss(torch.nn.Module):
    def __init__(self, base_criterion, teacher_model, alpha = 0.5, tau = 1.0, distillation_type = 'none'):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.tau = tau
        self.distillation_type = distillation_type
        assert distillation_type in ['none', 'hard', 'soft']

    def forward(self, inputs, outputs, labels):
        outputs_kd = None

        # assume that the model outputs a tuple of [outputs, outputs_kd]
        outputs, outputs_kd = outputs

        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1)
            )

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


def train_epoch(net, train_iter, loss, optimizer, device):  
    # Set the model to training mode
    net.train()
    # Sum of training loss, sum of training correct predictions, no. of examples
    total_loss = 0
    total_hits = 0
    total_samples = 0
    for X, y in train_iter:
        # Compute gradients and update parameters
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        
        if not isinstance(loss, DistilationLoss):
            l = loss(y_hat, y)
        else:
            l = loss(X, y_hat, y)
        # Using PyTorch built-in optimizer & loss criterion
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total_loss += float(l)

        if not isinstance(loss, DistilationLoss):
            total_hits += sum(y_hat.argmax(axis=1).type(y.dtype) == y)
        else:
            total_hits += sum(y_hat[0].argmax(axis=1).type(y.dtype) == y)
        total_samples += y.numel()
    # Return training loss and training accuracy
    return float(total_loss) / len(train_iter), float(total_hits) / total_samples  * 100


def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device, loss = nn.CrossEntropyLoss(), evaluation_loss = nn.CrossEntropyLoss()):
    """Train a model."""
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('Training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer, device)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss, val_acc = evaluate_accuracy(net, val_iter, evaluation_loss, device)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)
        print(f'Epoch {epoch + 1}, Train loss {train_loss:.2f}, Train accuracy {train_acc:.2f}, Validation loss {val_loss:.2f}, Validation accuracy {val_acc:.2f}')
    test_loss, test_acc = evaluate_accuracy(net, test_iter, evaluation_loss, device)
    print(f'Test loss {test_loss:.2f}, Test accuracy {test_acc:.2f}')

    return train_loss_all, train_acc_all, val_loss_all, val_acc_all, test_acc, test_loss


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def plot_loss(train_loss_all, val_loss_all):
    epochs = range(1, len(train_loss_all) + 1) 
    plt.plot(epochs, train_loss_all, 'bo', label='Training loss') 
    plt.plot(epochs, val_loss_all, 'b', label='Validation loss') 
    plt.title('Training and validation loss') 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.legend()  
    plt.show()


def plot_accuracy(train_acc_all, val_acc_all):
    epochs = range(1, len(train_acc_all) + 1)
    plt.plot(epochs, train_acc_all, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_all, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs') 
    plt.ylabel('Accuracy') 
    plt.legend()
    plt.show()


class TargetTransform():
    def __init__(self, p):
        self.p = p

    def __call__(self, n):
        if n > 0:
            change_label = rnd.random() < self.p
            n = (rnd.randint(0, 9) if change_label else n)

        return n


def random_sample(n, num):
    a = torch.zeros(n, dtype=torch.bool)

    indices = rnd.sample(range(n), num)
    for x in indices:
        a[x] = True

    return a


class DatasetWrapper(Dataset):
    def __init__(self, dataset, error_rate = 0.0):
        self.dataset = dataset

        n = len(dataset)
        num = int(n * error_rate)
        self.p = random_sample(n, num)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.p[index]:
            newLabel = rnd.randint(0, 9)
            img = self.dataset[index][0]
            return img, newLabel

        return self.dataset[index]


def load_data_cifar10(batch_size, resize=None, error_rate=0.0):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = DatasetWrapper(torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=trans, download=True), error_rate=error_rate)
    mnist_test = DatasetWrapper(torchvision.datasets.CIFAR10(
        root="../data", train=False, transform=trans, download=True))
    mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [45000, 5000],
                                                           generator=torch.Generator().manual_seed(42))
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            torch.utils.data.DataLoader(mnist_val, batch_size, shuffle=False,
                            num_workers=0),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=2))


def load_data_fashion_mnist(batch_size, resize=None, error_rate=0.0):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = DatasetWrapper(torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True), error_rate=error_rate)
    mnist_test = DatasetWrapper(torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True))
    mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [50000, 10000],
                                                           generator=torch.Generator().manual_seed(42))
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            torch.utils.data.DataLoader(mnist_val, batch_size, shuffle=False,
                            num_workers=0),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=2))
