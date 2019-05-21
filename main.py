import argparse
from datetime import datetime
import torch.optim as optim


from compressors import *
from dataloaders import *
from quantizers import *

from models import *
from logger import Logger


NETWORK         = None
COMPRESSOR      = None
DATASET_LOADER  = None
LOGGER          = None
LOSS_FUNC       = nn.CrossEntropyLoss()

quantizer_choices = {
    'sgd':  IdenticalCompressor,
    'qsgd': QSGDCompressor,
    'hsq':  NearestNeighborCompressor,
    'sign': SignSGDCompressor
}

network_choices = {
    'resnet18'  : ResNet18,
    'resnet34'  : ResNet34,
    'resnet50'  : ResNet50,
    'resnet101' : ResNet101,
    'resnet152' : ResNet152,
    'vgg11'     : vgg11,
    'vgg13'     : vgg13,
    'vgg16'     : vgg16,
    'vgg19'     : vgg19,
    'dense'     : densenet_cifar,
    'fcn'       : FCN
}

data_loaders = {
    'mnist':    minst,
    'cifar10':  cifar10,
    'cifar100': cifar100,
    'stl10':    stl10,
    'svhn':     svhn,
    'tinyimg':  tinyimgnet
}

classes_choices = {
    'mnist':    10,
    'cifar10':  10,
    'cifar100': 100,
    'stl10':    10,
    'svhn':     10,
    'tinyimg':  200
}


def get_config(args):
    global COMPRESSOR
    global LOGGER
    global NETWORK
    global DATASET_LOADER

    COMPRESSOR = quantizer_choices[args.quantizer]
    NETWORK = network_choices[args.network]
    DATASET_LOADER = data_loaders[args.dataset]
    args.num_classes = classes_choices[args.dataset]

    if args.logdir is None:
        assert False, "The logdir is not defined"
    LOGGER = Logger(args.logdir)

    args.no_cuda = args.no_cuda or not torch.cuda.is_available()


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Gradient Quantization Samples')
    parser.add_argument('--network', type=str, default='resnet18', choices=network_choices.keys())
    parser.add_argument('--dataset', type=str, default='cifar10', choices=data_loaders.keys())
    parser.add_argument('--num-classes', type=int, default=10, choices=classes_choices.values())
    parser.add_argument('--quantizer', type=str, default='hsq', choices=quantizer_choices.keys())

    parser.add_argument('--c-dim', type=int, default=32)
    parser.add_argument('--k-bit', type=int, default=8)
    parser.add_argument('--n-bit', type=int, default=8)
    parser.add_argument('--random', type=int, default=True)

    parser.add_argument('--num-users', type=int, default=8, metavar='N',
                        help='num of users for training (default: 8)')
    parser.add_argument('--logdir', type=str, default=None,
                        help='For Saving the logs')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 350)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                        help='weight decay momentum (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-epoch', type=int, default=1, metavar='N',
                        help='logging training status at each epoch')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--two-phase', action='store_true', default=False,
                        help='For Compression two phases')


    args = parser.parse_args()
    get_config(args)

    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda else "cuda")

    train_loader, test_loader = DATASET_LOADER(args)
    model = NETWORK(num_classes=args.num_classes).to(device)
    quantizer = Quantizer(COMPRESSOR, model.parameters(), args)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    if args.dataset == 'mnist':
        epochs = []
        lrs = []
        args.epochs = 20
    elif args.dataset == 'tinyimg':
        epochs = [51]
        lrs = [0.01]
        args.epochs = 1000
    else:
        epochs = [51, 71]
        lrs = [0.01, 0.005]
        args.epochs = 150

    if COMPRESSOR == SignSGDCompressor:
        epochs = [51, 71]
        lrs = [0.0005, 0.0001]
        args.epochs = 150
        args.momentum = 0.0
        args.weight_decay = 0.1
        optimizer = optim.SGD(model.parameters(), lr=1e-3,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 2):
        for i_epoch, i_lr in zip(epochs, lrs):
            if epoch == i_epoch:
                optimizer = optim.SGD(model.parameters(), lr=i_lr,
                                      momentum=args.momentum, weight_decay=5e-4)

        train(args, model, device, train_loader, test_loader,
              optimizer, quantizer, epoch)
        # origin_train(args, model, device, train_loader, optimizer, epoch)
        # test(args, model, device, test_loader)

    if args.save_model:
        filename = "saved_{}_{}.pt".format(args.network, datetime.now())
        torch.save(model.state_dict(), filename)


def train(args, model, device, train_loader, test_loader, optimizer, quantizer, epoch):
    global LOGGER
    model.train()
    batch_size = args.batch_size
    num_users = args.num_users
    train_data = list()
    iteration = len(train_loader.dataset)//(num_users*batch_size) + \
        int(len(train_loader.dataset) % (num_users*batch_size) != 0)
    log_interval = [iteration // args.log_epoch * (i+1) for i in range(args.log_epoch)]
    # here the real batch size is (num_users * batch_size)
    for batch_idx, (data, target) in enumerate(train_loader):
        user_batch_size = len(data) // num_users
        train_data.clear()

        for user_id in range(num_users-1):
            train_data.append((data[user_id*user_batch_size:(user_id+1)*user_batch_size],
                               target[user_id*user_batch_size:(user_id+1)*user_batch_size]))
        train_data.append((data[(num_users-1)*user_batch_size:],
                           target[(num_users-1)*user_batch_size:]))

        loss = one_iter(model, device, LOSS_FUNC, optimizer,
                        quantizer, train_data, num_users)
        if (batch_idx+1) in log_interval:
            test_accuracy = test(args, model, device, test_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Test Accuracy: {:.2f}%'.format(
                epoch,
                batch_idx * num_users * batch_size + len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(),
                test_accuracy*100))

            info = {'loss': loss.item(), 'accuracy(%)': test_accuracy*100}

            for tag, value in info.items():
                LOGGER.scalar_summary(
                    tag, value, iteration*(epoch-1)+batch_idx)

    print('Train Epoch: {} Done.\tLoss: {:.6f}'.format(epoch, loss.item()))


def one_iter(model, device, loss_func, optimizer, quantizer, train_data, num_users):
    assert num_users == len(train_data)
    model.train()
    user_gradients = [list() for _ in model.parameters()]
    all_losses = []
    for user_id in range(num_users):
        optimizer.zero_grad()
        _data, _target = train_data[user_id]
        data, target = _data.to(device), _target.to(device)
        pred = model(data)
        loss = loss_func(pred, target)
        # print(loss)
        all_losses.append(loss)
        loss.backward()
        quantizer.record()
    quantizer.apply()
    optimizer.step()
    return torch.stack(all_losses).mean()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += LOSS_FUNC(output, target).sum().item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def origin_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = LOSS_FUNC(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


if __name__ == "__main__":
    main()

