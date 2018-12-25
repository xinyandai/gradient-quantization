import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import CNN, ResNet18
from quantizers import SGDQuantizer
from quantizers import HyperSphereQuantizer
from quantizers import QSGDQuantizer
from quantizers import NearestNeighborQuantizer
from dataloaders import minst, cifar10
from logger import Logger


NETWORK = ResNet18
QUANTIZER = HyperSphereQuantizer
DATASET_LOADER = cifar10
LOSS_FUNC = nn.CrossEntropyLoss()
LOGGER = None


def get_config(args):
    global QUANTIZER
    global LOGGER
    global NETWORK
    global DATASET_LOADER
    if args.quantizer == 'sgd':
        QUANTIZER = SGDQuantizer
    elif args.quantizer == 'qsgd':
        QUANTIZER = QSGDQuantizer
    elif args.quantizer == 'hsq':
        QUANTIZER = HyperSphereQuantizer
    elif args.quantizer == 'nnq':
        QUANTIZER = NearestNeighborQuantizer
    else:
        assert False, "no quantizer {}".format(args.quantizer)

    if args.save_log != None:
        LOGGER = Logger(args.save_log)

    if args.network == 'resnet':
        NETWORK = ResNet18
    elif args.network == 'cnn':
        NETWORK = CNN
    else:
        assert False, "no network {}".format(args.network)

    if args.dataset == 'cifar10':
        DATASET_LOADER = cifar10
    elif args.dataset == 'mnist':
        DATASET_LOADER = minst
    else:
        assert False, "no dataset {}".format(args.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Gradient Quantization Samples')
    parser.add_argument('--network', type=str, default='resnet')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--quantizer', type=str, default='hsq')
    parser.add_argument('--num-users', type=int, default=8, metavar='N',
                        help='num of users for training (default: 8)')
    parser.add_argument('--save-log', type=str, default=None,
                        help='For Saving the logs')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 350)')
    # parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
    #                     help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    get_config(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader, test_loader = DATASET_LOADER(args)
    model = NETWORK().to(device)
    quantizer = QUANTIZER(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=args.momentum, weight_decay=5e-4)

    epochs = [51, 101, 201]
    lrs = [0.01, 0.001, 0.0005]

    for epoch in range(1, args.epochs + 1):
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
        if batch_idx % args.log_interval == 0:
            test_accuracy = test(args, model, device, test_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Test Accuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
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
