import matplotlib.pyplot as plt
import numpy as np

fontsize=44
ticksize=40
legendsize=30
plt.style.use('seaborn-white')

# network, parameters = "resnet50", 23520842
# network, parameters = "resnet101", 42512970
network, parameters = "vgg19", 20040522
dataset = "cifar10"
plot_accuracy = True
plot_epoch = True
num_users = 8

def accuracy(quantizer):
    return "logs/{}/{}/{}/csv/accuracy(%).csv".format(network, dataset, quantizer)

def loss(quantizer):
    return "logs/{}/{}/{}/csv/loss.csv".format(network, dataset, quantizer)

def read_csv(file_name):
    return np.genfromtxt(fname=file_name, delimiter=',', skip_header=1)


def _plot_setting():
    plt.xlabel('# Epochs' if plot_epoch else "Communication Cost(GB)", fontsize=fontsize)
    plt.ylabel('Accuracy' if plot_accuracy else "Loss", fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize)

    plt.legend(loc='lower right' if plot_accuracy else 'upper right', fontsize=legendsize)

    plt.show()

def plot_one(quantizer, label, color, linestyle, marker, lines=150, cr=1.0):
    try:
        data = read_csv(accuracy(quantizer) if plot_accuracy else loss(quantizer))
        y = np.array(data[0:lines, 1])
        x = np.arange(1, len(y)+1).astype(np.float64)
        if not plot_epoch:
            x = x * ( parameters * 4.0 * num_users * ( 1.0 / cr ) / (2**30 + 0.0) )

        # y = np.sort(y) if plot_accuracy else  np.sort(-y)
        # label = "%s (, %.2f%s)" % (label, np.max(y), "%")
        # label = label.replace(') (', ' ')
        plt.plot(x, y, color, label=label, linestyle=linestyle, marker=marker)
    except Exception as e:
        print(e)

def plot_mnist():
    # plt.subplot(1, 3, 1)
    lines = 150
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch
    plot_epoch = False
    network, parameters = "fcn", 42512970
    dataset = 'mnist'
    lines = 90
    plot_one('nnq_d8_k8_n6_u8_b32_log_1',      'HSQ(d=8)',    'red', '--', ' ', lines=lines, cr=(32. * 32) / (8 + 6))
    plot_one('nnq_d16_k8_n6_u8_b32_log_1',      'HSQ(d=16)',    'red',  '-',  ' ', lines=lines, cr=(16.*32) / (8 + 6))
    # plot_one('nnq_d8_k8_n6_u8_b32_log_1',       'HSQ(d=8)', 'red', ':', ' ', lines=lines, cr=(8. * 32) / (8 + 6))
    plot_one('nnq_d32_k8_n6_u8_b32_log_1',     'HSQ(d=32)',      'gray', '-', ' ', lines=lines, cr=32.)

    plot_one('qsgd_d0_k8_n1_u8_b32_log_1',      'TernGrad',     'blue', '-.', ' ', lines=lines, cr=32. / 2)
    plot_one('qsgd_d128_k8_n2_u8_b32_log_1',    'QSGD(2bit)',   'blue', '-.', ' ', lines=lines, cr=32./3)
    plot_one('qsgd_d512_k8_n4_u8_b32_log_1',    'QSGD(4bit)',   'blue', '--', ' ', lines=lines, cr=32. / 5)
    plot_one('qsgd_d512_k8_n8_u8_b32_log_1',    'QSGD(8bit)',   'blue', '-', ' ', lines=lines, cr=32. / 9)
    plot_one('sgd_d32_k8_n6_u8_b32_log_1',      'SGD',          'black', '-.', ' ', lines=lines, cr=1.0)

    _plot_setting()

def plot_():
    # plt.subplot(1, 3, 1)
    lines = 150
    plot_one('nnq_d64_k8_n6_u8_b32_log_1',      'HSQ(d=64)',    'red', '--', ' ', lines=lines, cr=(32. * 32) / (8 + 6))
    plot_one('nnq_d16_k8_n6_u8_b32_log_1',      'HSQ(d=16)',    'red',  '-',  ' ', lines=lines, cr=(16.*32) / (8 + 6))
    # plot_one('nnq_d8_k8_n6_u8_b32_log_1',       'HSQ(d=8)', 'red', ':', ' ', lines=lines, cr=(8. * 32) / (8 + 6))
    plot_one('sign_d32_k8_n6_u8_b32_log_1',     'SignSGD',      'gray', '-', ' ', lines=lines, cr=32.)

    plot_one('qsgd_d0_k8_n1_u8_b32_log_1',      'TernGrad',     'blue', '-.', ' ', lines=lines, cr=32. / 2)
    # plot_one('qsgd_d128_k8_n2_u8_b32_log_1',    'QSGD(2bit)',   'blue', '-.', ' ', lines=lines, cr=32./3)
    plot_one('qsgd_d512_k8_n4_u8_b32_log_1',    'QSGD(4bit)',   'blue', '--', ' ', lines=lines, cr=32. / 5)
    plot_one('qsgd_d512_k8_n8_u8_b32_log_1',    'QSGD(8bit)',   'blue', '-', ' ', lines=lines, cr=32. / 9)
    plot_one('sgd_d32_k8_n6_u8_b32_log_1',      'SGD',          'black', '-.', ' ', lines=lines, cr=1.0)

    _plot_setting()

def plot_cifar100():
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch
    plot_epoch = False
    network, parameters = "resnet101", 42512970
    dataset = 'cifar100'
    lines = 90
    # plot_one('nnq_d64_k8_n6_u8_b32_log_1',      'HSQ(d=64)',    'red', '--', ' ', lines=lines, cr=(32. * 32) / (8 + 6))
    plot_one('nnq_d16_k8_n6_u8_b32_log_1',      'HSQ(d=16)',    'red',  '--',  ' ', lines=lines, cr=(16.*32) / (8 + 6))
    plot_one('nnq_d8_k8_n6_u8_b32_log_1',       'HSQ(d=8)',         'red', '-', ' ', lines=lines, cr=(8. * 32) / (8 + 6))
    # plot_one('sign_d32_k8_n6_u8_b32_log_1',     'SignSGD',      'gray', '-', ' ', lines=lines, cr=32.)

    plot_one('qsgd_d0_k8_n1_u8_b32_log_1',      'TernGrad',     'blue', '-.', ' ', lines=lines, cr=32. / 2)
    # plot_one('qsgd_d128_k8_n2_u8_b32_log_1',    'QSGD(2bit)',   'blue', '-.', ' ', lines=lines, cr=32./3)
    # plot_one('qsgd_d512_k8_n4_u8_b32_log_1',    'QSGD(4bit)',   'blue', '--', ' ', lines=lines, cr=32. / 5)
    plot_one('qsgd_d512_k8_n8_u8_b32_log_1',    'QSGD(8bit)',   'blue', '-', ' ', lines=lines, cr=32. / 9)
    plot_one('sgd_d32_k8_n6_u8_b32_log_1',      'SGD',          'black', '-.', ' ', lines=lines, cr=1.0)
    _plot_setting()



def plot_more():
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch

    lines = 150
    dataset = "cifar10"
    network = "resnet50"
    plot_epoch = False
    # plt.subplot(1, 3, 1)
    lines = 150

    plot_one('nnq_d16_k8_n6_u8_b32_log_1',      'HSQ(d=16)',    'red',  '-',  ' ', lines=lines, cr=(16.*32) / (8 + 6))
    plot_one('qsgd_d128_k8_n2_u8_b32_log_1',    'QSGD(2bit)',   'blue', '-', ' ', lines=lines, cr=32./3)
    plot_one('sgd_d32_k8_n6_u8_b32_log_1',      'SGD',          'black', '-.', ' ', lines=lines, cr=1.0)
    #
    # plot_one('nnq_d512_k0_n6_u8_b32_log_1',   'HSQ(d=512)',  'red',  '-.', ' ',  lines=lines)
    # plot_one('nnq_d256_k0_n6_u8_b32_log_1',   'HSQ(d=256)',  'gray',  '-.', ' ', lines=lines)
    # plot_one('nnq_d128_k0_n6_u8_b32_log_1',   'HSQ(d=128)',  'red', '--', ' ',  lines=lines)
    # plot_one('nnq_d32_k5_n6_u8_b32_log_1',    'HSQ(d=32)',   'blue', ':', ' ',   lines=lines)
    # plot_one('nnq_d16_k8_n6_u8_b32_log_1',    'HSQ(d=16)',   'red', '-', ' ',   lines=lines)
    # plot_one('sgd_d32_k8_n6_u8_b32_log_1',    'SGD',         'black', '-', ' ',   lines=lines)

    _plot_setting()

def plot_varies_norm():
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch

    lines = 150
    dataset = "cifar10"
    network = "resnet50"
    plot_one('nnq_d32_k8_n2_u8_b32_log_1', 'HSQ(2bit)', 'red', '--', ' ', lines=lines)
    plot_one('nnq_d32_k8_n4_u8_b32_log_1', 'HSQ(4bit)', 'black', '-', ' ', lines=lines)
    plot_one('nnq_d32_k8_n6_u8_b32_log_1', 'HSQ(6bit)', 'red', '-', ' ', lines=lines)
    plot_one('nnq_d32_k8_n32_u8_b32_log_1', 'HSQ(32bit)', 'gray', '--', ' ', lines=lines)
    plot_one('sgd_d32_k8_n6_u8_b32_log_1', 'SGD', 'blue', '-', ' ', lines=lines)
    _plot_setting()



def plot_mnist():
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch

    plot_epoch = True

    network, parameters = "fcn", 50
    dataset = 'mnist'

    lines = 150

    plot_one('nnq_d64_k8_n6_u8_b32_log_1',      'HSQ(d=64)',    'red', '-', ' ', lines=lines, cr=(32. * 32) / (8 + 6))
    plot_one('nnq_d16_k8_n6_u8_b32_log_1',      'HSQ(d=16)',    'red',  '-.',  ' ', lines=lines, cr=(16.*32) / (8 + 6))
    plot_one('nnq_d8_k8_n6_u8_b32_log_1',       'HSQ(d=8)', 'red', '-', ' ', lines=lines, cr=(8. * 32) / (8 + 6))

    plot_one('qsgd_d0_k8_n1_u8_b32_log_1',      'TernGrad',     'blue', '-.', ' ', lines=lines, cr=32. / 2)
    # plot_one('qsgd_d128_k8_n2_u8_b32_log_1',    'QSGD(2bit)',   'blue', '-.', ' ', lines=lines, cr=32./3)
    plot_one('qsgd_d512_k8_n4_u8_b32_log_1',    'QSGD(4bit)',   'black', '-.', ' ', lines=lines, cr=32. / 5)
    plot_one('qsgd_d512_k8_n8_u8_b32_log_1',    'QSGD(8bit)',   'black', '--', ' ', lines=lines, cr=32. / 9)
    plot_one('sgd_d32_k8_n6_u8_b32_log_1',      'SGD',          'blue', '-', ' ', lines=lines, cr=1.0)
    _plot_setting()


def plot_varies_K():
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch

    network, parameters = "resnet50", 0
    dataset = 'cifar10'
    plot_epoch = True

    lines = 150


    plot_one('sgd_d32_k8_n6_u8_b32_log_1',  'SGD',          'gray', '-', ' ', lines=lines)
    plot_one('nnq_d32_k5_n6_u8_b32_log_1',  'HSQ(m=32)',   'blue', ':', ' ',  lines=lines)
    plot_one('nnq_d32_k8_n6_u8_b32_log_1',  'HSQ(m=256)',  'black', '--', ' ',  lines=lines)
    plot_one('nnq_d32_k10_n6_u8_b32_log_1', 'HSQ(m=1024)', 'red',  '-.', ' ', lines=lines)
    # plot_one('qsgd_d4096_k8_n2_u8_b32_log_1', '2bit QSGD', 'gray',  '-', '+', lines=lines)

    _plot_setting()

def plot_varies_codebook():
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch

    plot_epoch = True

    network, parameters = "resnet50", 50
    dataset = 'cifar10'

    lines = 150

    plot_one('nnq_d32_k5_n6_u8_b32_log_1_randomcodebook',   'HSQ(Gaussian)',     'blue',  ':',  ' ', lines=lines)
    # plot_one('nnq_d32_k8_n6_u8_b32_log_1_randomcodebook2',  'HSQ(Gaussian-256)',    'blue',  '-',  ' ', lines=lines)
    plot_one('nnq_d32_k5_n6_u8_b32_log_1',                  'HSQ(RR-1)', 'gray',   '-',  ' ', lines=lines)
    plot_one('nnq_d32_k5_n6_u8_b32_log_1_orthongal',        'HSQ(RR-2)', 'gray',   ':',  ' ', lines=lines)
    plot_one('nnq_d32_k5_n6_u8_b32_log_1_kmeanscodebook',   'HSQ(KMeans)',       'black', '-',  ' ', lines=lines)
    # plot_one('nnq_d32_k8_n6_u8_b32_log_1',                  'HSQ(KMeans-256)',      'black', '-',  ' ', lines=lines)
    plot_one('nnq_d32_k5_n6_u8_b32_log_1_identitycodebook', 'HSQ(SOB)',          'red',   '-.', ' ', lines=lines)
    # plot_one('qsgd_d4096_k8_n2_u8_b32_log_1', '2bit QSGD', 'gray',  '-', '+', lines=lines)

    _plot_setting()

def plot_varies_dimension():
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch

    network, parameters = "resnet50", 50
    dataset = 'cifar10'
    plot_epoch = True
    plot_accuracy = False

    lines = 150
    plot_one('nnq_d512_k0_n6_u8_b32_log_1',   'HSQ(d=512)',  'red',  '-.', ' ',  lines=lines)
    plot_one('nnq_d256_k0_n6_u8_b32_log_1',   'HSQ(d=256)',  'gray',  '-.', ' ', lines=lines)
    plot_one('nnq_d128_k0_n6_u8_b32_log_1',   'HSQ(d=128)',  'black', '-', ' ',  lines=lines)
    plot_one('nnq_d32_k5_n6_u8_b32_log_1',    'HSQ(d=32)',   'blue', ':', ' ',   lines=lines)
    plot_one('nnq_d16_k8_n6_u8_b32_log_1',    'HSQ(d=16)',   'red', '--', ' ',   lines=lines)
    plot_one('sgd_d32_k8_n6_u8_b32_log_1',    'SGD',         'gray', '-', ' ',   lines=lines)

    _plot_setting()


def show_accuracy(quantizer, x=0, y=0):
    try:
        data = read_csv(accuracy(quantizer))
        accuracy_ = np.array(data[:, 1])
        a, b = np.max(accuracy_), np.max(accuracy_[-30:])
        # print("{} {} {} ".format(quantizer, a - x, b - y))
        print("&%.2f" % b, end=' ')
        return a, b
    except:
        pass

def show_accuracies():
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch

    network, parameters = "resnet50", 0

    for network in ["vgg19", "resnet50", "resnet101"]:
        for dataset in ['cifar10']:
            a, b = 0, 0
            show_accuracy('sgd_d32_k8_n6_u8_b32_log_1')
            show_accuracy('sign_d32_k8_n6_u8_b32_log_1', a, b)
            show_accuracy('qsgd_d0_k8_n1_u8_b32_log_1', a, b)
            show_accuracy('qsgd_d128_k8_n2_u8_b32_log_1', a, b)
            show_accuracy('qsgd_d512_k8_n4_u8_b32_log_1', a, b)
            show_accuracy('qsgd_d512_k8_n8_u8_b32_log_1', a, b)

            show_accuracy('nnq_d8_k8_n6_u8_b32_log_1', a, b)
            show_accuracy('nnq_d16_k8_n6_u8_b32_log_1', a, b)
            # show_accuracy('nnq_d32_k8_n6_u8_b32_log_1', a, b)
            show_accuracy('nnq_d64_k8_n6_u8_b32_log_1', a, b)

            print()


def show_accuracies_veries_dim():
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch

    network, parameters = "resnet50", 0

    for network in ["resnet101"]:
        for dataset in ['cifar100']:
            a, b = 0, 0
            show_accuracy('sgd_d32_k8_n6_u8_b32_log_1')
            show_accuracy('nnq_d8_k8_n6_u8_b32_log_1', a, b)
            show_accuracy('nnq_d16_k8_n6_u8_b32_log_1', a, b)
            show_accuracy('nnq_d32_k5_n6_u8_b32_log_1', a, b)
            show_accuracy('nnq_d128_k0_n6_u8_b32_log_1', a, b)
            show_accuracy('nnq_d256_k0_n6_u8_b32_log_1', a, b)

            show_accuracy('nnq_d512_k0_n6_u8_b32_log_1', a, b)

            print()

def show_accuracies_veries_dim():
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch

    network, parameters = "resnet50", 0

    for network in ["vgg19", "resnet50", "resnet101"]:
        for dataset in ['cifar10']:
            a, b = 0, 0
            show_accuracy('sgd_d32_k8_n6_u8_b32_log_1')
            show_accuracy('nnq_d8_k8_n6_u8_b32_log_1', a, b)
            show_accuracy('nnq_d16_k8_n6_u8_b32_log_1', a, b)
            show_accuracy('nnq_d32_k5_n6_u8_b32_log_1', a, b)
            show_accuracy('nnq_d128_k0_n6_u8_b32_log_1', a, b)
            show_accuracy('nnq_d256_k0_n6_u8_b32_log_1', a, b)

            show_accuracy('nnq_d512_k0_n6_u8_b32_log_1', a, b)

            print()


def plot_two_phase():
    # plt.subplot(1, 3, 1)
    global network
    global parameters
    global dataset
    global plot_accuracy
    global plot_epoch

    network, parameters = "resnet50", 23520842
    plot_epoch = True
    plot_accuracy = True
    lines = 150
    plot_one('nnq_d32_k8_n6_u8_b32_log_1_two_phase',      'HSQ(d=64)',    'red', '--', ' ', lines=lines, cr=(32. * 32) / (8 + 6))
    plot_one('nnq_d16_k8_n6_u8_b32_log_1_two_phase',      'HSQ(d=16)',    'red',  '-',  ' ', lines=lines, cr=(16.*32) / (8 + 6))
    plot_one('nnq_d8_k8_n6_u8_b32_log_1_two_phase',       'HSQ(d=8)', 'red', ':', ' ', lines=lines, cr=(8. * 32) / (8 + 6))

    plot_one('qsgd_d0_k8_n1_u8_b32_log_1_two_phase',      'TernGrad',     'blue', '-.', ' ', lines=lines, cr=32. / 2)
    plot_one('qsgd_d128_k8_n2_u8_b32_log_1_two_phase',    'QSGD(2bit)',   'gray', '-.', ' ', lines=lines, cr=32./3)
    # plot_one('qsgd_d512_k8_n4_u8_b32_log_1_two_phase',    'QSGD(4bit)',   'blue', '--', ' ', lines=lines, cr=32. / 5)
    # plot_one('qsgd_d512_k8_n8_u8_b32_log_1_two_phase',    'QSGD(8bit)',   'blue', '-', ' ', lines=lines, cr=32. / 9)
    # plot_one('sgd_d32_k8_n6_u8_b32_log_1',      'SGD',          'black', '-.', ' ', lines=lines, cr=1.0)

    _plot_setting()


plot_two_phase()
# plot_cifar100()
# plot_varies_norm()
# plot_()
# plot_mnist()
# plot_more()
# show_accuracies()
# show_accuracies_veries_dim()
# plot_varies_K()
# plot_varies_dimension()
# plot_varies_codebook()
# /research/jcheng2/xinyan/anaconda3/bin/python main.py  --quantizer nnq  --network resnet50 --dataset cifar10 --num-users 8 --batch-size 32 --c-dim 32 --n-bit 6 --num-users 8 --k-bit 5
# /research/jcheng2/xinyan/anaconda3/bin/python main.py  --quantizer nnq  --network resnet50 --dataset cifar10 --num-users 8 --batch-size 32 --c-dim 32 --n-bit 6 --num-users 8 --k-bit 8
