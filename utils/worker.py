class Worker(object):
    def __init__(self, network, quantizer, batch_size, two_phases=False):
        self.two_phases = two_phases
        self.net = network
        self.batch_size = batch_size
        self.dataset = self.net.dataset
        self.quantizer = quantizer

    def compute_gradients(self):
        xs, ys = self.dataset.train.next_batch(self.batch_size)
        g = self.net.compute_gradients(xs, ys)
        return self.quantizer.encode(g)

    def apply_gradients(self, gradients_):
        decompressed = self.quantizer.decode(gradients_)
        if self.two_phases:
            compressed = self.quantizer.encode(decompressed)
            decompressed = self.quantizer.decode([compressed])
        self.net.apply_gradients(decompressed)

    def test_loss_accuracy(self):
        test_xs_, test_ys_ = self.dataset.test.next_batch(self.batch_size)
        loss_, accuracy_ = self.net.compute_loss_accuracy(test_xs_, test_ys_)
        return loss_, accuracy_

    def train_loss_accuracy(self):
        test_xs_, test_ys_ = self.dataset.valid.next_batch(self.batch_size)
        loss_, accuracy_ = self.net.compute_loss_accuracy(test_xs_, test_ys_)
        return loss_, accuracy_
