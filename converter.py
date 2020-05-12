import os
import shutil
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])
    max_ = np.min([len(out[tag]) for tag in tags])
    for tag in tags:
        out[tag] = out[tag][:max_]
    return out, steps


def to_csv(dpath):
    dirs = os.listdir(dpath)

    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
        df.to_csv(get_file_path(dpath, tag))


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


def convert_all():
    path = "logs"
    for model in os.listdir(path):
        model_path = path + "/" + model
        for dateset in os.listdir(model_path):
            dateset_path = model_path + "/" + dateset
            for quantizer in os.listdir(dateset_path):
                quantizer_path = dateset_path + "/" + quantizer
                try:
                    if os.listdir(quantizer_path):
                        # print(quantizer_path)
                        shutil.rmtree(quantizer_path+"/csv", ignore_errors=True)
                        to_csv(quantizer_path)
                except:
                    print("error {}".format(quantizer_path))


def convert(quantizer_path):
    shutil.rmtree(quantizer_path + "/csv", ignore_errors=True)
    to_csv(quantizer_path)

# convert("logs/vgg19/cifar10/nnq_d8_k8_n6_u8_b32_log_1")
# convert("logs/resnet101/cifar10/sgd_d32_k8_n6_u8_b32_log_1")
# convert("logs/resnet101/cifar10/qsgd_d0_k8_n1_u8_b32_log_1")
if __name__ == "__main__":
    import sys
    if (len(sys.argv)) > 1:
        convert(sys.argv[1])
    else:
        convert_all()
