from config_NYU import args as args_config
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)

from lib.summary.cfsummary import CompletionFormerSummary
from lib.metric.cfmetric import CompletionFormerMetric
from lib.data import get as get_data

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from lib.model.backbone import Backbone
from lib.model.SDR import SDR
from lib.model.MSPN import MSPN as spn


# Minimize randomness
def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_args(args):
    new_args = args
    current_time = time.strftime('%y%m%d_%H%M%S_')
    new_args.save_dir = new_args.log_dir + current_time + '{}_{}'.format(args.mode, args.data_name)

    if args.mode == 'SDR':
        if args.data_name == 'NYU':
            new_args.sparse_depths = [5, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    elif args.mode == 'conventional':
        if args.data_name == 'NYU':
            new_args.sparse_depths = [500]

    return new_args


def test(args):
    # Prepare dataset
    data = get_data(args)

    data_test = data(args, 'test')

    loader_test = DataLoader(dataset=data_test, batch_size=1,
                             shuffle=False, num_workers=args.num_threads)

    # Network
    net = SDR(args, Backbone, spn)

    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            # raise KeyError
        print('Checkpoint loaded from {}!'.format(args.pretrain))

    net = nn.DataParallel(net)

    metric = CompletionFormerMetric(args)

    try:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/test', exist_ok=True)
        os.makedirs(args.save_dir + '/csv', exist_ok=True)
    except OSError:
        pass

    writer_test = CompletionFormerSummary(args.save_dir, 'test', args, None, metric.metric_name)

    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    pbar = tqdm(total=num_sample)

    t_total = 0

    init_seed()
    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() for key, val in sample.items()
                  if val is not None}

        t0 = time.time()
        with torch.no_grad():
            output = net(sample)
        t1 = time.time()

        t_total += (t1 - t0)

        metric_val = metric.evaluate(sample, output, 'test')

        writer_test.add(None, metric_val)

        # Save data for analysis
        if args.save_image:
            writer_test.save(args.epochs, batch, sample, output)

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test'.format(current_time)
        if batch % args.print_freq == 0:
            pbar.set_description(error_str)
            pbar.update(loader_test.batch_size)

    pbar.close()

    writer_test.update(args.epochs, sample, output)

    t_avg = t_total / num_sample
    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))


def main(args):
    args = update_args(args)
    init_seed()

    for sd in args.sparse_depths:
        args.num_sample_test = sd
        test(args)


if __name__ == '__main__':
    args_main = update_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    main(args_main)