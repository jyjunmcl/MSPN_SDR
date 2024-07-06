import argparse


parser = argparse.ArgumentParser(description='MSPN')


# Dataset
parser.add_argument('--dir_data',
                    type=str,
                    default='/media/jyjun/b40d9ff6-71a9-42f6-9a4b-222e2ab503bb/y2024_CVPR/dataset_DC/kitti_depth/',
                    help='path to dataset')
parser.add_argument('--data_name',
                    type=str,
                    default='KITTIDC',
                    choices=('NYU', 'KITTIDC'),
                    help='dataset name')
parser.add_argument('--split_json',
                    type=str,
                    default='data_json/kitti_dc_1e4.json',
                    help='path to json file')
parser.add_argument('--patch_height',
                    type=int,
                    # default=228,
                    default=240,
                    help='height of a patch to crop')
parser.add_argument('--patch_width',
                    type=int,
                    # default=304,
                    default=1216,
                    help='width of a patch to crop')
parser.add_argument('--top_crop',
                    type=int,
                    # default=0,
                    default=100,
                    help='top crop size for KITTI dataset')


# Hardware
parser.add_argument('--seed',
                    type=int,
                    default=43,
                    help='random seed point')
parser.add_argument('--gpus',
                    type=str,
                    default="0",
                    help='visible GPUs')
parser.add_argument('--port',
                    type=str,
                    default='29500',
                    help='master port')
parser.add_argument('--address',
                    type=str,
                    default='localhost',
                    help='master address')
parser.add_argument('--num_threads',
                    type=int,
                    default=4,
                    help='number of threads')
parser.add_argument('--no_multiprocessing',
                    action='store_true',
                    default=False,
                    help='do not use multiprocessing')


# Network
parser.add_argument('--prop_time',
                    type=int,
                    default=6,
                    help='number of propagation')

# Training
parser.add_argument('--loss',
                    type=str,
                    default='1.0*SILog',
                    help='loss function configuration')
parser.add_argument('--epochs',
                    type=int,
                    default=36,
                    help='number of epochs to train')
parser.add_argument('--milestones',
                    nargs="+",
                    type=int,
                    default=[6, 12, 18, 24, 30],
                    help='learning rate decay schedule')
parser.add_argument('--opt_level',
                    type=str,
                    default='O0',
                    choices=('O0', 'O1', 'O2', 'O3'))
parser.add_argument('--resume',
                    # default=True,
                    action='store_true',
                    help='resume training')
parser.add_argument('--test_only',
                    action='store_true',
                    help='test only flag')
parser.add_argument('--batch_size',
                    type=int,
                    default=3,
                    help='input batch size for training')
parser.add_argument('--max_depth',
                    type=float,
                    default=90.0,
                    help='maximum depth')
parser.add_argument('--augment',
                    type=bool,
                    default=True,
                    help='data augmentation')
parser.add_argument('--no_augment',
                    action='store_false',
                    dest='augment',
                    help='no augmentation')
parser.add_argument('--num_sample',
                    type=int,
                    default=500,
                    help='number of sparse samples')
parser.add_argument('--lidar_lines',
                    type=int,
                    default=64,
                    help='the extracted lidar lines')
parser.add_argument('--test_crop',
                    action='store_true',
                    default=True,
                    help='crop for test')


# Summary
parser.add_argument('--num_summary',
                    type=int,
                    default=4,
                    help='maximum number of summary images to save')


# Optimizer
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='learning rate')
parser.add_argument('--gamma',
                    type=float,
                    default=0.5,
                    help='learning rate multiplicative factors')
parser.add_argument('--optimizer',
                    default='ADAMW',
                    choices=('SGD', 'ADAM', 'ADAMW', 'RMSPROP'),
                    help='optimizer to use (SGD | ADAM | RMSprop | ADAMW)')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas',
                    type=tuple,
                    default=(0.9, 0.999),
                    help='ADAM | ADAMW beta')
parser.add_argument('--epsilon',
                    type=float,
                    default=1e-8,
                    help='ADAM | ADAMW epsilon for numerical stability')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.01,
                    help='weight decay')
parser.add_argument('--warm_up',
                    action='store_true',
                    default=True,
                    help='do lr warm up during the 1st epoch')
parser.add_argument('--no_warm_up',
                    action='store_false',
                    dest='warm_up',
                    help='no lr warm up')

# Logs
parser.add_argument('--log_dir',
                    type=str,
                    default='experiments/',
                    help='dir for log')
parser.add_argument('--print_freq',
                    type=int,
                    default=1,
                    help='print frequency of tqdm')
parser.add_argument('--save_full',
                    action='store_true',
                    default=True,
                    help='save optimizer, scheduler and amp in '
                         'checkpoints (large memory)')
parser.add_argument('--save_image',
                    action='store_true',
                    default=True,
                    help='save images for test')
parser.add_argument('--save_result_only',
                    action='store_true',
                    default=True,
                    help='save result images only with submission format')
parser.add_argument('--save_result_npy',
                    action='store_true',
                    default=False,
                    help='save result as npy format')


# Settings
parser.add_argument('--mode',
                    type=str,
                    default='SDR',
                    choices=('SDR', 'conventional'),
                    help='choose task')

parser.add_argument('--embed_dim',
                    type=int,
                    default=64,
                    help='channels of guidance feature')
parser.add_argument('--num_lidar_lines_train',
                    type=int,
                    default=64,
                    help='number of lidar lines for training')
parser.add_argument('--num_lidar_lines_test',
                    type=int,
                    default=64,
                    help='number of lidar lines for test')
parser.add_argument('--train_with_random_sds',
                    type=bool,
                    default=True,
                    help='training with randomly sampled number of sparse depths')
parser.add_argument('--pretrain',
                    type=str,
                    default='test_models/SDR_KITTI.pt',
                    help='test model dir')


args = parser.parse_args()
args.num_gpus = len(args.gpus.split(','))
