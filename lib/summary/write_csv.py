import csv
import os


def write_csv(args, save_fn, step, contents, values):
    if args.data_name == 'NYU':
        if not os.path.isfile(save_fn):
            print('{} created'.format(save_fn))
            with open(save_fn, mode='a') as validate_csv:
                writer = csv.writer(validate_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['epoch'] + ['num_samples'] + contents)
                writer.writerow([step] + [args.num_sample_test] + list(values[0]))
        else:
            with open(save_fn, mode='a') as validate_csv:
                writer = csv.writer(validate_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([step] + [args.num_sample_test] + list(values[0]))

    elif args.data_name == 'KITTIDC':
        if not os.path.isfile(save_fn):
            print('{} created'.format(save_fn))
            with open(save_fn, mode='a') as validate_csv:
                writer = csv.writer(validate_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['epoch'] + ['num_lines'] + contents)
                writer.writerow([step] + [args.lidar_lines_test] + list(values[0]))
        else:
            with open(save_fn, mode='a') as validate_csv:
                writer = csv.writer(validate_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([step] + [args.lidar_lines_test] + list(values[0]))

