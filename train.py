import os
import sys
import fire
import time
import torch
import torchplus

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path)
sys.path.append(os.path.join(base_path, 'third_party/second.pytorch'))

MAJOR_VERSION = 1
MINOR_VERSION = 0

from tqdm import tqdm
from tensorboardX import SummaryWriter
from configparser import ConfigParser
from optimizer import optimizer_builder
from optimizer import lr_scheduler_builder
from ops.common import get_class
from models.lpcc_net import LPCC_Net
from ops.common import Logger, flatten_deep_dict


def train_single_dataset(train_loader, val_loader, training_config, lrs_config, net, optimizer, lr_scheduler,
                         result_dir, display_step):
    # get configuration
    total_step = int(lrs_config['total_step'])
    eval_step_list = eval(training_config['eval_step_list'])
    save_step_list = eval(training_config['save_step_list'])

    # initialization
    optimizer.zero_grad()
    current_step = 0
    writer = SummaryWriter(logdir=os.path.join(result_dir, 'tensorboardX'))

    # main loop
    while current_step < total_step:
        for example_train in train_loader:
            # start step timer
            t = time.time()
            torch.cuda.synchronize()

            # lr scheduler step
            lr_scheduler.step(current_step)

            # network forward
            loss_dict, loss_info, _ = net(example_train)

            # get loss & backward
            loss = loss_dict['loss']
            loss.backward()

            # clip grad & optimizer step
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()

            # stop step timer
            torch.cuda.synchronize()
            step_time = time.time() - t

            # display and write to tensorboardX
            if current_step % display_step == 0 and current_step > 0:
                print(f"@@@ step: {current_step} @@@ --- loss: {loss}, step_time: {step_time}", flush=True)
                print(loss_info, flush=True)
                flat_dict = flatten_deep_dict(loss_info)
                for key, value in flat_dict.items():
                    writer.add_scalars(key, value, current_step)

            # save checkpoints
            save_flag = (isinstance(save_step_list, list) and current_step in save_step_list) or \
                        (isinstance(save_step_list, int) and current_step % save_step_list == 0 and current_step > 0)
            if save_flag:
                torchplus.train.save_models(result_dir, [net, optimizer], current_step)

            # evaluation
            eval_flag = (isinstance(eval_step_list, list) and current_step in eval_step_list) or \
                        (isinstance(eval_step_list, int) and current_step % eval_step_list == 0 and current_step > 0)
            if eval_flag:
                eval_func = get_class(training_config['eval_fn'])
                net.eval()
                pred_results = []
                print('*****************************************', flush=True)
                print('generating predicted outputs... ', flush=True)
                sys.stdout.flush()
                for example_val in tqdm(val_loader):
                    with torch.no_grad():
                        pred_out, _ = net(example_val)
                    pred_results.append(pred_out)
                sys.stderr.flush()
                print('done.', flush=True)
                print('evaluating... ', flush=True)
                eval_res = eval_func(pred_results, val_loader)
                print('done.', flush=True)
                print(eval_res, flush=True)
                print('*****************************************', flush=True)
                net.train()

            current_step += 1
            if current_step >= total_step:
                torchplus.train.save_models(result_dir, [net, optimizer], current_step - 1)
                break
    writer.close()


def train(dataset_cfg_path,
          model_cfg_path,
          train_cfg_path,
          result_dir,
          display_step=50,
          resume=False):
    """
    main entrance for training

    :param dataset_cfg_path: configuration file for dataset
    :param model_cfg_path: configuration file for model
    :param train_cfg_path: configuration file for training
    :param result_dir: directory for saving models and logs
    :param display_step: display logs every display steps
    :param resume: try resuming training from checkpoints, if specified
    :return: None
    """
    # model dir & resume check
    if os.path.exists(result_dir) and resume:
        raise NotImplementedError('not implemented yet')
    elif os.path.exists(result_dir) and not resume:
        raise Exception('result_dir exists, but resume=False')
    elif not os.path.exists(result_dir) and resume:
        raise Exception('result_dir dose not exist, but resume=True')
    else:
        os.makedirs(result_dir)

    # initialize logger
    logger = Logger(os.path.join(result_dir, 'log.txt'))
    logger.bind()

    # log info
    print(f'LPCC-Net v{MAJOR_VERSION}.{MINOR_VERSION}', flush=True)

    # get configurations
    dataset_cfg = ConfigParser()
    model_cfg = ConfigParser()
    train_cfg = ConfigParser()
    dataset_cfg.read(dataset_cfg_path)
    model_cfg.read(model_cfg_path)
    train_cfg.read(train_cfg_path)

    dataset_train_config = dataset_cfg['DATASET-TRAIN']
    dataset_val_config = dataset_cfg['DATASET-VAL']
    model_config = model_cfg['MODEL']
    optimizer_config = train_cfg['OPTIMIZER']
    lrs_config = train_cfg['LR-SCHEDULER']
    training_config = train_cfg["TRAINING"]

    # log info
    print({section: dict(dataset_cfg[section]) for section in dataset_cfg.sections()}, flush=True)
    print({section: dict(model_cfg[section]) for section in model_cfg.sections()}, flush=True)
    print({section: dict(train_cfg[section]) for section in train_cfg.sections()}, flush=True)

    # dataset
    dataset_train = get_class(dataset_train_config['class'])(dataset_train_config)
    dataset_val = get_class(dataset_val_config['class'])(dataset_val_config)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=eval(dataset_train_config['batch_size']),
        shuffle=True,
        num_workers=eval(dataset_train_config['num_workers']),
        pin_memory=False,
        collate_fn=get_class(dataset_train_config['collate_fn']),
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=eval(dataset_val_config['batch_size']),
        shuffle=False,
        num_workers=eval(dataset_val_config['num_workers']),
        pin_memory=False,
        collate_fn=get_class(dataset_val_config['collate_fn']),
    )

    # prepare network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LPCC_Net(model_config).to(device)

    # optimizer
    optimizer = optimizer_builder.build(optimizer_config, net)
    lr_scheduler = lr_scheduler_builder.build(lrs_config, optimizer)

    # start training
    train_single_dataset(train_loader, val_loader, training_config, lrs_config, net, optimizer, lr_scheduler,
                         result_dir, display_step)

    # release logger
    logger.release()


if __name__ == '__main__':
    fire.Fire()
