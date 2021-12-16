from dataclasses import dataclass
from typing import List, Tuple
import os
import argparse


import torch
import numpy as np


@dataclass
class BaseConfig:
    seed: int = 0
    batch_size: int = 200
    device: str = 'cpu'
    p: int = 3
    q: int = 3
    hidden_dims: Tuple[int] = 3 * (50,)
    total_steps: int = 1000
    mc_samples: int = 1000

@dataclass
class SignatureConfig:
    augmentations: Tuple
    depth: int
    basepoint: bool = False

@dataclass
class SigCWGANConfig:
    mc_size: int
    sig_config_future: SignatureConfig
    sig_config_past: SignatureConfig



def main(args):  # algo_id, base_config, base_dir, dataset, spec, data_params={}):

    experiment_directory = os.path.join(base_dir, dataset, spec,)
    if not os.path.exists(experiment_directory):
        # if the experiment directory does not exist we create the directory
        os.makedirs(experiment_directory)
    # Set seed for exact reproducibility of the experiments
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # initialise dataset and algo
    x_real = get_data(dataset, base_config.p, base_config.q, **data_params)
    x_real = x_real.to(base_config.device)
    ind_train = int(x_real.shape[0] * 0.8)
    x_real_train, x_real_test = x_real[:ind_train], x_real[ind_train:] #train_test_split(x_real, train_size = 0.8)

    algo = get_algo(algo_id, base_config, dataset, data_params, x_real_train)
    # Train the algorithm
    algo.fit()
    # create summary
    create_summary(dataset, base_config.device, algo.G, base_config.p, base_config.q, x_real_test)
    savefig('summary.png', experiment_directory)
    x_fake = create_summary(dataset, base_config.device, algo.G, base_config.p, 8000, x_real_test, one=True)
    savefig('summary_long.png', experiment_directory)
    plt.plot(x_fake.cpu().numpy()[0, :2000])
    savefig('long_path.png', experiment_directory)
    # Pickle generator weights, real path and hyperparameters.
    pickle_it(x_real, pt.join(pt.dirname(experiment_directory), 'x_real.torch'))
    pickle_it(x_real_test, pt.join(pt.dirname(experiment_directory), 'x_real_test.torch'))
    pickle_it(x_real_train, pt.join(pt.dirname(experiment_directory), 'x_real_train.torch'))
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cpu').state_dict(), pt.join(experiment_directory, 'G_weights.torch'))
    # Log some results at the end of training
    algo.plot_losses()
    savefig('losses.png', experiment_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument("-gpu", default=0)

    # env
    parser.add_argument("-seed", default=0)

    # model
    parser.add_argument("-batch-size", default=32)
    parser.add_argument("-buffer-size", default=1024)
    parser.add_argument("-num-updates", default=8)
    parser.add_argument("-gamma", default=0.99)
    parser.add_argument("-lambd", default=0.95)
    parser.add_argument("-lr", default=0.001)

    # train
    parser.add_argument("-num-steps", default=10000)
    parser.add_argument("-eval-steps", default=100)
    parser.add_argument("-save-steps", default=2000)

    # performance
    parser.add_argument("-window", default=20)


    args = parser.parse_args()
    main(args)
