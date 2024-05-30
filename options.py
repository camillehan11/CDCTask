import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    #dataset and model
    parser.add_argument(
        '--dataset',
        type = str,
        default = 'mnist',
        help = 'name of the dataset: mnist, cifar10,FMNIST,svhn'
    )
    parser.add_argument(
        '--model',
        type = str,
        default = 'mnistlenet',
        help='name of model. mnist: logistic, mnistlenet; cifar10: cnn_tutorial, cnn_complex,SimpleDLA,lenet'
    )
    parser.add_argument(
        '--mu',
        type=float,
        default=0.8
        # help='number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--fed',
        type=str,
        default='fedavg'
        # help='number of communication rounds with the cloud server'
    )


    parser.add_argument(
        '--input_channels',
        type = int,
        default = 3,
        help = 'input channels. mnist:1, cifar10 :3'
    )
    parser.add_argument(
        '--Lagrange',
        type = float,
        default = 1,
        help = 'Lagrange'
    )
    parser.add_argument(
        '--output_channels',
        type = int,
        default = 10,
        help = 'output channels'
    )
    #nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type = int,
        default =100,
        help = 'batch size when trained on TCclient'
    )
    parser.add_argument(
        '--num_communication',
        type = int,
        default=50,
        help = 'number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=5,
        help='number of local update (tau_1)'
    )
    parser.add_argument(
        '--num_TGserver_aggregation',
        type = int,
        default=2,
        help = 'number of TGserver aggregation (tau_2)'
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.01,
        help = 'learning rate of the SGD when trained on TCclient'
    )


    parser.add_argument(
        '--alpha',
        type = float,
        default = 0.8,
        # help = 'learning rate of the SGD when trained on TCclient'
    )


    parser.add_argument(
        '--lr_decay',
        type = float,
        default= '0.998',
        help = 'lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type = int,
        default=1,
        help= 'lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type = float,
        default = 0.9,
        help = 'SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type = float,
        default = 5e-4,
        help= 'The weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type = int,
        default = 0,
        help = 'verbose for print progress bar'
    )
    #setting for federeated learning
    parser.add_argument(
        '--iid',
        type = int,
        default = 1,
        help = 'distribution of the data, 1,0, -2(one-class)'
    )
    parser.add_argument(
        '--TGserveriid',
        type=int,
        default=1,
        help='distribution of the data under TGservers, 1 (TGserveriid),0 (TGserverniid) (used only when iid = -2)'
    )
    parser.add_argument(
        '--frac',
        type = float,
        default = 1,
        help = 'fraction of participated TCclients'
    )
    parser.add_argument(
        '--num_TCclients',
        type = int,
        default = 100,
        help = 'number of all available TCclients'
    )
    parser.add_argument(
        '--num_TGservers',
        type = int,
        default=5,
        help= 'number of TGservers'
    )
    parser.add_argument(
        '--nclass',
        type = int,
        default= 2,
        help= 'number of TGservers'
    )
    parser.add_argument(
        '--nsamples',
        type = int,
        default= 2000,
        help= 'number of TGservers'
    )
    parser.add_argument(
        '--rate_unbalance',
        type = int,
        default= 1.0,
        help= 'number of TGservers'
    )
    # num_users_cifar = 400
    # nclass_cifar = 2
    # nsamples_cifar = 20
    # rate_unbalance_cifar = 1.0

    parser.add_argument(
        '--seed',
        type = int,
        default = 11,
        help = 'random seed (defaul: 1)'
    )
    parser.add_argument(
        '--dataset_root',
        type = str,
        default = 'data',
        help = 'dataset root folder'
    )
    parser.add_argument(
        '--show_dis',
        type= int,
        default= 1,
        help='whether to show distribution'
    )
    parser.add_argument(
        '--classes_per_TCclient',
        type=int,
        default = 10,
        help='under artificial non-iid distribution, the classes per TCclient'
    )
    parser.add_argument(
        '--gpu',
        type = int,
        default=0,
        help = 'GPU to be selected, 0, 1, 2, 3'
    )

    parser.add_argument(
        '--mtl_model',
        default=0,
        type = int
    )
    parser.add_argument(
        '--global_model',
        default=1,
        type=int
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int
    )


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
