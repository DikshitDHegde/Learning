import argparse

def config():
    parser = argparse.ArgumentParser(description='Pre-Training for Contrastive Learning DIKSHIT')
    parser.add_argument(
        '--dataset_path',
        type=str, 
        default="/home/cvgws-06/Desktop/Dikshit/DATASET/CIFAR10", 
        help="Dataset for training the models having Train and Test folder"
    )

    parser.add_argument(
        '--resolution', 
        type=int, 
        default=28, 
        help="Resolution of the square image"
    )
    
    parser.add_argument(
        '--out_dimension', 
        type=int, 
        default=10, 
        help="out_dimension at the end for feature extraction"
    )
    
    parser.add_argument(
        '--channels', 
        type=int, 
        default=3, 
        help="Channels of the square image"
    )

    parser.add_argument(
        '-o',
        '--optimizer',
        type=str,
        default='Adam',
        help='Optimizer to be used Adam or SGC'
    )
    
    parser.add_argument(
        '-r',
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate to be used'
    )
    
    parser.add_argument(
        '-b',
        '--batch', 
        type= int, 
        default = 256, 
        help='batch_size'
    )
    
    parser.add_argument(
        "--out_dir",
        type=str,
        default='Auto_VAE_005',
        help='Output directory path to save models'
    )
    
    parser.add_argument(
        "--epochs",
        type=int, 
        default=200,
        help="Number of epochs to be trained"
    )
    
    parser.add_argument(
        "--seed",
        type = int, 
        default=0, 
        help="Seed for the experiment"
    )
    
    parser.add_argument(
        "--resume",
        type = bool,
        default=False,
        help='To Resume the training'
    )

    parser.add_argument(
        "--n_clusters",
        type = int,
        default = 10,
        help = "No of cluster centriods for kmeans default = 10",
    )

    parser.add_argument(
        "--recentre",
        type = int,
        default = 10,
        help = "recentre updation instervel default = 2",
    )

    parser.add_argument(
        "--A",
        type = float,
        default = 0.5,
        help = "alpha in reconstruction loss default = 0.5"
    )

    parser.add_argument(
        "--B",
        type = float,
        default = 0.5,
        help = "beta in encoder loss default = 0.5"
    )

    parser.add_argument(
        "--checkpoint",
        type = str, 
        default=None, 
        help='Checkpoint path to resume the training'
    )
    
    parser.add_argument(
        "--experiment",
        type = str, 
        default="Batchsize 128 Alpha 0.5 Beta 0.7 lr 0.001 optimizer Adam", 
        help='Experimentation name'
    )
    
    parser.add_argument(
        "--set_device",
        type = int,
        default=1, 
        help='SET GPU DEVICE'
    )

    return  parser.parse_args()
