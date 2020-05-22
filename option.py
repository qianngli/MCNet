import argparse
# Training settings
parser = argparse.ArgumentParser(description="Hyperspectral Image Super-Resolution")
parser.add_argument("--upscale_factor", default=4, type=int, help="super resolution upscale factor")
parser.add_argument('--seed', type=int, default=1,  help='random seed (default: 1)')
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="maximum number of epochs to train")
parser.add_argument("--show", action="store_true", help="show Tensorboard")

parser.add_argument("--lr", type=int, default=1e-4, help="initial  lerning rate")
parser.add_argument("--cuda", action="store_true", help="Use cuda")
parser.add_argument("--gpus", default="0,1,2,3", type=str, help="gpu ids")
parser.add_argument("--threads", type=int, default=8, help="number of threads for dataloader to use")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")               

parser.add_argument("--datasetName", default="CAVE", type=str, help="data name")

# Network settings
parser.add_argument('--n_conv', type=int, default=1, help='number of  blocks')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--n_bands', type=int, default=31, help='number of bandS')


# Test image
parser.add_argument('--model_name', default='checkpoint/model_4_epoch_200.pth', type=str, help='super resolution model name ')
opt = parser.parse_args() 
