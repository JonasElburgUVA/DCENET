parser = argparse.ArgumentParser()

#parameters for training
parser.add_argument('--nn', type=str, default='linear')
parser.add_argument('--layers', type=int, nargs='+', default=[160, 160, 160])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--attention', action='store_true', default=False)
parser.add_argument('--bidirectional', action='store_true', default=False)
parser.add_argument('--supervised', action='store_true', default=False)
parser.add_argument('--cpu', action='store_true', default=False)
parser.add_argument('--exp', type=int, default=0)
#parameters for evaluation
parser.add_argument('--var_seq', action='store_true', default=False)
parser.add_argument('--var_hct', action='store_true', default=False)
parser.add_argument('--results', action='store_true', default=False)

args = parser.parse_args()

hp = hyperparams.Hyperparams()

hp.training.lr = args.lr
hp.training.batch_size = args.batch_size
hp.network.nn = args.nn
hp.network.layers = args.layers
hp.network.attention = args.attention
hp.network.bidirectional = args.bidirectional
hp.supervised = args.supervised
if args.cpu:
    hp.device = torch.device('cpu')