import argparse
import os
import torch
# from lorahub import pretrains  # Module not available in public repo
from method.pre_dfmeta_ft import pre_dfmeta_ft
from preGenerate import pre_generate
from tool import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lora_recycle')
    #basic
    parser.add_argument('--multigpu', type=str, default='0', help='seen gpu')
    parser.add_argument('--gpu', type=int, default=0, help="gpu")
    parser.add_argument('--dataset', type=str, default='flower', help='cifarfs/miniimagenet/cub/flower/mix')
    parser.add_argument('--testdataset', type=str, default='flower', help='cifarfs/miniimagenet/cub/flower/cropdiseases/eurosat/isic/chest')
    parser.add_argument('--val_interval',type=int, default=2000)
    parser.add_argument('--backbone', type=str, default='base_clip_16')
    parser.add_argument('--resolution',type=int, default=224)
    parser.add_argument('--method', type=str, default='pre_dfmeta_ft')
    #meta
    parser.add_argument('--episode_batch',type=int, default=1)
    parser.add_argument('--way_train', type=int, default=5, help='way')
    parser.add_argument('--num_sup_train', type=int, default=5)
    parser.add_argument('--num_qur_train', type=int, default=15)
    parser.add_argument('--way_test', type=int, default=5, help='way')
    parser.add_argument('--num_sup_test', type=int, default=5)
    parser.add_argument('--num_qur_test', type=int, default=15)
    parser.add_argument('--episode_train', type=int, default=60000)
    parser.add_argument('--episode_test', type=int, default=600)
    parser.add_argument('--outer_lr', type=float, default=0.001)
    parser.add_argument('--use_mask', action='store_true',default=False)
    #pre-ft
    parser.add_argument("--rank", "--r", type=int, default=4)
    #data free
    parser.add_argument('--synthesizer', type=str, default='inversion')
    parser.add_argument('--prune_layer', nargs='+', type=int, default=[-1])
    parser.add_argument('--prune_ratio', nargs='+', type=float, default=[0.0])
    parser.add_argument('--mask_ratio', type=float, default=-1, help="-1: automatically mask the inverted data based on the positions of remaining tokens. e.g., 0.5: mask extra 50% remaining tokens after the last layer.")
    #else
    parser.add_argument('--extra', type=str, default='')
    #pretrain
    parser.add_argument('--preGenerate', action='store_true',default=False,help='pre-generate data')
    parser.add_argument('--lorahub', action='store_true',default=False,help='pre-tune loras')
    parser.add_argument('--lora_num', type=int, default=100)
    parser.add_argument('--pre_datapool_path', type=str, default=None,help='PATH_TO_PREDATAPOOL; it will not use pre_inverted data if this is set as None.')
    #quick test
    parser.add_argument('--quick_test', action='store_true', default=False, help='Quick test mode: generate only 2 images per class for 5 classes')
    parser.add_argument('--test_num_classes', type=int, default=5, help='Number of classes for quick test')
    parser.add_argument('--test_images_per_class', type=int, default=2, help='Images per class for quick test')
    # Removed MapReduce options

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.multigpu
    setup_seed(42)
    args.device=torch.device('cuda:{}'.format(args.gpu))
    ########
    if args.preGenerate:
        pre_generate(args)
    elif args.lorahub:
        print("ERROR: --lorahub option requires 'lorahub' module which is not included in the public repository.")
        print("Please skip this step and proceed with direct training or preGenerate.")
        print("\nSuggested commands:")
        print("  1. Generate synthetic data: python main.py --preGenerate --dataset flower --pre_datapool_path ./pre_datapool/flower")
        print("  2. Direct training: python main.py --dataset flower --testdataset flower")
        exit(1)
    else:
        method_dict = dict(
            pre_dfmeta_ft=pre_dfmeta_ft,
        )
        method=method_dict[args.method](args)
        method.train_loop()