"""
main.py — Entry point for Conditional DDPM training and testing

Usage:
  Train:  python main.py --mode train --img_dir ./iclevr --epochs 300
  Test:   python main.py --mode test --ckpt ./results/best_model.pth
  Both:   python main.py --mode both --img_dir ./iclevr --epochs 300
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Conditional DDPM for iCLEVR')

    # Mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'both'],
                        help='train / test / both')

    # Paths
    parser.add_argument('--img_dir', type=str, default='./iclevr',
                        help='Path to training images directory')
    parser.add_argument('--train_json', type=str, default='./file/train.json',
                        help='Path to train.json')
    parser.add_argument('--test_json', type=str, default='./file/test.json',
                        help='Path to test.json')
    parser.add_argument('--new_test_json', type=str, default='./file/new_test.json',
                        help='Path to new_test.json')
    parser.add_argument('--objects_json', type=str, default='./file/objects.json',
                        help='Path to objects.json')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save checkpoints')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save generated images')
    parser.add_argument('--ckpt', type=str, default='./results/best_model.pth',
                        help='Checkpoint path for testing')
    parser.add_argument('--resume', type=str, default='',
                        help='Checkpoint path to resume training from')

    # Model
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels in UNet')
    parser.add_argument('--time_dim', type=int, default=256,
                        help='Time embedding dimension')
    parser.add_argument('--num_classes', type=int, default=24,
                        help='Number of object classes')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate in ResBlocks')

    # DDPM
    parser.add_argument('--schedule', type=str, default='linear',
                        choices=['linear', 'quad', 'cos'],
                        help='Noise schedule')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps T')
    parser.add_argument('--beta_start', type=float, default=1e-4,
                        help='Beta schedule start')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='Beta schedule end')
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--p_uncond', type=float, default=0.1,
                        help='Probability of dropping condition (for CFG training)')

    # Training
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='AdamW weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                        help='EMA decay rate')
    parser.add_argument('--img_size', type=int, default=64,
                        help='Image resolution')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--eval_every', type=int, default=20,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda / cpu)')

    args = parser.parse_args()

    # Channel multipliers (not easily specified via argparse)
    args.channel_mults = (1, 2, 4, 8)

    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("  Conditional DDPM for iCLEVR")
    print("=" * 60)
    print(f"  Mode:         {args.mode}")
    print(f"  Device:       {args.device}")
    print(f"  Image dir:    {args.img_dir}")
    print(f"  Schedule:     {args.schedule}")
    print(f"  Timesteps:    {args.num_timesteps}")
    print(f"  CFG scale:    {args.cfg_scale}")
    if args.mode in ('train', 'both'):
        print(f"  Epochs:       {args.epochs}")
        print(f"  Batch size:   {args.batch_size}")
        print(f"  LR:           {args.lr}")
    print("=" * 60)

    if args.mode in ('train', 'both'):
        from code.train import Trainer
        trainer = Trainer(args)
        trainer.train()

    if args.mode in ('test', 'both'):
        from code.test import Tester
        tester = Tester(args)
        tester.run()


if __name__ == '__main__':
    main()
