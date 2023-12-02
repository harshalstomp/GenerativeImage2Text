import argparse

import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from generativeimage2text.p_data import CHPDataModule
from generativeimage2text.pl_model import TestImageCaptioningModel 
import warnings
import torch

from generativeimage2text.inference import test_git_inference_single_tsv
warnings.filterwarnings("ignore")

import os
os.environ['AZFUSE_TSV_USE_FUSE'] = '1'

def main(args):
    seed_everything(args.seed)

    model = TestImageCaptioningModel(
        args.model_name,
        args.tokenizer_name,
        learning_rate=args.learning_rate
    )

    data_module = CHPDataModule(
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        batch_size_test=args.batch_size_test,
        dataloader_num_workers=args.dataloader_num_workers
    )

    loss_checkpoint_callback = ModelCheckpoint(
        monitor='train_loss_epoch',
        filename='{epoch}-{val_loss_epoch:.6f}',
        save_top_k=args.early_stop_patience,
        mode='min',
    )
    early_stop_callback = EarlyStopping(
        monitor='train_loss_epoch',
        patience=args.early_stop_patience,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[
            early_stop_callback,
            loss_checkpoint_callback,
            lr_monitor
        ],
        default_root_dir="./checkpoints/",
        accelerator="gpu", devices=1,  precision=16,  max_epochs=10
    )

    if args.do_train:
        trainer.fit(model, data_module)
    
    
    
    torch.save(model.state_dict(), "output/test_model/snapshot/model.pt")
    kwargs = {"image_tsv": 'data/coco_caption/test.img.tsv', 
              "model_name": "NEW_MODEL",
              "question_tsv": None,
              "out_tsv": "inference/NEW_MODEL/coco.tsv",
              "model": model 
             }
    
    test_git_inference_single_tsv( **kwargs)
    
    
#     if args.do_test:
#         ckpt_path = 'best' if args.do_train else None
#         trainer.test(model, data_module, ckpt_path=ckpt_path)

#         if args.save_test_results is not None:
#             assert model.test_results is not None
#             results = pd.DataFrame(model.test_results)
#             results.to_csv(args.save_test_results, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--skip_train', dest='do_train',
                        action='store_false', default=True)
    parser.add_argument('--skip_test', dest='do_test',
                        action='store_false', default=True)
    parser.add_argument('--save_test_results', type=str, default=False)
    parser = Trainer.add_argparse_args(parser)
    parser = TestImageCaptioningModel.add_argparse_args(parser)
    parser = CHPDataModule.add_argparse_args(parser)
    main(parser.parse_args())
