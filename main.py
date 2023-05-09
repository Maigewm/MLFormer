import os
import torch
import argparse
import importlib
import pytorch_lightning as pl
import numpy as np
from lit_model.lit_model import TransformerLitModel
from model.modeling_vilt import ViltForMaskedLM, ViltConfig
from ptflops import get_model_complexity_info
from data.data_module import KGC
from torchstat import stat
import ipdb
from pytorch_lightning.core.memory import ModelSummary
#from pytorch_lightning.callbacks import StochasticWeightAveraging #change

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#Attention! kgc may change
#remember to import class from other module


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="KGC")
    parser.add_argument("--chunk", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--task_name", type=str, default='wn18')
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--use_swa", type=bool, default=True)#change
    parser.add_argument(" --check_val_every_n_epoch", type=int, default=10)
    parser.add_argument("--num_workers", type=str, default=32)

    data_group = parser.add_argument_group("Data Args")
    KGC.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    #if hasattr(ViltForMaskedLM, "add_to_argparse"):
    #ViltForMaskedLM.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    TransformerLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    vilt_config = ViltConfig.from_pretrained("dandelin/vilt-b32-mlm")
    model=ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm",config=vilt_config)

    #ipdb.set_trace()
    #change calculate flops and params


    '''
    from thop import profile
    from thop import clever_format
    from transformers import ViltProcessor
    from PIL import Image
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    text = "hello world"
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    inputs = processor(image, text, return_tensors="pt")
    #ipdb.set_trace()
    flops, params = profile(model, inputs)
    flops, params = clever_format([flops, params], "%.3f")
    print("flops:",flops, "params:",params)
    '''
    data=KGC(args, model)
    print('Load data.')
    tokenizer=data.tokenizer
    print('Load model.')

    lit_model=TransformerLitModel(args=args, model=model, tokenizer=tokenizer, data_config=data.get_config())
    #ModelSummary(lit_model,mode='full')
    #ipdb.set_trace()
    #print(ModelSummary(model, mode='full'))
    print('Load litmodel.')
    if args.checkpoint:
        lit_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])

    

    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="kgc", name=args.data_dir.split("/")[-1])
        logger.log_hyperparams(vars(args))

    metric_name = "Eval/hits10"

    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/mrr", mode="max", patience=15, strict=False)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor=metric_name, mode="max",
        filename=args.data_dir.split("/")[-1] + '/{epoch}-{Eval/hits10:.2f}-{Eval/hits1:.2f}' if not args.pretrain else args.data_dir.split("/")[-1] + '/{epoch}-{step}-{Eval/hits10:.2f}',
        dirpath="output",
        save_weights_only=True,
    )
    callbacks = [early_callback, model_checkpoint]
    print('call back')
    #change:swa
    #if args.use_swa:
     #   callbacks.append(pl.callbacks.StochasticWeightAveraging(swa_epoch_start=2))

    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=callbacks,
                                            logger=logger,
                                            default_root_dir="training/logs",
                                            )

    print('trainer')

#find lr
    # lr_finder = trainer.tuner.lr_find(lit_model, datamodule=data)
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig("learning_rate.png",)
    # fig.show()
    # print("lr finder",lr_finder.suggestion())
    # print("lr finder",lr_finder.suggestion())

    #trainer.tune(lit_model, datamodule=data)
    '''macs, params = get_model_complexity_info(lit_model, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))'''

    trainer.fit(lit_model, datamodule=data)
    path = model_checkpoint.best_model_path
    lit_model.load_state_dict(torch.load(path)["state_dict"])

    result = trainer.test(lit_model, datamodule=data)
    print(result)

    # _saved_pretrain(lit_model, tokenizer, path)
    print("*path"*30)
    print(path)

if __name__ == "__main__":

    main()
