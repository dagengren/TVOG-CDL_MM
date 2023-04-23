from utils.parser_train import relative_path_to_absolute_path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.pseudo_VidSTG import VidSTGDataset
from models import build_model
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
import utils.misc as utils_misc
from models.postprocessors import build_postprocessors
import os
import random
from functools import partial
from models.da_prob_model import CustomModel

def calc_prototype(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    src_dataset = VidSTGDataset("train", opt)
    # if opt.distributed:
    #     sampler_src = DistributedSampler(src_dataset, shuffle=False)
    # else:
    #     sampler_src = torch.utils.data.SequentialSampler(src_dataset)
    sampler_src = torch.utils.data.SequentialSampler(src_dataset)
    data_loader_src = DataLoader(
        src_dataset,
        opt.batch_size,
        sampler=sampler_src,
        drop_last=False,
        collate_fn=partial(utils_misc.collate_fn, False),
        num_workers=opt.num_workers,
    )

    dst_dataset = VidSTGDataset("test", opt)
    # if opt.distributed:
    #     sampler_dst = DistributedSampler(dst_dataset)
    # else:
    #     sampler_dst = torch.utils.data.SequentialSampler(dst_dataset)
    sampler_dst = torch.utils.data.SequentialSampler(dst_dataset)
    data_loader_dst = DataLoader(
        dst_dataset,
        opt.batch_size,
        sampler=sampler_dst,
        drop_last=False,
        collate_fn=partial(utils_misc.collate_fn, False),
        num_workers=opt.num_workers,
    )

    model, criterion, contrastive_criterion, weight_dict = build_model(opt)
    model.to(device)
    model_without_ddp = model
    # if opt.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module
    if opt.load:
        print("loading from", opt.load)
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "model_ema" in checkpoint:
            model_without_ddp.load_state_dict(checkpoint["model_ema"], strict=False)
        else:
            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

    calc_proto(model, data_loader_dst, data_loader_src, device, opt)

@torch.no_grad()
def calc_proto(
    model: torch.nn.Module,
    data_loader_dst,
    data_loader_src,
    device: torch.device,
    args
):

    class_features = Class_Features(numbers=opt.n_class)

    customModel = CustomModel(args)

    model.eval()
    for epoch in range(1):#args.epochs):
        for dst_data in data_loader_dst:
            # src_data = data_loader_src.next()

            dst_img_tensor = dst_data["samples"].to(device)
            dst_captions = [t["caption"] for t in dst_data["targets"]]

            memory_cache = model(dst_img_tensor, dst_captions, encode_and_save=True)
            dst_outputs, dst_features = model(dst_img_tensor, dst_captions, encode_and_save=False, memory_cache=memory_cache)
            # dst_features [6, 1, 360, 256]
            # dst_outputs['pred_logits'] [1, 360, 2]
            # dst_features [6, 1, 36, 256]
            # dst_outputs['pred_logits_prediction'] [1, 36, 10]

            vectors, ids = class_features.calculate_mean_vector(dst_features[-1, 0, ].cpu(),
                                                 dst_outputs['pred_logits_prediction'][0].cpu())

            for t in range(len(ids)):
                customModel.update_objective_SingleVector(ids[t], vectors[t].numpy(), 'mean')
##################

    save_path = os.path.join(os.path.dirname(opt.output_dir), "prototypes_on_{}_from_{}_clean".format("dstData", "srcModel"))
    torch.save(customModel.objective_vectors, save_path)

    # print(customModel.objective_vectors)
    # print(customModel.objective_vectors.shape)
    #         # print(vectors)
    #         # print()
    #         # print(ids)
    #         # print()
    # exit()

class Class_Features:
    def __init__(self, numbers=10):
        self.class_numbers = numbers
        self.class_feature = [[] for i in range(self.class_numbers)]
        self.num = np.zeros(numbers)

    def calculate_mean_vector(self, feat_cls, prob_list, labels_val=None, model=None):
        vectors = feat_cls
        ids = prob_list.max(-1)[1].tolist()
        # print(ids)
        # exit()
        # proposal_num = len(prob_list)
        # prob_max = -9999
        # prob_max_index = -1
        # for i in range(proposal_num):
        #     if prob_list[i]>prob_max:
        #         prob_max = prob_list[i]
        #         prob_max_index = i
        #     if (i+1) % self.class_numbers == 0:
        #         prob_max = -9999
        #         ids.append(prob_max_index%self.class_numbers)
        #         vectors.append(feat_cls[prob_max_index])
        return vectors, ids

def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    file_path = os.path.join(logdir, 'run.log')
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument('--n_class', type=int, default=10, help='19|16|13')

    # Dataset specific
    parser.add_argument("--dataset_config", default=None, required=True)
    parser.add_argument("--do_qa", action="store_true", help="Whether to do question answering")
    parser.add_argument("--predict_final",action="store_true",help="If true, will predict if a given box is in the actual referred set. Useful for CLEVR-Ref+ only currently.")
    parser.add_argument("--no_detection", action="store_true", help="Whether to train the detector")
    parser.add_argument("--split_qa_heads", action="store_true", help="Whether to use a separate head per question type in vqa")

    # Training hyper-parameters 
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=84, type=int)
    parser.add_argument("--lr_drop", default=35, type=int)
    parser.add_argument("--epoch_chunks",default=-1,type=int,help="If greater than 0, will split the training set into chunks and validate/checkpoint after each chunk")
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument("--eval_skip",default=1,type=int,help='do evaluation every "eval_skip" frames')
    parser.add_argument("--schedule",default="linear_with_warmup",type=str,choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"))
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--fraction_warmup_steps", default=0.01, type=float, help="Fraction of total number of steps")

    # Model parameters
    parser.add_argument("--frozen_weights",type=str,default=None,help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument("--freeze_text_encoder", action="store_true", help="Whether to freeze the weights of the text encoder")
    parser.add_argument("--text_encoder_type", default="/hzzd/zhy/da-stvg/transformer_pretrain")

    # Backbone
    parser.add_argument("--backbone",default="resnet101",type=str,help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns")
    parser.add_argument("--dilation",action="store_true",help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument("--position_embedding",default="sine",type=str,choices=("sine", "learned"),help="Type of positional embedding to use on top of the image features")

    # Transformer
    parser.add_argument("--enc_layers",default=6,type=int,help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers",default=6,type=int,help="Number of decoding layers in the transformer")
    parser.add_argument("--dim_feedforward",default=2048,type=int,help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument("--hidden_dim",default=256,type=int,help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument("--nheads",default=8,type=int,help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=36, type=int,help="Number of frames")
    parser.add_argument("--num_queries", default=360, type=int, help="Number of query slots")
    parser.add_argument("--num_queries_per_frame", default=10, type=int, help="Number of query slots per frame")
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--no_pass_pos_and_query",dest="pass_pos_and_query",action="store_false",help="Disables passing the positional encodings to each attention layers")

    # Segmentation
    parser.add_argument("--mask_model",default="none",type=str,choices=("none", "smallconv", "v2"),help="Segmentation head to be used (if None, segmentation will not be trained)")
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--masks", action="store_true")

    # Loss
    parser.add_argument("--no_aux_loss",dest="aux_loss",action="store_false",help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument("--set_loss",default="hungarian",type=str,choices=("sequential", "hungarian", "lexicographical"),help="Type of matching to perform in the loss")
    parser.add_argument("--contrastive_loss", action="store_true", help="Whether to add contrastive loss")
    parser.add_argument("--no_contrastive_align_loss",dest="contrastive_align_loss",action="store_false",help="Whether to add contrastive alignment loss")
    parser.add_argument("--contrastive_loss_hdim",type=int,help="Projection head output size before computing normalized temperature-scaled cross entropy loss")
    parser.add_argument("--temperature_NCE", type=float, default=0.07, help="Temperature in the  temperature-scaled cross entropy loss")

    # * Matcher
    parser.add_argument("--set_cost_class",default=1,type=float,help="Class coefficient in the matching cost")
    parser.add_argument("--set_cost_bbox",default=5,type=float,help="L1 box coefficient in the matching cost")
    parser.add_argument("--set_cost_giou",default=2,type=float,help="giou box coefficient in the matching cost")
    # Loss coefficients
    parser.add_argument("--ce_loss_coef", default=1, type=float)
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--qa_loss_coef", default=1, type=float)
    parser.add_argument("--l1_loss_coef", default=0.1, type=float)
    parser.add_argument("--eos_coef",default=0.1,type=float,help="Relative classification weight of the no-object class")
    parser.add_argument("--contrastive_loss_coef", default=0.1, type=float)
    parser.add_argument("--contrastive_align_loss_coef", default=1, type=float)

    # Run specific
    parser.add_argument("--test", action="store_true", help="Whether to run evaluation on val or test set")
    parser.add_argument("--test_type", type=str, default="test", choices=("testA", "testB", "test"))
    parser.add_argument("--output-dir", default="class_pro_model", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--load", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--num_workers", default=5, type=int)

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training and evaluation script", parents=[get_args_parser()])
    parser.add_argument('--name', type=str, default='', help='pretrain source model')
    opt = parser.parse_args()
    opt.logdir = os.path.join('logs', opt.name)
    opt.n_class = int(opt.num_queries/opt.num_frames)

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = None#get_logger(opt.logdir)

    calc_prototype(opt, logger)