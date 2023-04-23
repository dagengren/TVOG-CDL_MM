import torch
import math
import sys
from typing import Dict, Iterable, Optional
from utils.metrics import MetricLogger, SmoothedValue
from utils.misc import targets_to
import utils.dist as dist
from utils.optim import adjust_learning_rate, update_ema
import numpy as np
from utils.calculator import calculate_iou3d, calculate_iou1d
import utils.box_ops as box_ops

def train_one_epoch(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    contrastive_criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[torch.nn.Module] = None,
):
    model.train()
    if criterion is not None:
        criterion.train()
    if contrastive_criterion is not None:
        contrastive_criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        curr_step = epoch * len(data_loader) + i
        samples = batch_dict["samples"].to(device)
        targets = batch_dict["targets"]
        captions = [t["caption"] for t in targets]

        targets = targets_to(targets, device)

        memory_cache = None
        if args.masks:
            outputs = model(samples, captions)
        else:
            memory_cache = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, None))

        if contrastive_criterion is not None:
            assert memory_cache is not None
            contrastive_loss = contrastive_criterion(memory_cache["text_pooled_op"], memory_cache["img_pooled_op"])
            loss_dict["contrastive_loss"] = contrastive_loss

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

#prediction（cenX, cenY, width, height)
#target（xMin， yMin， xMax， yMax）
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    contrastive_criterion: Optional[torch.nn.Module],
    postprocessors: Dict[str, torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader,
    device: torch.device,
    args,
):
    model.eval()
    if criterion is not None:
        criterion.eval()
    if contrastive_criterion is not None:
        contrastive_criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    all_tIoU = list()
    all_qtIoU = list()
    all_ctIoU = list()
    all_vIoU = list()
    all_qvIoU = list()
    all_cvIoU = list()
    all_vIoU2 = list()
    all_qvIoU2 = list()
    all_cvIoU2 = list()

    for batch_dict in data_loader:#metric_logger.log_every(data_loader, 10, header):
        samples = batch_dict["samples"].to(device)
        positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
        ori_targets = batch_dict["targets"]
        answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
        captions = [t["caption"] for t in ori_targets]
        targets = targets_to(ori_targets, device)

        # outputs = model(samples, captions)


        memory_cache = None
        if args.masks:
            outputs = model(samples, captions)
        else:
            memory_cache = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

        # outputs = model(samples, captions)
        #
        # loss_dict = {}
        # if criterion is not None:
        #     loss_dict.update(criterion(outputs, targets, positive_map))
        #
        # if contrastive_criterion is not None:
        #     assert memory_cache is not None
        #     contrastive_loss = contrastive_criterion(memory_cache["text_pooled_op"], memory_cache["img_pooled_op"])
        #     loss_dict["contrastive_loss"] = contrastive_loss
        #
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = dist.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        # metric_logger.update(
        #     loss=sum(loss_dict_reduced_scaled.values()),
        #     **loss_dict_reduced_scaled,
        #     **loss_dict_reduced_unscaled,
        # )

        if not args.no_detection:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors["bbox"](outputs, orig_target_sizes)

            if "segm" in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)

            # res = {target["video_id"].item(): output for target, output in zip(targets, results)}
            pred_region, traj_gts, pred_start, pred_end, temporal_gts = \
                generate_pred(args, ori_targets, results, orig_target_sizes, targets[0]["boxes"].device)

            # print("pred_region: ")
            # print(results)
            # print(pred_region)
            # print(traj_gts)
            # exit()

            #pred eval
            # training_progress_bar.update(regions.size(0))
            pred_start = pred_start.detach()
            pred_end = pred_end.detach()
            true_start = temporal_gts[:, 0].float()
            true_end = temporal_gts[:, 1].float()

            tIoU_result = calculate_iou1d(pred_start, pred_end, true_start, true_end)

            traj_gts, types = get_batch_tensor(targets)

            qtIoU_result = tIoU_result[types == 1]
            ctIoU_result = tIoU_result[types == 0]

            # print(tIoU_result,qtIoU_result,ctIoU_result)
            # print(len(tIoU_result),len(qtIoU_result),len(ctIoU_result))
            # print(sum(tIoU_result),sum(qtIoU_result),sum(ctIoU_result))
            all_tIoU.append(np.mean(tIoU_result))
            all_qtIoU.append(np.mean(qtIoU_result))
            all_ctIoU.append(np.mean(ctIoU_result))

            # types = torch.tensor(types).cuda(self.device)
            pred_start = pred_start.long()
            pred_end = pred_end.long()
            vIoU_result, vIoU_result2 = calculate_iou3d(pred_region, traj_gts, pred_start, pred_end, temporal_gts)
            # vIoU_result = 0
            qvIoU_result = vIoU_result[types == 1]
            cvIoU_result = vIoU_result[types == 0]
            all_vIoU.append(np.mean(vIoU_result))
            all_qvIoU.append(np.mean(qvIoU_result))
            all_cvIoU.append(np.mean(cvIoU_result))

            qvIoU_result2 = vIoU_result2[types == 1]
            cvIoU_result2 = vIoU_result2[types == 0]
            all_vIoU2.append(np.mean(vIoU_result2))
            all_qvIoU2.append(np.mean(qvIoU_result2))
            all_cvIoU2.append(np.mean(cvIoU_result2))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    stats.update(generate_stats([np.mean(all_tIoU), np.mean(all_ctIoU), np.mean(all_tIoU), \
            np.mean(all_qvIoU), np.mean(all_cvIoU), np.mean(all_vIoU), \
            np.mean(all_qvIoU2), np.mean(all_cvIoU2), np.mean(all_vIoU2)] ))

    return stats

def get_batch_tensor(targets):
    traj_gts = None
    types = None
    device = targets[0]['boxes'].device
    for target in targets:
        if traj_gts is None:
            traj_gts = target['boxes'].unsqueeze(0)
            types = target['type'].unsqueeze(0)
        else:
            traj_gts = torch.cat((traj_gts, target['boxes'].unsqueeze(0)), 0)
            types = torch.cat((types, target['type'].unsqueeze(0)), 0)
    return traj_gts, types.cpu().numpy()

def generate_stats(result_list):
    stats = {}
    stats["all_tIoU"] = result_list[0].tolist()
    stats["all_ctIoU"] = result_list[1].tolist()
    stats["all_tIoU"] = result_list[2].tolist()
    stats["all_qvIoU"] = result_list[3].tolist()
    stats["all_cvIoU"] = result_list[4].tolist()
    stats["all_vIoU"] = result_list[5].tolist()
    stats["all_qvIoU2"] = result_list[6].tolist()
    stats["all_cvIoU2"] = result_list[7].tolist()
    stats["all_vIoU2"] = result_list[8].tolist()
    return stats

def process_target(target_sizes, out_bbox):
    # convert to [x0, y0, x1, y1] format
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    # print("img_h, img_w: ")
    # print(scale_fct)

    return boxes

def generate_pred(args, targets, results, orig_target_sizes, device):
    pred_region = []
    traj_gts = []
    pred_start = []
    pred_tail = []
    temporal_gt = []

    for target, result in zip(targets, results):
        scores = result['scores'].tolist()
        labels = result['labels'].tolist()
        boxes = result['boxes'].tolist()
        object = target['object']
        target_boxes = target['boxes'].squeeze(1).tolist()
        temporal_gt.append(target['temporal_gt'].tolist())
        traj_gts.append(target_boxes)
        num_queries_per_frame = args.num_queries_per_frame

        frame_min = 9999
        frame_max = -1
        frame_count = -1
        score_max = -9999.
        index = -1
        pred_region_per_batch = []
        for i in range(0, len(scores)):
            if labels[i] == 1:
                if scores[i]>score_max:
                    score_max = scores[i]
                    index = i
            if (i+1)%num_queries_per_frame == 0:
                if index!=-1:
                    pred_region_per_batch.append(boxes[index])
                else:
                    pred_region_per_batch.append([-1, -1, -1, -1])
                frame_count += 1
                if index != -1:
                    frame_min = min(frame_min, index)
                    frame_max = max(frame_max, index)

                score_max = -9999.
                index = -1

        pred_region.append(pred_region_per_batch)
        pred_start.append(frame_min)
        pred_tail.append(frame_max)

    traj_gts = torch.from_numpy(np.array(traj_gts)).to(device)

    return torch.from_numpy(np.array(pred_region)).to(device), \
    process_target(orig_target_sizes, traj_gts), \
    torch.from_numpy(np.array(pred_start)).to(device), \
    torch.from_numpy(np.array(pred_tail)).to(device), \
    torch.from_numpy(np.array(temporal_gt)).to(device)


