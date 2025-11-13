import argparse
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8/'
os.environ['HF_DATASETS_CACHE']=""
os.environ['HF_HOME']=""
os.environ['HUGGINGFACE_HUB_CACHE']=""
os.environ['TRANSFORMERS_CACHE']=""


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA_grpo import LISAForCausalLM
from VideoLLaMA2av.videollama2 import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

from VideoLLaMA2av.videollama2 import model_init, mm_infer
from VideoLLaMA2av.videollama2.utils import disable_torch_init
from VideoLLaMA2av.videollama2.mm_utils import process_image, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria, process_audio_file
import deepspeed


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="./VideoLLaMA2.1-7B-AV/"
    )
    # parser.add_argumen
    #     "--version", default="Qwen/Qwen2.5-VL-7B-Instruct"
    # )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="./clip-vit-large-patch14/", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # parser.add_argument(
    #     "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    # )
    parser.add_argument(
        "--dataset", default="ref-avs", type=str
    )
    # parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument("--sample_rates", default="1", type=str)
    parser.add_argument("--sem_seg_data", default="ref-avs|train", type=str)
    parser.add_argument("--val_dataset", default="ref-avs|val", type=str)
    parser.add_argument("--dataset_dir", default="./REFAVS/", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--steps_per_epoch", default=20, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=4,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=1, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained",
                        default="./sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llama2",
        type=str,
    )
    parser.add_argument("--meta_path", default="./REFAVS/metadata.csv", type=str,)
    parser = deepspeed.add_config_arguments(parser) 
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None
    
    tokenizer = model_init(args.version)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    if args.conv_type == "qwen2_vl":
        tokenizer.pad_token_id = 151643
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    vision_tower = model.model.vision_tower
    audio_tower = model.model.audio_tower
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    # NOTE: videollama2 adopts the same processor for processing image and video.
    processor = vision_tower.image_processor

    num_frames = 10 
    processor = {
        'image': partial(process_image, processor=processor, aspect_ratio=None),
        'video': partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames),
        'audio': process_audio_file,
    }

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if not args.eval_only:
        model.model.initialize_lisa_modules(model.config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in audio_tower.parameters():
        p.requires_grad = False
    # for p in model.get_model().mm_projector.parameters():
    #     p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "vision_tower",
                                # "audio_tower",
                                "text_hidden_fcs",
                                # "mask_decoder"
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))


    # distill load
    # model_ref = LISAForCausalLM_ref.from_pretrained(
    #     args.version, torch_dtype=torch_dtype, **model_args
    # )
    # model_ref.config.eos_token_id = tokenizer.eos_token_id
    # model_ref.config.bos_token_id = tokenizer.bos_token_id
    # if args.conv_type == "qwen2_vl":
    #     tokenizer.pad_token_id = 151643
    # model_ref.config.pad_token_id = tokenizer.pad_token_id
    # vision_tower = model_ref.model.vision_tower
    # audio_tower = model_ref.model.audio_tower
    # if not vision_tower.is_loaded:
    #     vision_tower.load_model()

    # model_ref = get_peft_model(model_ref, lora_config)
    #     model_ref.print_trainable_parameters()

    # model_ref.resize_token_embeddings(len(tokenizer))
    # checkpoint = torch.load("", 
    #                             map_location="cpu")  # 先加载到CPU
    # if "module" in checkpoint:
    #     state_dict = checkpoint["module"]
    # else:
    #     state_dict = checkpoint 
    # missing_keys, unexpected_keys = model_ref.load_state_dict(state_dict, strict=False)

    # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # del checkpoint
    # del state_dict

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        processor,
        args.vision_tower,
        args.meta_path,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        explanatory=args.explanatory,
    )

    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            processor,
            args.vision_tower,
            args.meta_path,
            args.val_dataset,
            args.image_size,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 30,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            # "offload_optimizer": {
            #    "device": "cpu"
            # },
            "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
            "ratio": 1.0
        },
            # "prefetch_bucket_size": 0,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=trainable_params,
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            processor=processor,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )
    # model = model.to("cuda")

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join("/root/", "ckpt_model_grpo_reward")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = 3
        # args.start_epoch = (
        #     int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        # )
        model_engine.global_steps = 0
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )


    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    # if args.eval_only:
    # giou, ciou = validate(val_loader, model_engine, 0, writer, args)
    # exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print(epoch)
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        # if args.no_eval == False:
        #     giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
        #     is_best = giou > best_score
        #     best_score = max(giou, best_score)
            #cur_ciou = ciou if is_best else cur_ciou

        #if args.no_eval or is_best:
        save_dir = os.path.join("/root/ckpt_model_grpo_reward/")
        if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
        # if os.path.exists(save_dir):
        #             shutil.rmtree(save_dir)
        torch.distributed.barrier()
        model_engine.save_checkpoint(save_dir, tag="global_step")


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.7f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    reward_alls = AverageMeter("Reward_all", ":.4f")
    reward_ious = AverageMeter("Reward_iou", ":.4f")
    reward_formats = AverageMeter("Reward_format", ":.4f")
    reward_classs = AverageMeter("Reward_class", ":.4f")
    length_all = AverageMeter("Response_length", ":.4f")
    std_all = AverageMeter("Response_length", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
            reward_alls,
            reward_ious,
            reward_formats,
            reward_classs,
            length_all,
            std_all,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        args.grad_accumulation_steps =4
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()

            output_dict = model(**input_dict)
            # distill
            # output = model_ref(**input_dict)

            # pred_stu = output_dict['pred_embeddings']
            # pred_tea = output['pred_embeddings']
            # import torch.nn as nn
            # mse_loss = nn.MSELoss()
            # distill_loss = mse_loss(pred_stu, pred_tea.detach().clone())

            loss = output_dict["loss"]
            lr_loss = output_dict["lr_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            reward_all = output_dict["reward_all"]
            reward_iou = output_dict["reward_iou"]
            reward_format = output_dict["reward_format"]
            reward_class = output_dict["reward_class"]
            length = output_dict["response_length"]
            std = output_dict["std"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(lr_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            reward_formats.update(reward_format, input_dict["images"].size(0))
            reward_alls.update(reward_all, input_dict["images"].size(0))
            reward_ious.update(reward_iou, input_dict["images"].size(0))
            reward_classs.update(reward_class, input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            length_all.update(length, input_dict["images"].size(0))
            std_all.update(std.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                reward_formats.all_reduce()
                reward_alls.all_reduce()
                reward_ious.all_reduce()
                reward_classs.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()
                length_all.all_reduce()
                std_all.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/reward_format", reward_formats.avg, global_step
                )
                writer.add_scalar(
                    "metrics/reward_all", reward_alls.avg, global_step
                )
                writer.add_scalar(
                    "metrics/reward_iou", reward_ious.avg, global_step
                )
                writer.add_scalar(
                    "metrics/reward_class", reward_classs.avg, global_step
                )
                writer.add_scalar(
                    "metrics/length", length_all.avg, global_step
                )
                writer.add_scalar(
                    "metrics/std", std_all.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()
            reward_formats.reset()
            reward_alls.reset()
            reward_ious.reset()
            reward_classs.reset()
            length_all.reset()
            std_all.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):
    from utility import mask_iou1, Eval_Fmeasure1, AverageMeter, MetricLogger, save_single_mask
    miou = AverageMeter("miou", ":6.3f", Summary.SUM)
    f_score = AverageMeter("f_score", ":6.3f", Summary.SUM)

    model_engine.eval()
    num = 0

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()
        num += 1
        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            # input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        miou1 = mask_iou1(pred_masks[0], masks_list)
        f_score1 = Eval_Fmeasure1(pred_masks[0], masks_list.float())

        miou.add({'miou': miou1})
        f_score.add({'f_score': f_score1})

    miou_mean = (miou.pop('miou'))
    f_score_mean = (f_score.pop('f_score'))


    if args.local_rank == 0:
        writer.add_scalar("val/giou", miou_mean, epoch)
        writer.add_scalar("val/ciou", f_score_mean, epoch)
        print("miou: {:.4f}, f_score: {:.4f}".format(miou_mean, f_score_mean))

    return miou_mean, f_score_mean


if __name__ == "__main__":
    main(sys.argv[1:])
