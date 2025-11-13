from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from VideoLLaMA2av.videollama2.model.videollama2_qwen2 import (
    Videollama2Qwen2ForCausalLM, 
    Videollama2Qwen2Model
)

# from transformers import Qwen2_5_VLForConditionalGeneration
# from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
#                                                      LlavaLlamaModel)
from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, Videollama2Qwen2Model):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        # self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

class LISAForCausalLM(Videollama2Qwen2ForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            # config.mm_vision_tower = kwargs.get(
            #     "vision_tower", "openai/clip-vit-large-patch14"
            # )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super().__init__(config)

        self.model = LisaModel(config, **kwargs)
        # self.av_model = self.model.av_model
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images,
        output,
        masks_list,
        resize,
        initial_size,
        tokenizer,
        class_name,
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images).view(-1, 10, 256, 14, 14) #10,256,64,64
        batch_size = image_embeddings.shape[0]
        

        if inference:
            input_ids = output[0]['input_ids'].cuda()
            attention_masks = output[0]['attention_mask'].cuda()
            images = output[0]['images']
            labels1 = output[0]['targets'].cuda()
            # class_name = output[0]['class_name']

            keywords = [tokenizer.eos_token]
            from VideoLLaMA2av.videollama2.mm_utils import KeywordsStoppingCriteria
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, output[0]['input_ids'].cuda())

            do_sample = kwargs.get('do_sample', False)
            temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
            top_p = kwargs.get('top_p', 0.9)
            max_new_tokens = kwargs.get('max_new_tokens', 2048)
            
            with torch.inference_mode():
                output_ids = super(LISAForCausalLM, self).generate(
                    input_ids,
                    images=images,
                    attention_mask=attention_masks,
                    do_sample=False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.eos_token_id,
                )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                print(outputs)

                if "[SEG]" not in outputs:
                    fallback_text = "It is [SEG]."
                    seg_ids = tokenizer(fallback_text, return_tensors="pt").input_ids.cuda()
                    output_ids = torch.cat((output_ids, seg_ids), dim=1)
                full_input_ids = torch.cat([input_ids, output_ids], dim=1)
                full_attention_mask = torch.ones_like(full_input_ids)
                if attention_masks is not None:
                    full_attention_mask[:, :attention_masks.size(1)] = attention_masks
                past_key_values = None
                labels = torch.cat((labels1, output_ids), dim=-1)
                (input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_token_mask) = self.prepare_inputs_labels_for_multimodal(
                    full_input_ids,
                    full_attention_mask,
                    past_key_values,
                    labels,
                    images,
                    self.seg_token_idx
                )
                output = super(LISAForCausalLM, self).forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        use_cache=None,
                        output_attentions=None,
                        output_hidden_states=True,
                        return_dict=True,
                        cache_position=None,
                        )
                
                output_hidden_states = output.hidden_states


        else:
            input_ids = output['input_ids'].cuda()
            labels = output['labels'].cuda()
            attention_masks = output['attention_mask'].cuda()
            images = output['images']
            past_key_values=None

            (input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_token_mask) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    attention_masks,
                    past_key_values,
                    labels,
                    images,
                    self.seg_token_idx
                )
        
            output = super(LISAForCausalLM, self).forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=True,
                    return_dict=True,
                    cache_position=None,
                    )
            output_hidden_states = output.hidden_states 

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        if inference:
            seg_token_mask = (output_ids == self.seg_token_idx)
            pred_embeddings = last_hidden_state[:,-output_ids.shape[1]:,:][seg_token_mask]
        else:
            seg_token_mask = seg_token_mask.unsqueeze(0)
            pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)

        multimask_output = False
        pred_masks = []
        if len(pred_embeddings) > 1:
            print(pred_embeddings.shape)
            pred_embeddings = pred_embeddings[-1].unsqueeze(0)
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(0).unsqueeze(0),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i],
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize[i],
                original_size=initial_size[i],
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            for batch_idx in range(len(pred_masks)):
                gt_mask = gt_masks[batch_idx]
                pred_mask = pred_masks[batch_idx]

                pred_mask1 = torch.sigmoid(pred_mask)  
                pred_mask1 = (pred_mask1 * 255).byte()

                for i in range(pred_mask1.shape[0]):
                    from PIL import Image
                    mask_image = Image.fromarray(pred_mask1[i].cpu().numpy())  
                    mask_image.save(f"./mask_{i}.png") 

            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            pred_mask1 = torch.sigmoid(pred_mask) 
            pred_mask1 = (pred_mask1 * 255).byte()

            for i in range(pred_mask1.shape[0]):
                from PIL import Image
                mask_image = Image.fromarray(pred_mask1[i].cpu().numpy())
                mask_image.save(f"./mask_{i}.png") 

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "pred_embeddings": pred_embeddings,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
