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

import transformers
from transformers import GenerationConfig
import re


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
                The predictions for each example.F
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
        # if not hasattr(config, "train_mask_decoder"):
        config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
        # config.mm_vision_tower = kwargs.get(
        #     "vision_tower", "openai/clip-vit-large-patch14"
        # )
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        # else:
        #     config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.num_generations = 3

        self.beta = 0.04 # followed to simple GRPO
        # Initialize weights and apply final processing
        self.post_init()

    def format_reward(self, answer, **kwargs):
        # pattern = r"^<think>(?:(?!</?think>)[\s\S]*?)</think>\s*<answer>(?:(?!</?answer>)[\s\S]*?)</answer><\|im_end\|>$"
        answer = answer.replace("<|im_end|>", "")
        if answer.endswith("It is [SEG]."):
            return 1.0
        else:
            return 0.0
        
    def class_reward(self, answer, class_name, **kwargs):
        reward = 0
        sentences = [s.strip() for s in re.split(r'[,.!?]+', answer) if s.strip()]
        if len(sentences) < 2:
            return 0.0
        second_last_sentence = sentences[-2]
    
        pattern = r'the target is (?:the )?(\w+)'
        match = re.match(pattern, second_last_sentence, re.IGNORECASE)
        if match:
            reward += 1.0
            raw_category = match.group(1).strip().lower()
            if raw_category == class_name:
                reward += 1.0
                return reward
            else:
                return reward
        else:
            return 0.0
    
    def iou_reward(self, pred, target, eps=1e-7, size_average=True):
            r"""
                param: 
                    pred: size [N x H x W]
                    target: size [N x H x W]
                output:
                    iou: size [1] (size_average=True) or [N] (size_average=False)
            """
            assert len(pred.shape) == 3 and pred.shape == target.shape

            N = pred.size(0)
            num_pixels = pred.size(-1) * pred.size(-2)
            no_obj_flag = (target.sum(2).sum(1) == 0)

            temp_pred = torch.sigmoid(pred)
            pred = (temp_pred > 0.5).int()
            # pred = (pred > 0.5).int()
            inter = (pred * target).sum(2).sum(1)
            union = torch.max(pred, target).sum(2).sum(1)

            inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
            inter[no_obj_flag] = inter_no_obj[no_obj_flag]
            union[no_obj_flag] = num_pixels

            if size_average:
                iou = torch.sum(inter / (union+eps)) / N
                return iou
            else:
                iou = inter / (union+eps)
                return iou

    def correct_reward(self, answer, category_name):
        match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL | re.IGNORECASE)  
        if match:  
            answer_content = match.group(1) 
            return 1 if category_name in answer_content.lower() else -1
        else:
            return -1 
        
    def _get_per_token_logps(self, input_ids1, attention_mask, labels, images, generation):
        past_key_values=None
        for i in range(generation-1):
            images.append(images[0])
        (input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_token_mask) = self.prepare_inputs_labels_for_multimodal(
                    input_ids1,
                    attention_mask,
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
        logits = output.logits  # (B, L, V)
        output_hidden_states = output.hidden_states
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids1 = input_ids1[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids1):
            log_probs = logits_row.log_softmax(dim=-1)
            valid_mask = input_ids_row >= 0
            valid_ids = input_ids_row.clone()
            valid_ids[~valid_mask] = 0
            token_log_prob = torch.gather(log_probs, dim=1, index=valid_ids.unsqueeze(1)).squeeze(1)
            token_log_prob[~valid_mask] = 0.0
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps), output_hidden_states, attention_mask, inputs_embeds, labels
        
    def _get_ref_per_token_logps(self, input_ids1, attention_mask, inputs_embeds, labels, hidden_state):
        past_key_values=None
        # (input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_token_mask) = self.prepare_inputs_labels_for_multimodal(
        #             input_ids1,
        #             attention_mask,
        #             past_key_values,
        #             labels,
        #             images,
        #             self.seg_token_idx
        #         )
        output = self.model.ref_model.forward(
                    input_ids=None,
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
        logits = output.logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids1 = input_ids1[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids1):
            log_probs = logits_row.log_softmax(dim=-1)
            valid_mask = input_ids_row >= 0
            valid_ids = input_ids_row.clone()
            valid_ids[~valid_mask] = 0
            token_log_prob = torch.gather(log_probs, dim=1, index=valid_ids.unsqueeze(1)).squeeze(1)
            token_log_prob[~valid_mask] = 0.0
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

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
        image_embeddings = self.get_visual_embs(images).view(-1, 10, 256, 14, 14)

        if inference:
            input_ids = output[0]['input_ids'].cuda()
            attention_masks = output[0]['attention_mask'].cuda()
            images = output[0]['images']
            labels1 = output[0]['targets'].cuda()

            keywords = [tokenizer.eos_token]
            from VideoLLaMA2av.videollama2.mm_utils import KeywordsStoppingCriteria
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, output[0]['input_ids'].cuda())

            do_sample = kwargs.get('do_sample', False)
            temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
            top_p = kwargs.get('top_p', 0.9)
            max_new_tokens = kwargs.get('max_new_tokens', 2048)
            
            with torch.inference_mode():
                # past_key_values=None
                # (input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_token_mask) = self.prepare_inputs_labels_for_multimodal(
                #     input_ids,
                #     attention_masks,
                #     past_key_values,
                #     labels,
                #     images,
                #     self.seg_token_idx
                # )
        
                # output = super(LISAForCausalLM, self).forward(
                #         input_ids=input_ids,
                #         attention_mask=attention_mask,
                #         past_key_values=past_key_values,
                #         inputs_embeds=inputs_embeds,
                #         labels=labels,
                #         use_cache=None,
                #         output_attentions=None,
                #         output_hidden_states=True,
                #         return_dict=True,
                #         cache_position=None,
                #         )
                # output_hidden_states = output.hidden_states
            
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
            input_ids = output[0]['input_ids'].cuda()
            attention_masks = output[0]['attention_mask'].cuda()
            images = output[0]['images']
            labels1 = output[0]['targets'].cuda()
            class_name = output[0]['class_name']

            keywords = [tokenizer.eos_token]
            from VideoLLaMA2av.videollama2.mm_utils import KeywordsStoppingCriteria
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, output[0]['input_ids'].cuda())

            do_sample = kwargs.get('do_sample', True)
            temperature = kwargs.get('temperature', 1.0 if do_sample else 0.0)
            top_p = kwargs.get('top_p', 0.9)
            max_new_tokens = kwargs.get('max_new_tokens', 150)

            with torch.inference_mode():
                prompt_completion_ids1 = super(LISAForCausalLM, self).generate(
                    input_ids,
                    images=images,
                    attention_mask=attention_masks,
                    do_sample=True,
                    temperature=temperature, 
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.eos_token_id, 
                    num_return_sequences=self.num_generations,
                )

            prompt_completion_ids = torch.cat((input_ids.repeat(self.num_generations, 1), prompt_completion_ids1), dim=1)
            prompt_length = len(input_ids[0])
            completion_ids = prompt_completion_ids1
            length = len(prompt_completion_ids1[0])
            # answers = [self.processor.decode(x).replace('<|endoftext|>', '') for x in completion_ids]
            answers = [tokenizer.batch_decode([x], skip_special_tokens=True)[0].strip() for x in completion_ids]

            is_eos = completion_ids == tokenizer.eos_token_id
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_attention_mask = torch.cat([attention_masks.repeat_interleave(self.num_generations, dim=0), completion_mask], dim=1)  
            labels = prompt_completion_ids.clone()
            labels[:,:prompt_length] = -100

            per_token_logps, output_hidden_states, attention_mask, inputs_embeds, labels = self._get_per_token_logps(
                prompt_completion_ids, 
                prompt_attention_mask, 
                labels, 
                images,
                self.num_generations,
            )
            per_token_logps = per_token_logps[:, prompt_length - 1 :]
            seg_token_mask = labels[:, 1:] == self.seg_token_idx

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        def reset_extra_true_tokens(token_mask):
            no_seg_indices = []
            for i in range(token_mask.shape[0]):  
                true_indices = torch.where(token_mask[i])[0]  
                if len(true_indices) > 1:  
                    token_mask[i] = False  
                    token_mask[i, true_indices[-1]] = True
                if len(true_indices) == 0: 
                    no_seg_indices.append(i) 
            return token_mask, no_seg_indices

        seg_token_mask, no_seg_indices = reset_extra_true_tokens(seg_token_mask)
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        last_hidden_state = last_hidden_state[:, :-1, :] # [5, 1305, 256]

        pred_embeddings = last_hidden_state[seg_token_mask]
        for idx in no_seg_indices:
            if idx < pred_embeddings.shape[0]: 
                pred_embeddings = torch.cat((pred_embeddings[:idx],
                                             last_hidden_state[:, -1, :][idx].unsqueeze(0),
                                             pred_embeddings[idx:]), dim=0)
            else:
                pred_embeddings = torch.cat((pred_embeddings, last_hidden_state[:, -1, :][idx].unsqueeze(0)), dim=0)

        # if pred_embeddings.shape[0] > 1:
        #     pred_embeddings = pred_embeddings[-1, :].unsqueeze(0)

        multimask_output = False
        pred_masks_list = []

        for i in range(len(pred_embeddings)):
            pred_embedding = pred_embeddings[i].unsqueeze(0)
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embedding.unsqueeze(0),
            )

            sparse_embeddings = sparse_embeddings.to(pred_embeddings.dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[0],
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize[0],
                original_size=initial_size[0],
            )
            pred_masks_list.append(pred_mask[:, 0])

        gt_masks = masks_list

        rewards = []
        rewards_iou = []
        rewards_format = []
        rewards_class = []
        for i in range(self.num_generations):
            format_reward = self.format_reward(answers[i])
            class_reward = self.class_reward(answers[i], class_name)
            pred_masks = pred_masks_list[i].unsqueeze(0)
            iou_reward = 0
            for batch_idx in range(len(pred_masks)):
                gt_mask = gt_masks[batch_idx]
                pred_mask = pred_masks[batch_idx]
                iou_reward += self.iou_reward(pred_mask, gt_mask)
            rewards_iou.append(iou_reward)
            rewards_format.append(format_reward)
            rewards_class.append(class_reward)
            rewards.append(format_reward + iou_reward + class_reward)
        all_reward = sum(rewards) / len(rewards)
        all_reward_format = sum(rewards_format) / len(rewards_format)
        all_reward_class = sum(rewards_class) / len(rewards_class)
        all_reward_iou = sum(rewards_iou) / len(rewards_iou)
        rewards = torch.tensor(rewards, device=per_token_logps.device)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss)
        # self.beta * per_token_kl
        loss_r1 = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "answer_ids": answer_output_ids,
            }

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        for i in range(len(pred_masks_list)):
            pred_masks = pred_masks_list[i].unsqueeze(0)
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

        loss = loss_r1 + mask_loss

        return {
            "loss": loss,
            "lr_loss": loss_r1,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            # "cosine_loss": loss_cosine,
            "reward_all": all_reward,
            "reward_iou": all_reward_iou,
            "reward_format": all_reward_format,
            "reward_class": all_reward_class,
            "response_length": length,
            "std":std_grouped_rewards[0],

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
