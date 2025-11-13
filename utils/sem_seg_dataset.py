import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)

# from .qwen_vl_utils import process_vision_info
from .vision_process import process_vision_info
from VideoLLaMA2av.videollama2.constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, DEFAULT_AUDIO_TOKEN, IGNORE_INDEX
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler


def get_image_info(image_path, min_pixel, max_pixel):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    messages = [
        {"role": "user",
         "content": [
             {
                 "type": "image",
                 "image": image_path,
                 "min_pixel": min_pixel,
                 ### here do some change
                 "max_pixel": min_pixel

             }
         ]
         }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def cv_random_flip(images, masks):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag <0.5:
        images = np.flip(images, axis=1).copy()
        masks = np.flip(masks, axis=1).copy()
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return images, masks

def randomCrop(images, labels, border=30):
    """
    :param images: numpy array, shape (B, H, W, 3)
    :param labels: numpy array, shape (B, H, W)
    :return: cropped_images, cropped_labels
    """
    B, H, W, _ = images.shape

    crop_w = np.random.randint(W - border, W)
    crop_h = np.random.randint(H - border, H)

    x1 = (W - crop_w) // 2
    y1 = (H - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped_images = images[:, y1:y2, x1:x2, :]
    cropped_labels = labels[:, y1:y2, x1:x2]

    return cropped_images, cropped_labels


from PIL import Image
import numpy as np
import random

from PIL import Image
import numpy as np
import random

def randomRotation(image, label):
    mode = Image.BICUBIC
    label_mode = Image.NEAREST  

    if random.random() > 0.8:
        angle = np.random.randint(-15, 15)

        rotated_images = []
        rotated_labels = []

        for i in range(image.shape[0]):
            img_pil = Image.fromarray(image[i])  # (H, W, 3)
            mask_pil = Image.fromarray((label[i] * 255).astype(np.uint8))  # (H, W)

            img_rot = img_pil.rotate(angle, resample=mode)
            mask_rot = mask_pil.rotate(angle, resample=label_mode)

            rotated_images.append(np.array(img_rot))
            rotated_labels.append(np.array(mask_rot) / 255.0)  

        image = np.stack(rotated_images, axis=0)
        label = np.stack(rotated_labels, axis=0)

    return image, label

import numpy as np
import soundfile as sf
import os

def load_audio_from_video(file: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    # Check file extension
    _, ext = os.path.splitext(file)
    ext = ext.lower()

    # If it's a WAV file, use soundfile for direct reading
    if ext == '.wav':
        try:
            audio, original_sr = sf.read(file)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            if original_sr != sr:
                from scipy import signal
                ratio = sr / original_sr
                audio = signal.resample(audio, int(len(audio) * ratio))

            return audio.astype(np.float32), sr
        
        except Exception as e:
            raise RuntimeError(f"Failed to load WAV audio: {str(e)}") from e

    # For non-WAV files, use ffmpeg method
    cmd = ["ffmpeg", "-nostdin", "-i", file, "-vn",  # no video
        "-acodec", "pcm_s16le",  # output audio codec (pcm_s16le for .wav)
        "-ac", "1",  # audio channels (1 for mono)
        "-ar", str(sr),  # audio sample rate
        "-f", "s16le",  # output format (s16le for 16-bit PCM)
        "-"  # output to stdout
        ]
    
    from subprocess import CalledProcessError, run, Popen, PIPE
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0, sr


def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
        #print(int(start))
        #print(int(end))
        #print("AAAA")
    return all_clips_timepoints


def process_audio_from_video(audio_path, clip_duration, device="cpu", num_mel_bins=128, sample_rate=16000, clips_per_video=10, mean=-4.268, std=9.138):
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=1, clips_per_video=clips_per_video
    )
    try:
        waveform, sr = load_audio_from_video(audio_path)
    except Exception as audio_error:
        print(f"Failed to process audio from video due to error: {audio_error}")
        waveform = torch.zeros(480000)
        waveform = waveform.numpy()
        sr = 16000
    all_clips_timepoints = get_clip_timepoints(clip_sampler, waveform.shape[0] / sample_rate)
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        waveform_clip = waveform[
            int(clip_timepoints[0] * sample_rate) : int(
                clip_timepoints[1] * sample_rate)]
        all_clips.append(waveform_clip)
    all_clips_tensors = [torch.from_numpy(clip) for clip in all_clips]
    wav = torch.cat(all_clips_tensors)
    if len(wav) > 30 * sr:
        max_start = len(wav) - 30 * sr
        start = torch.randint(0, max_start, (1,)).item()
        wav = wav[start: start + 30 * sr]
    if len(wav) < 30 * sr:
        pad_length = 30 * sr - len(wav)
        wav = torch.nn.functional.pad(wav, (0, pad_length), mode='constant', value=0.0)
    waveform = wav.unsqueeze(0) * 2 ** 15
    import torchaudio.compliance.kaldi as ta_kaldi
    fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10).to(torch.bfloat16)
    return fbank.unsqueeze(0)


from typing import Dict, Optional, Sequence, List
def preprocess_multimodal(
    sources: Sequence[str],
    modal_token: str = None,
) -> Dict:
    for source in sources:
        for sentence in source:
            if modal_token in sentence['value']:
                sentence['value'] = sentence['value'].replace(modal_token, '').strip()
                sentence['value'] = modal_token + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = modal_token
            # TODO: fix this for multimedia, e.g., <video>, <audio>, etc.
            sentence["value"] = sentence["value"].replace(modal_token, replace_token)

    return sources


import transformers
def preprocess_plain(
    sources: Sequence[str],
    system_prompt,
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    conversations = []
    input_ids = []
    targets = []
    for source in sources:
        # 1. apply chat template for input conversation
        assert len(source) == 2
        assert modal_token in source[0]['value']
        message = [
            {'role': 'user', 'content': source[0]['value']},
            {'role': 'assistant', 'content': source[1]['value']}
        ]
        message = system_prompt + message
        conversation = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        # 2. tokenize conversations
        from VideoLLaMA2av.videollama2.mm_utils import tokenizer_multimodal_token
        input_ids.append(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))
        # 3. make targets
        import copy
        targets.append(copy.deepcopy(input_ids[-1]))
        instruction = tokenizer.apply_chat_template(message[:2], tokenize=False, add_generation_prompt=True)
        instruction_len = len(tokenizer_multimodal_token(instruction, tokenizer, modal_token, return_tensors='pt'))
        targets[-1][:instruction_len] = IGNORE_INDEX
        # print("instruction: ----------------")
        # print(instruction)
        # print("conversation: ----------------")
        # print(conversation)
        # print("training targets: ----------------")
        # print(tokenizer.decode(targets[-1][instruction_len:]))
        # print(input_ids[-1])
        # print(targets[-1])
    return dict(input_ids=input_ids, labels=targets)


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids[0],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels[0],
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal` of LlavaMetaForCausalLM in llava_arch.py
        batch['images'] = []
        for instance in instances:
            for modal_token in MODAL_INDEX_MAP.keys():
                modal_token = modal_token.lower()
                # MODAL_TOKEN shape like: <image>, <video>, ...
                import re
                modal_name = re.findall(f'[<](.*)[>]', modal_token)
                assert len(modal_name) == 1
                modal_name = modal_name[0]
                if modal_name in instance:
                    batch['images'].append((instance[modal_name], modal_name))

        return batch


class SemSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 224
    ignore_label = 255
    image_min_pixel = 401408
    image_max_pixel = 1003520

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        processor,
        vision_tower,
        meta_path,
        samples_per_epoch=14113,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.processor = processor
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")[0]
        import pandas as pd
        df_all = pd.read_csv(meta_path, sep=',')
        df_split = df_all[df_all['split'] == splits]
        import json
        with open("/root/reason_cleaned.json", "r") as f:
            data = json.load(f)

        images = []
        audio = []
        gts = []
        texts = []
        reason = []
        class_all = []
        num = 0

        def matches_relevant_pattern(exp):
            if isinstance(exp, str):
                return bool(re.search(keywords_regex, exp.lower()))
            return False

        for index, row in df_split.iterrows():
            vid = row['vid']
            fid = row['fid']
            exp = row['exp']
            clas = row['uid'].split("_")[-2]
            # if matches_relevant_pattern(exp):
            images_frame = glob.glob(os.path.join(base_image_dir, "media", vid, "frames", "*.jpg"))
            sorted_images = sorted(images_frame, key=lambda x: int(os.path.basename(x).split('.')[0]))
            audio_frame = os.path.join(base_image_dir, "media", vid, "audio.wav")
            gt_frame = glob.glob(os.path.join(base_image_dir, "gt_mask", vid, "fid_{}".format(fid), "*.png"))
            sorted_gt = sorted(gt_frame, key=lambda x: int(os.path.basename(x).split('.')[0]))

            if len(data[num]['reasoning'])<2000:
                images.append(sorted_images)
                audio.append(audio_frame)
                gts.append(sorted_gt)
                texts.append(exp)
                reason.append(data[num]['reasoning'])
                class_all.append(clas)
            else:
                print(data[num]['reasoning'])
            num += 1

        print(len(images))
        print(len(audio))
        print(len(gts))
        print(len(texts))
        self.reason_seg_data = (images, audio, gts, texts, reason, class_all)

        print("number of reason_seg samples: ", len(images))

        explanatory = -1
        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            with open(
                os.path.join(
                    base_image_dir,
                    "reason_seg",
                    reason_seg_data,
                    "explanatory",
                    "train.json",
                )
            ) as f:
                items = json.load(f)
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            print("len(self.img_to_explanation): ", len(self.img_to_explanation))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, audio, gts, texts, reasons, class_all = self.reason_seg_data
        for m in range(len(images)):
            if len(gts[idx]) !=10:
                print(gts[idx][0])
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        audio_path = audio[idx]
        gts_path = gts[idx]
        text = texts[idx]
        reason = reasons[idx]
        class_name = class_all[idx]
        
        preprocess_video = self.processor['image']
        video_data = []
        for i in range(10):
            from PIL import Image
            video_data.append(preprocess_video(image_path[i]))
        video_data = torch.cat(video_data)
        video_duration_seconds = 1
        audio = process_audio_from_video(audio_path, video_duration_seconds)
        video = {'video': video_data, 'audio': audio}
        modal_token = "<video>"
        tensor = [(video, 'video')]
        
        # sources = [
        #     {
        #         "id": 0,
        #         "video": "images/xxx.jpg", 
        #         "conversations": [
        #             {
        #                 "from": "human",
        #                 "value": "<video>\nThe reference is: {} Please segment the corresponding object in the images.".format(text)
        #             },
        #             {
        #                 "from": "gpt",
        #                 "value": "{} It is [SEG].".format(reason)
        #             }
        #         ]
        #     }
        # ]
        # import copy
        # sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), modal_token)
        # system_message = [
        #     {'role': 'system', 'content': (
        #     """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
        #     """\n"""
        #     """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
        #     }
        # ]
        # data_dict = preprocess_plain(sources, system_message, self.tokenizer, modal_token=modal_token)
        # data_dict['video'] = video
        # data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        # llm_input = dict(train_dataset=data_dict,
        #         data_collator=data_collator)

        message1 = [{'role': 'user', 'content': modal_token + '\n' + "The reference is: {} Please segment the corresponding object in the images.".format(text)}]
        message2 = [{'role': 'assistant', 'content': "It is [SEG]."}]
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
        message = system_message + message1
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        from VideoLLaMA2av.videollama2.mm_utils import process_image, process_video, tokenizer_multimodal_token
        input_ids = tokenizer_multimodal_token(prompt, self.tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long()
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long()

        import copy
        targets = copy.deepcopy(input_ids)
        message1 = system_message + message1
        prompt1 = self.tokenizer.apply_chat_template(message1, tokenize=False, add_generation_prompt=True)
        instruction_len = len(tokenizer_multimodal_token(prompt1, self.tokenizer, modal_token, return_tensors='pt'))
        targets[:instruction_len] = IGNORE_INDEX

        images_all = []
        for i in range(10):
            image = cv2.imread(image_path[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            initial_size = image.shape[:2]
            image = self.transform.apply_image(image)
            resize = image.shape[:2]
            image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            images_all.append(image)
        image = torch.stack(images_all, dim=0)

        mask_all = []
        if len(gts_path) == 0:
            for j in range(10):
                sample = cv2.imread(image_path[i], cv2.IMREAD_GRAYSCALE)/255
                sampled_masks = np.zeros_like(sample, dtype=np.float32) 
                sampled_mask = torch.from_numpy(sampled_masks)
                mask_all.append(sampled_mask)
        else:
            for j in range(10):
                mask = cv2.imread(gts_path[j], cv2.IMREAD_GRAYSCALE)/255
                if np.all(mask == 0):
                    sampled_masks = np.zeros_like(mask, dtype=np.float32) 
                else:
                    sampled_masks = (mask == 1).astype(np.float32)  

                sampled_mask = torch.from_numpy(sampled_masks)
                mask_all.append(sampled_mask)
        masks = torch.stack(mask_all, axis=0)

        llm_input = {}
        llm_input["input_ids"] = input_ids
        llm_input["attention_mask"] = attention_masks
        llm_input["images"] = tensor
        llm_input["targets"] = targets
        llm_input["class_name"] = class_name

        return (
            image_path,
            image,
            llm_input,
            masks,
            resize,
            initial_size,
            class_name
        )
