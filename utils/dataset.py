import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from .vqa_dataset import VQADataset
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
import numpy as np
import soundfile as sf
import os
from VideoLLaMA2av.videollama2.mm_utils import process_image, process_video, tokenizer_multimodal_token


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


def collate_fn(
    batch, tokenizer=None, processor=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_videollama2_list = []
    masks_list = []
    resize_list = []
    initial_size_list = []
    class_name_all = []

    offset_list = [0]
    cnt = 0
    inferences = []
    _, _, _, _, _, _, _, inference1 = batch[0]
    if inference1 == True:
        for (
            image_path,
            images,
            llm_input,
            masks,
            resize,
            initial_size,
            inference,
        ) in batch:
            image_path_list.append(image_path)
            images_list.append(images)
            images_videollama2_list.append(llm_input)
            masks_list.append(masks.float())
            resize_list.append(resize)
            initial_size_list.append(initial_size)
            inferences.append(inference)

        output = images_videollama2_list
        tokenizer = tokenizer
    else:
        # for (
        #     image_path,
        #     images,
        #     llm_input,
        #     masks,
        #     resize,
        #     initial_size,
        #     inference,
        # ) in batch:
        #     image_path_list.append(image_path)
        #     images_list.append(images)
        #     images_videollama2_list.append(llm_input['train_dataset'])
        #     masks_list.append(masks.float())
        #     resize_list.append(resize)
        #     initial_size_list.append(initial_size)
        #     inferences.append(inference)
        for (
            image_path,
            images,
            llm_input,
            masks,
            resize,
            initial_size,
            class_name,
            inference,
        ) in batch:
            image_path_list.append(image_path)
            images_list.append(images)
            images_videollama2_list.append(llm_input)
            masks_list.append(masks.float())
            resize_list.append(resize)
            initial_size_list.append(initial_size)
            inferences.append(inference)
            class_name_all.append(class_name)

        output = images_videollama2_list
        tokenizer = tokenizer

        # data_collator = llm_input['data_collator']
        # output = data_collator(images_videollama2_list)
        # tokenizer = tokenizer


    return {
        "images": torch.cat(images_list, dim=0),
        "output": output,
        "masks_list": masks_list, 
        "resize":resize_list,
        "initial_size": initial_size_list,
        "tokenizer": tokenizer,
        "class_name":class_name,
        "inference": inference,
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        processor,
        vision_tower,
        meta_path,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="refclef||refcoco||refcoco+||refcocog",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "ref-avs":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        processor,
                        vision_tower,
                        meta_path,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 224
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        processor,
        vision_tower,
        meta_path,
        val_dataset,
        image_size=224,
    ):
        
        reason_seg_data, splits = val_dataset.split("|")
        splits = splits.split("_")[0]
        import pandas as pd
        df_all = pd.read_csv(meta_path, sep=',')
        df_split = df_all[df_all['split'] == 'test_s']

        refer_seg_ds = {}

        refer_seg_ds["images"] = []
        refer_seg_ds["audio"] = []
        refer_seg_ds["gts"] = []
        refer_seg_ds["texts"] = []
        refer_seg_ds["classes"] = []

        for index, row in df_split.iterrows():
            vid = row['vid']
            fid = row['fid']
            exp = row['exp']
            clas = row['uid'].split("_")[-2]
            
            images_frame = glob.glob(os.path.join(base_image_dir, "media", vid, "frames", "*.jpg"))
            sorted_images = sorted(images_frame, key=lambda x: int(os.path.basename(x).split('.')[0]))
            audio_frame = os.path.join(base_image_dir, "media", vid, "audio.wav")
            gt_frame = glob.glob(os.path.join(base_image_dir, "gt_mask", vid, "fid_{}".format(fid), "*.png"))
            sorted_gt = sorted(gt_frame, key=lambda x: int(os.path.basename(x).split('.')[0]))

            refer_seg_ds["images"].append(sorted_images)
            refer_seg_ds["audio"].append(audio_frame)
            refer_seg_ds["gts"].append(sorted_gt)
            refer_seg_ds["texts"].append(exp)
            refer_seg_ds["classes"].append(clas)

        self.refer_seg_ds = refer_seg_ds
        self.data_type = "refer_seg"

        self.image_size = image_size
        self.tokenizer = tokenizer
        self.processor = processor
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

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
        refer_seg_ds = self.refer_seg_ds
        images = refer_seg_ds["images"]
        audio = refer_seg_ds["audio"]
        texts = refer_seg_ds["texts"]
        gts = refer_seg_ds["gts"]
        classes = refer_seg_ds["classes"]

        image_path = images[idx]
        audio_path = audio[idx]
        text = texts[idx]
        gts_path = gts[idx]
        class_name = classes[idx]

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
        
        message1 = [{'role': 'user', 'content': modal_token + '\n' + "The reference is: {} Please segment the corresponding object in the images.".format(text)}]
        message2 = [{'role': 'assistant', 'content': "It is {} [SEG].".format(class_name)}]
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
        message = system_message + message1
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
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
        for i in range(10):
            mask = cv2.imread(gts_path[i], cv2.IMREAD_GRAYSCALE)/255
            sampled_masks = (mask == 1).astype(np.float32)
            # sampled_masks = self.transform.apply_image(sampled_masks)
            # h, w = sampled_masks.shape[-2:]
            sampled_mask = torch.from_numpy(sampled_masks)
            # padh = self.img_size - h
            # padw = self.img_size - w
            mask_all.append(sampled_mask)
        masks = torch.stack(mask_all, axis=0)
        inference = True

        llm_input = {}
        llm_input["input_ids"] = input_ids
        llm_input["attention_mask"] = attention_masks
        llm_input["images"] = tensor
        llm_input["targets"] = targets

        return (
            image_path,
            image,
            llm_input,
            masks,
            resize,
            initial_size,
            inference,
        )