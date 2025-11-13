<div align="center">

# AURORA: Augmented Understanding via Structured Reasoning and Reinforcement Learning for Reference Audio-Visual Segmentation

**Approach**: [[arxiv Paper]](https://arxiv.org/pdf/2508.02149)

</div>

## Overview
Reference Audio-Visual Segmentation (Ref-AVS) tasks challenge models to precisely locate sounding objects by integrating visual, auditory, and textual cues. Existing methods
often lack genuine semantic understanding, tending to memorize fixed reasoning patterns. Furthermore, jointly training for reasoning and segmentation can compromise pixel-level
precision. To address these issues, we introduce AURORA, a novel framework designed to enhance genuine reasoning and language comprehension in reference audio-visual segmentation. We employ a structured Chain-of-Thought (CoT) prompting mechanism to guide the model through a step-by-step reasoning process and introduce a novel segmentation
feature distillation loss to effectively integrate these reasoning abilities without sacrificing segmentation performance. To further cultivate the model’s genuine reasoning capabilities, we devise a further two-stage training strategy: first, a “corrective reflective-style training” stage utilizes self-correction to enhance the quality of reasoning paths, followed by reinforcement learning via Group Reward Policy Optimization (GRPO) to bolster robustness in challenging scenarios. Experiments demonstrate that AURORA achieves state-of-the-art performance on Ref-AVS benchmarks and generalizes effectively to unreferenced segmentation.

<img src="https://github.com/Sssssuperior/AURORA/blob/main/model3_00.png">

## Environmental Setups
```
pip install -r requirements.txt
```

## Start Training
For training, please run the following scripts and change the visible device according to yourself.
```
deepspeed --num_gpus=2 train_ds.py
```
For testing, 
```
deepspeed --num_gpus=2 train_ds_test.py
```

## Citation
If you use AURORA in your research or wish to refer to the baseline results, please use the following BibTeX entry.
If you have any questions, please contact me: ziyangluo1110@gmail.com
```
@article{luo2025aurora,
  title={AURORA: Augmented Understanding via Structured Reasoning and Reinforcement Learning for Reference Audio-Visual Segmentation},
  author={Luo, Ziyang and Liu, Nian and Khan, Fahad Shahbaz and Han, Junwei},
  journal={arXiv preprint arXiv:2508.02149},
  year={2025}
}
```

