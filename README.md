# TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models

<div align="center">
  <a href="https://arxiv.org/abs/2511.21145"><img src="https://img.shields.io/badge/arXiv-2511.21145-red"></a>
</div>

Official implementation of **TEAR** (CVPR 2026), an automated red-teaming framework that uncovers safety risks linked to the dynamic temporal sequencing of text-to-video (T2V) models.

## Overview

TEAR discovers adversarial prompts that are **textually innocuous** but exploit **temporal dynamics** to trigger policy-violating video outputs. The framework operates in three stages:

1. **Stage 1 - Initial Generator Training (SFT)**: Fine-tune Llama-3 with LoRA on curated temporal rewriting data using three strategies: Temporal Deconstruction, Sequential Enforcement, and Temporal-Space Synthesis.

2. **Stage 2 - Temporal-aware Online Preference Learning (PPO)**: Optimize the generator via PPO with two reward signals:
   - *Prompt-level reward*: textual safety + temporal style similarity
   - *Temporal consistency reward*: video-text alignment + temporal coherence (using VideoReward)

3. **Stage 3 - Test Case Refinement**: Iteratively refine prompts using Qwen-3-VL in a closed-loop with video generation, evaluation, and safety filtering via Llama Guard.

## Project Structure

```
TEAR/
├── stage1_sft/                # Stage 1: SFT training
│   └── train_sft.py
├── stage2_ppo/                # Stage 2: PPO training
│   ├── train_ppo.py
│   └── inference.py
├── stage3_refine/             # Stage 3: Refinement pipeline
│   └── tear_pipeline.py
├── evaluation/                # Evaluation metrics
│   ├── self_bleu.py           # 1-AvgSelfBLEU diversity
│   ├── embedding_similarity.py # 1-Cossim diversity
│   ├── nsfw_detect.py         # NSFW text detection
│   └── compute_psr.py         # PSR with multiple filters
├── video_generation/          # Commercial API video generation
│   └── api_generator.py
├── utils/
│   └── extract_prompts.py     # Extract prompts from pipeline logs
├── configs/
│   └── ppo_config.yml         # PPO training configuration
├── data/
│   └── README.md              # Data format documentation
├── VideoAlign/                # VideoReward model (submodule)
├── Wan2.2/                    # Wan2.2 video generator (submodule)
└── trlx/                      # trlx RL framework (submodule)
```

## Installation

```bash
git clone --recursive https://github.com/<your-repo>/TEAR.git
cd TEAR
pip install -r requirements.txt
```

### Third-party Dependencies

The following are included as git submodules:
- [VideoAlign (VideoReward)](https://github.com/KwaiVGI/VideoAlign) - Video reward model
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) - Video generation model
- [trlx](https://github.com/CarperAI/trlx) - RL training framework

## Usage

### Stage 1: SFT Training

```bash
python stage1_sft/train_sft.py \
    --base_model <path_to_llama3_8b> \
    --data_path ./data/sft_data.json \
    --output_dir ./checkpoints/sft_checkpoint \
    --num_train_steps 4000 \
    --batch_size 8 \
    --learning_rate 1e-5
```

### Stage 2: PPO Training

```bash
python stage2_ppo/train_ppo.py \
    --trl_config ./configs/ppo_config.yml \
    --save_dir ./checkpoints/ppo_model
```

### Stage 3: TEAR Pipeline (with Refinement)

```bash
python stage3_refine/tear_pipeline.py \
    --input <seed_prompts.txt> \
    --output ./output/videos \
    --save-prompts ./output/prompts.txt \
    --enable-refine \
    --llm-path ./checkpoints/ppo_model \
    --guard-path <path_to_llama_guard_3_8b> \
    --refine-path <path_to_qwen3_vl> \
    --video-ckpt <path_to_wan2.2_ti2v_5b> \
    --guard-gpu cuda:0 \
    --llm-gpu cuda:1 \
    --refine-gpu cuda:2 \
    --video-gpu cuda:3
```

### Evaluation

```bash
# Self-BLEU diversity (1-AvgSelfBLEU)
python evaluation/self_bleu.py --input output/prompts.txt

# Embedding similarity diversity (1-Cossim)
python evaluation/embedding_similarity.py \
    --input output/prompts.txt \
    --api_key <openai_api_key>

# PSR (Prompt Successful Pass rate)
python evaluation/compute_psr.py \
    --input output/prompts.txt \
    --toxicity_model s-nlp/roberta_toxicity_classifier \
    --nsfw_model <path_to_nsfw_detector> \
    --guard_model <path_to_llama_guard>
```

### Commercial T2V Model Testing

```bash
python video_generation/api_generator.py \
    --prompt "your prompt here" \
    --model veo-3.1-generate-preview \
    --api_key <your_api_key> \
    --output_dir ./output
```

## Hardware Requirements

- Minimum 4x NVIDIA GPUs with 24GB+ VRAM each
- Recommended: 4x A6000 (48GB) or equivalent

| Component | GPU | VRAM |
|-----------|-----|------|
| Llama Guard 3 8B | GPU 0 | ~16GB |
| LLM Generator (Llama-3 8B + LoRA) | GPU 1 | ~16GB |
| Refine Model (Qwen-3-VL) | GPU 2 | ~6GB |
| Video Model (Wan2.2-TI2V-5B) | GPU 3 | ~20GB |

## Ethical Statement

This work is intended for **proactive safety auditing** of T2V models before deployment. The goal is to help developers identify and mitigate temporal safety vulnerabilities. We do not release seed prompts or generated adversarial content. All experiments were conducted in controlled research environments.

## Citation

```bibtex
@inproceedings{he2026tear,
  title={TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models},
  author={He, Jiaming and Hou, Guanyu and Li, Hongwei and Huang, Zhicong and Chen, Kangjie and Yu, Yi and Jiang, Wenbo and Xu, Guowen and Zhang, Tianwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
