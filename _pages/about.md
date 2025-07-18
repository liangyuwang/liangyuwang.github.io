---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
I am Liangyu Wang, a Ph.D. candidate in Computer Science at King Abdullah University of Science and Technology ([KAUST](https://www.kaust.edu.sa/en/)), specializing in efficient training and inference for large language models (LLMs) through distributed computing and advanced GPU programming. 
Before that, I completed my master degree at The Chinese University of Hong Kong, focusing on multimodal machine learning.

Currently, I am conducting LLM pretraining research at the [Alibaba Qwen Team](https://huggingface.co/Qwen).

My research interests include optimizing distributed training and inference of LLMs, improving multi-threaded and multi-stream scheduling, and enhancing privacy-preserving methods for LLMs. I have interned as a LLM Pretraining Engineer at [Aramco](https://www.aramco.com/), working with large-scale GPU clusters to boost training throughput and model scalability. Currently, I am working on:

* Efficient reinforcement learning (RL) for LLMs reasoning
* Distributed training and inference of LLMs
* Efficient algorithm and infrastructure design for LLMs
* Efficient privacy-preserving methods

News
====

* 07/2025: ZO2 is accepted by [COLM 2025](https://colmweb.org/index.html).

* 07/2025: Released Infinite-Sampling ([paper](https://arxiv.org/pdf/2506.22950)).

* 06/2025: Joined [Alibaba Qwen Team](https://huggingface.co/Qwen) for LLM Pretraining.

* 04/2025: Attended the [ICLR 2025](https://openreview.net/group?id=ICLR.cc/2025/Conference), Singapore.

* 03/2025: Released ZO2 ([paper](https://arxiv.org/abs/2503.12668), [code](https://github.com/liangyuwang/zo2)).

Projects
========

**ZO2 (Zeroth-Order Offloading): Full Parameter Fine-Tuning 175B LLMs with 18GB GPU Memory** [![GitHub stars](https://img.shields.io/github/stars/liangyuwang/zo2?style=social)](https://github.com/liangyuwang/zo2)
- A framework that enables fine-tuning of extremely large language models (like OPT-175B) on limited GPU memory through zeroth-order optimization and CPU-GPU offloading.

**Tiny-LLM-Libs: Minimalistic Re-Implementations of Popular LLM Libraries**
A collection of concise re-implementations of popular LLM training libraries, designed to help users understand the core functionalities:
- **Tiny-DeepSpeed** [![GitHub stars](https://img.shields.io/github/stars/liangyuwang/Tiny-DeepSpeed?style=social)](https://github.com/liangyuwang/Tiny-DeepSpeed)  
A minimalistic re-implementation of DeepSpeed's core functionalities for distributed training
- **Tiny-FSDP** [![GitHub stars](https://img.shields.io/github/stars/liangyuwang/Tiny-FSDP?style=social)](https://github.com/liangyuwang/Tiny-FSDP)  
A concise re-implementation of PyTorch FSDP for efficient model parallelism
- **Tiny-Megatron** [![GitHub stars](https://img.shields.io/github/stars/liangyuwang/Tiny-Megatron?style=social)](https://github.com/liangyuwang/Tiny-Megatron)  
A simplified version of NVIDIA's Megatron-LM for model parallelism and pipeline parallelism

Publications
============

* **ZO2: Scalable Zeroth-Order Fine-Tuning for Extremely Large Language Models with Limited GPU Memory**  
  **Liangyu Wang**, Jie Ren, Hang Xu, Junxiao Wang, Huanyi Xie, David E. Keyes, and Di Wang  
  COLM, 2025; NeurIPS workshop, 2024
  [Paper](https://arxiv.org/abs/2503.12668) | [Code](https://github.com/liangyuwang/zo2)

* **Infinite-Sampling: Efficient and Stable Grouped RL Training for Large Language Models**  
  **Liangyu Wang**, Huanyi Xie, Xinhai Wang, Tianjin Huang, Mengdi Li, and Di Wang  
  preprint arXiv:2506.22950, 2025 
  [Paper](https://arxiv.org/pdf/2506.22950)

* **DistZO2: : High-Throughput and Memory-Efficient Zeroth-Order Fine-tuning LLMs with Distributed Parallel Computing**  
  **Liangyu Wang**, Huanyi Xie, and Di Wang
  preprint arXiv:2507.03211, 2025 
  [Paper](https://arxiv.org/pdf/2507.03211) | [Code](https://github.com/liangyuwang/zo2)

* **FlashDP: Memory-Efficient and High-Throughput DP-SGD Training for Large Language Models**  
  **Liangyu Wang**, Junxiao Wang, Jie Ren, Zihang Xiang, David E. Keyes, and Di Wang  
  NeurIPS workshop 2024 
  [Paper](https://openreview.net/pdf?id=6izXTVVzoI) | [Code](https://github.com/kaustpradalab/flashdp)

* **WiP: Towards Light Adaptation of Large Language Models For Personal Hardware**  
  **Liangyu Wang**, Junxiao Wang and Di Wang  
  Mobisys workshop 2024 
  [Paper](https://dl.acm.org/doi/pdf/10.1145/3662006.3662065)

Reviewer Service
================

- Reviewer for [ICLR 2025](https://iclr.cc/)
- Reviewer for [COLM 2025](https://colmweb.org/)
