---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<div class="intro-text">
I am Liangyu Wang, a Ph.D. candidate in Computer Science at King Abdullah University of Science and Technology (<a href="https://www.kaust.edu.sa/en/">KAUST</a>), specializing in efficient training and inference for large language models (LLMs) through distributed computing and advanced GPU programming. 
Before that, I completed my master degree at The Chinese University of Hong Kong, focusing on multimodal machine learning.

Currently, I am conducting LLM pretraining research at the <a href="https://huggingface.co/Qwen">Alibaba Qwen Team</a>.

My research interests include optimizing distributed training and inference of LLMs, improving multi-threaded and multi-stream scheduling, and enhancing privacy-preserving methods for LLMs. I have interned as a LLM Pretraining Engineer at <a href="https://www.aramco.com/">Aramco</a>, working with large-scale GPU clusters to boost training throughput and model scalability. Currently, I am working on:

<ul>
<li>Efficient reinforcement learning (RL) for LLMs reasoning</li>
<li>Distributed training and inference of LLMs</li>
<li>Efficient algorithm and infrastructure design for LLMs</li>
<li>Efficient privacy-preserving methods</li>
</ul>
</div>

<div class="about-section news-section">
<h2 class="section-title">🔥 News</h2>
<ul>
<li>02/2026: Released Canzona (<a href="https://arxiv.org/pdf/2602.06079">Arxiv</a>).</li>
<li>09/2025: FlashDP is accepted by <a href="https://neurips.cc/">NeurIPS 2025</a>.</li>
<li>07/2025: ZO2 is accepted by <a href="https://colmweb.org/index.html">COLM 2025</a>.</li>
<li>07/2025: Released Infinite-Sampling (<a href="https://arxiv.org/pdf/2506.22950">Arxiv</a>).</li>
<li>06/2025: Joined <a href="https://huggingface.co/Qwen">Alibaba Qwen Team</a> for LLM Pretraining.</li>
<li>03/2025: Released ZO2 (<a href="https://arxiv.org/abs/2503.12668">Arxiv</a>, <a href="https://github.com/liangyuwang/zo2">code</a>).</li>
</ul>
</div>

<div class="about-section projects-section">
<h2 class="section-title">☃️ Projects</h2>

<div class="project-item">
<div class="project-title"><strong>ZO2 (Zeroth-Order Offloading): Full Parameter Fine-Tuning 175B LLMs with 18GB GPU Memory</strong> <span class="github-stars"><a href="https://github.com/liangyuwang/zo2"><img src="https://img.shields.io/github/stars/liangyuwang/zo2?style=social" alt="GitHub stars"></a></span></div>
<div class="project-description">A framework that enables fine-tuning of extremely large language models (like OPT-175B) on limited GPU memory through zeroth-order optimization and CPU-GPU offloading.</div>
</div>

<div class="project-item">
<div class="project-title"><strong>Tiny-LLM-Libs: Minimalistic Re-Implementations of Popular LLM Libraries</strong></div>
<div class="project-description">Minimal replicas that map major distributed-training stacks to tiny, readable implementations.</div>

<div class="tiny-compare-wrap">
  <table class="tiny-compare-table">
    <thead>
      <tr>
        <th>Original Stack</th>
        <th>Tiny Replica</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><a href="https://github.com/pytorch/pytorch">PyTorch FSDP</a></td>
        <td>
          <a href="https://github.com/liangyuwang/Tiny-FSDP">Tiny-FSDP</a>
          <span class="github-stars"><a href="https://github.com/liangyuwang/Tiny-FSDP"><img src="https://img.shields.io/github/stars/liangyuwang/Tiny-FSDP?style=social" alt="GitHub stars"></a></span>
        </td>
        <td>Side-by-side implementations of DDP, ZeRO-3, and FSDP, including their distinct communication patterns and memory trade-offs.</td>
      </tr>
      <tr>
        <td><a href="https://github.com/microsoft/DeepSpeed">DeepSpeed</a></td>
        <td>
          <a href="https://github.com/liangyuwang/Tiny-DeepSpeed">Tiny-DeepSpeed</a>
          <span class="github-stars"><a href="https://github.com/liangyuwang/Tiny-DeepSpeed"><img src="https://img.shields.io/github/stars/liangyuwang/Tiny-DeepSpeed?style=social" alt="GitHub stars"></a></span>
        </td>
        <td>A minimal DeepSpeed-style training stack with DDP + ZeRO-1/2/3, plus meta initialization, rank mapping, and compute-communication overlap.</td>
      </tr>
      <tr>
        <td><a href="https://github.com/NVIDIA/Megatron-LM">Megatron-LM</a></td>
        <td>
          <a href="https://github.com/liangyuwang/Tiny-Megatron">Tiny-Megatron</a>
          <span class="github-stars"><a href="https://github.com/liangyuwang/Tiny-Megatron"><img src="https://img.shields.io/github/stars/liangyuwang/Tiny-Megatron?style=social" alt="GitHub stars"></a></span>
        </td>
        <td>Educational TP, DP, and 2D TP+DP hybrid training with custom modules and a runtime auto-tuner for kernel selection.</td>
      </tr>
    </tbody>
  </table>
</div>
</div>

<div class="project-item">
<div class="project-title"><strong>Train Large Model from Scratch: A Minimal Pre-Training Stack for GPT-Style Language Models</strong> <span class="github-stars"><a href="https://github.com/liangyuwang/train-large-model-from-scratch"><img src="https://img.shields.io/github/stars/liangyuwang/train-large-model-from-scratch?style=social" alt="GitHub stars"></a></span></div>
<div class="project-description">A hackable and developer-friendly framework featuring modular GPT architecture with FA/GQA/MoE support, distributed training with ZeRO-1, mixed precision training, and comprehensive profiling utilities for efficient large model pre-training.</div>
</div>
</div>

<div class="about-section publications-section">
<h2 class="section-title">📝 Publications</h2>

<div class="publication-item">
<strong>Canzona: A Unified, Asynchronous, and Load-Balanced Framework for Distributed Matrix-based Optimizers</strong><br>
<strong>Liangyu Wang</strong>, Siqi Zhang, Junjie Wang, Yiming Dong, Bo Zheng, Zihan Qiu, Shengkun Tang, Di Wang, Rui Men, and Dayiheng Liu<br>
arXiv:2602.06079, 2026<br>
<a href="https://arxiv.org/pdf/2602.06079">Paper</a>
</div>

<div class="publication-item">
<strong>ZO2: Scalable Zeroth-Order Fine-Tuning for Extremely Large Language Models with Limited GPU Memory</strong><br>
<strong>Liangyu Wang</strong>, Jie Ren, Hang Xu, Junxiao Wang, Huanyi Xie, David E. Keyes, and Di Wang<br>
COLM, 2025<br>
<a href="https://arxiv.org/pdf/2503.12668">Paper</a> | <a href="https://github.com/liangyuwang/zo2">Code</a>
</div>

<div class="publication-item">
<strong>FlashDP: Memory-Efficient and High-Throughput DP-SGD Training for Large Language Models</strong><br>
<strong>Liangyu Wang</strong>, Junxiao Wang, Jie Ren, Zihang Xiang, David E. Keyes, and Di Wang<br>
NeurIPS 2025<br>
<a href="https://arxiv.org/pdf/2507.01154">Paper</a> | <a href="https://github.com/kaustpradalab/flashdp">Code</a>
</div>

<div class="publication-item">
<strong>Infinite-Sampling: Efficient and Stable Grouped RL Training for Large Language Models</strong><br>
<strong>Liangyu Wang</strong>, Huanyi Xie, Xinhai Wang, Tianjin Huang, Mengdi Li, and Di Wang<br>
arXiv:2506.22950, 2025<br>
<a href="https://arxiv.org/pdf/2506.22950">Paper</a>
</div>

<div class="publication-item">
<strong>DistZO2: High-Throughput and Memory-Efficient Zeroth-Order Fine-tuning LLMs with Distributed Parallel Computing</strong><br>
<strong>Liangyu Wang</strong>, Huanyi Xie, and Di Wang<br>
arXiv:2507.03211, 2025<br>
<a href="https://arxiv.org/pdf/2507.03211">Paper</a> | <a href="https://github.com/liangyuwang/zo2">Code</a>
</div>

<div class="publication-item">
<strong>WiP: Towards Light Adaptation of Large Language Models For Personal Hardware</strong><br>
<strong>Liangyu Wang</strong>, Junxiao Wang and Di Wang<br>
Mobisys workshop 2024<br>
<a href="https://dl.acm.org/doi/pdf/10.1145/3662006.3662065">Paper</a>
</div>
</div>

<div class="about-section reviewer-section">
<h2 class="section-title">📋 Academic Service</h2>

<ul>
<li>Conference <a href="https://colmweb.org/">COLM 2026</a>, <a href="https://eccv.ecva.net/">ECCV 2026</a>, <a href="https://icml.cc/">ICML 2026</a><, <a href="https://2026.aclweb.org/">ACL 2026</a>, <a href="https://cvpr.thecvf.com/">CVPR 2026</a>, <a href="https://aaai.org/conference/aaai/aaai-26/">AAAI 2026</a>, <a href="https://iclr.cc/">ICLR 2025</a>, <a href="https://colmweb.org/">COLM 2025</a></li>
</ul>
</div>
