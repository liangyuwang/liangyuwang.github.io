---
title: "Infinite-Sampling: Efficient and Stable Grouped RL Training for Large Language Models"
collection: publications
category: conferences
permalink: /publication/2025-06-28-infinite-sampling
date: 2025-06-28
venue: 'arXiv preprint'
paperurl: 'https://arxiv.org/pdf/2506.22950'
authors: 'Liangyu Wang, Huanyi Xie, Xinhai Wang, Tianjin Huang, Mengdi Li, and Di Wang'
abstract: 'An efficient and stable grouped reinforcement learning approach for training large language models.'
---

## Infinite-Sampling: Efficient and Stable Grouped RL Training for Large Language Models

[**Paper**](https://arxiv.org/pdf/2506.22950)

<details>
<summary><strong>Abstract</strong></summary>
Group-based reinforcement learning algorithms such as Group Reward Policy Optimization (GRPO) have proven effective for fine-tuning large language models (LLMs) with human feedback. However, generating and storing multiple responses per prompt incurs substantial memory overhead, especially as the sample group size increases, limiting scalability under constrained hardware. We propose Infinite Sampling, a framework that enables efficient and stable GRPO training by decoupling group size from GPU memory usage. It consists of: (1) micro sampling groups that decompose large groups into memory-feasible rounds; (2) continuous sampling that interleaves generation across groups to improve utilization; and (3) a length-aware scheduler combining token-conditioned sequence length prediction with a two-stage plan: global grouping via FPTAS and runtime refill via SJF. Experiments show that our Micro Sampling Groups reduce peak memory usage by over 50% compared to full-group decoding (e.g., from 21.55 GB to 10.64 GB on Qwen3-1.7B). Building on this, Infinite Sampling improves throughput by over 25% compared to the naive micro sampling group method, reducing decoding steps while maintaining full-length completions and memory usage. Our hybrid scheduling ensures efficient and stable GRPO training with larger groups under realistic GPU memory constraints.
</details>

**Authors:** Liangyu Wang, Huanyi Xie, Xinhai Wang, Tianjin Huang, Mengdi Li, and Di Wang

**Published in:** arXiv preprint 