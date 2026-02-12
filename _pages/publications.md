---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

You can also find my articles on [my Google Scholar profile](https://scholar.google.com/citations?user=mGkM_WgAAAAJ)

---

## 2026

### <span class="venue-tag venue-preprint">Preprint</span> Canzona: A Unified, Asynchronous, and Load-Balanced Framework for Distributed Matrix-based Optimizers  
**Liangyu Wang**, Siqi Zhang, Junjie Wang, Yiming Dong, Bo Zheng, Zihan Qiu, Shengkun Tang, Di Wang, Rui Men, and Dayiheng Liu  
[[paper](https://arxiv.org/pdf/2602.06079)]

<details>
<summary><strong>Abstract</strong></summary>
The scaling of Large Language Models (LLMs) drives interest in matrix-based optimizers (e.g., Shampoo, Muon, SOAP) for their convergence efficiency; yet their requirement for holistic updates conflicts with the tensor fragmentation in distributed frameworks like Megatron. Existing solutions are suboptimal: synchronous approaches suffer from computational redundancy, while layer-wise partitioning fails to reconcile this conflict without violating the geometric constraints of efficient communication primitives. To bridge this gap, we propose Canzona, a Unified, Asynchronous, and Load-Balanced framework that decouples logical optimizer assignment from physical parameter distribution. For Data Parallelism, we introduce an alpha-Balanced Static Partitioning strategy that respects atomicity while neutralizing the load imbalance. For Tensor Parallelism, we design an Asynchronous Compute pipeline utilizing Micro-Group Scheduling to batch fragmented updates and hide reconstruction overhead. Extensive evaluations on the Qwen3 model family (up to 32B parameters) on 256 GPUs demonstrate that our approach preserves the efficiency of established parallel architectures, achieving a 1.57x speedup in end-to-end iteration time and reducing optimizer step latency by 5.8x compared to the baseline.
</details>

### <span class="venue-tag venue-conf">ICLR 2026</span> Predicting LLM Output Length via Entropy-Guided Representations  
Huanyi Xie, Yubin Chen, **Liangyu Wang**, Lijie Hu, and Di Wang  
[[paper](https://openreview.net/pdf?id=3loQDtveWI)]

<details>
<summary><strong>Abstract</strong></summary>
The long-tailed distribution of sequence lengths in LLM serving and reinforcement learning (RL) sampling causes significant computational waste due to excessive padding in batched inference. Existing methods rely on auxiliary models for static length prediction, but they incur high overhead, generalize poorly, and fail in stochastic "one-to-many" sampling scenarios. We introduce a lightweight framework that reuses the main model's internal hidden states for efficient length prediction. Our framework features two core components: (1) Entropy-Guided Token Pooling (EGTP), which uses on-the-fly activations and token entropy for highly accurate static prediction with negligible cost, and (2) Progressive Length Prediction (PLP), which dynamically estimates the remaining length at each decoding step to handle stochastic generation. To validate our approach, we build and release ForeLen, a comprehensive benchmark with long-sequence, Chain-of-Thought, and RL data. On ForeLen, EGTP achieves state-of-the-art accuracy, reducing MAE by 29.16% over the best baseline. Integrating our methods with a length-aware scheduler yields significant end-to-end throughput gains. Our work provides a new technical and evaluation baseline for efficient LLM inference.
</details>

---

## 2025

### <span class="venue-tag venue-preprint">Preprint</span> Attributing Data for Sharpness-Aware Minimization  
Chenyang Ren, Yifan Jia, Huanyi Xie, Zhaobin Xu, Tianxing Wei, **Liangyu Wang**, Lijie Hu, and Di Wang  
[[paper](https://arxiv.org/pdf/2507.04059)]

<details>
<summary><strong>Abstract</strong></summary>
Sharpness-aware Minimization (SAM) improves generalization in large-scale model training by linking loss landscape geometry to generalization. However, challenges such as mislabeled noisy data and privacy concerns have emerged as significant issues. Data attribution, which identifies the contributions of specific training samples, offers a promising solution. However, directly rendering existing data influence evaluation tools such as influence functions (IF) to SAM will be inapplicable or inaccurate as SAM utilizes an inner loop to find model perturbations that maximize loss, which the outer loop then minimizes, resulting in a doubled computational structure. Additionally, this bilevel structure complicates the modeling of data influence on the parameters. In this paper, based on the IF, we develop two innovative data valuation methods for SAM, each offering unique benefits in different scenarios: the Hessian-based IF and the Gradient Trajectory-based IF. The first one provides a comprehensive estimation of data influence using a closed-form measure that relies only on the trained model weights. In contrast, the other IF for SAM utilizes gradient trajectory information during training for more accurate and efficient data assessment. Extensive experiments demonstrate their effectiveness in data evaluation and parameter tuning, with applications in identifying mislabeled data, model editing, and enhancing interpretability.
</details>

### <span class="venue-tag venue-preprint">Preprint</span> DistZO2: High-Throughput and Memory-Efficient Zeroth-Order Fine-tuning LLMs with Distributed Parallel Computing  
**Liangyu Wang**, Huanyi Xie, and Di Wang  
[[paper](https://arxiv.org/pdf/2507.03211)] [[code](https://github.com/liangyuwang/zo2)]

<details>
<summary><strong>Abstract</strong></summary>
Fine-tuning large language models (LLMs) remains resource-intensive due to their sheer scale. While zeroth-order (ZO) optimization provides a memory-efficient alternative by eliminating backward passes, its application to multi-hundred-billion-parameter models is constrained by GPU memory and compute throughput. The ZO2 framework addresses the memory bottleneck by offloading model parameters to CPU memory and overlapping transformer block transfer with dual forward computation on a single GPU. However, ZO2 remains limited by its single-device execution and achieves modest throughput. In this work, we present DistZO2, a high-throughput, memory-efficient framework for distributed zeroth-order fine-tuning of LLMs. DistZO2 introduces three parallel strategies: (1) Perturbation Parallelism (PertP), which parallelizes the two perturbed forward passes across devices; (2) Distributed Data Parallelism (DDP), adapted to the scalar-gradient nature of ZO training; and (3) a unified 2D Parallelism design that combines PertP and DDP. To further mitigate communication bottlenecks introduced by parameter offloading, we propose a hardware-aware communication strategy that slices parameter blocks and redistributes them across GPUs via high-speed interconnects such as NVLink. DistZO2 scales zeroth-order fine-tuning to modern multi-GPU systems, preserving ZO2's memory efficiency while substantially improving training throughput. In our experiments on OPT-175B, DistZO2 achieves a 3x speedup over ZO2 with distributed computing. DistZO2's code has been open-sourced in https://github.com/liangyuwang/zo2.
</details>

### <span class="venue-tag venue-conf">NeurIPS 2025</span> FlashDP: Private Training Large Language Models with Efficient DP-SGD  
**Liangyu Wang**, Junxiao Wang, Jie Ren, Zihang Xiang, David E. Keyes, and Di Wang  
[[paper](https://openreview.net/pdf?id=b6SWqFEOSF)] [[!code](https://github.com/kaustpradalab/flashdp)]

<details>
<summary><strong>Abstract</strong></summary>
As large language models (LLMs) increasingly underpin technological advancements, the privacy of their training data emerges as a critical concern. Differential Privacy (DP) serves as a rigorous mechanism to protect this data, yet its integration via Differentially Private Stochastic Gradient Descent (DP-SGD) introduces substantial challenges, primarily due to the complexities of per-sample gradient clipping. Current explicit methods, such as Opacus, necessitate extensive storage for per-sample gradients, significantly inflating memory requirements. Conversely, implicit methods like GhostClip reduce storage needs by recalculating gradients multiple times, which leads to inefficiencies due to redundant computations. This paper introduces FlashDP, an innovative cache-friendly per-layer DP-SGD that consolidates necessary operations into a single task, calculating gradients only once in a fused manner. This approach not only diminishes memory movement by up to 50% but also cuts down redundant computations by 20%, compared to previous methods. Consequently, FlashDP does not increase memory demands and achieves a 90% throughput compared to the Non-DP method on a four-A100 system during the pre-training of the Llama-13B model, while maintaining parity with standard per-layer clipped DP-SGD in terms of accuracy. These advancements establish FlashDP as a pivotal development for efficient and privacy-preserving training of LLMs. FlashDP's code has been open-sourced in https://github.com/kaustpradalab/flashdp.
</details>

### <span class="venue-tag venue-preprint">Preprint</span> Infinite-Sampling: Efficient and Stable Grouped RL Training for Large Language Models  
**Liangyu Wang**, Huanyi Xie, Xinhai Wang, Tianjin Huang, Mengdi Li, and Di Wang  
[[paper](https://arxiv.org/pdf/2506.22950)]

<details>
<summary><strong>Abstract</strong></summary>
Group-based reinforcement learning algorithms such as Group Reward Policy Optimization (GRPO) have proven effective for fine-tuning large language models (LLMs) with human feedback. However, generating and storing multiple responses per prompt incurs substantial memory overhead, especially as the sample group size increases, limiting scalability under constrained hardware. We propose Infinite Sampling, a framework that enables efficient and stable GRPO training by decoupling group size from GPU memory usage. It consists of: (1) micro sampling groups that decompose large groups into memory-feasible rounds; (2) continuous sampling that interleaves generation across groups to improve utilization; and (3) a length-aware scheduler combining token-conditioned sequence length prediction with a two-stage plan: global grouping via FPTAS and runtime refill via SJF. Experiments show that our Micro Sampling Groups reduce peak memory usage by over 50% compared to full-group decoding (e.g., from 21.55 GB to 10.64 GB on Qwen3-1.7B). Building on this, Infinite Sampling improves throughput by over 25% compared to the naive micro sampling group method, reducing decoding steps while maintaining full-length completions and memory usage. Our hybrid scheduling ensures efficient and stable GRPO training with larger groups under realistic GPU memory constraints.
</details>

### <span class="venue-tag venue-conf">COLM 2025</span> ZO2: Scalable Zeroth-Order Fine-Tuning for Extremely Large Language Models with Limited GPU Memory  
**Liangyu Wang**, Jie Ren, Hang Xu, Junxiao Wang, Huanyi Xie, David E. Keyes, and Di Wang  
[[paper](https://openreview.net/pdf?id=s0p9xpORgP)] [[code](https://github.com/liangyuwang/zo2)]

<details>
<summary><strong>Abstract</strong></summary>
Fine-tuning large pre-trained LLMs generally demands extensive GPU memory. Traditional first-order optimizers like SGD encounter substantial difficulties due to increased memory requirements from storing activations and gradients during both the forward and backward phases as the model size expands. Alternatively, zeroth-order (ZO) techniques can compute gradients using just forward operations, eliminating the need to store activations. Furthermore, by leveraging CPU capabilities, it's feasible to enhance both the memory and processing power available to a single GPU. We propose a novel framework, ZO2 (Zeroth-Order Offloading), for efficient zeroth-order fine-tuning of LLMs with only limited GPU memory. Our framework dynamically shifts model parameters between the CPU and GPU as required, optimizing computation flow and maximizing GPU usage by minimizing downtime. This integration of parameter adjustments with ZO's double forward operations reduces unnecessary data movement, enhancing the fine-tuning efficacy. Additionally, our framework supports an innovative low-bit precision approach in AMP mode to streamline data exchanges between the CPU and GPU. Employing this approach allows us to fine-tune extraordinarily large models, such as the OPT-175B with more than 175 billion parameters, on a mere 18GB GPU--achievements beyond the reach of traditional methods. Moreover, our framework achieves these results with almost no additional time overhead and absolutely no accuracy loss compared to standard zeroth-order methods. ZO2's code has been open-sourced in https://github.com/liangyuwang/zo2.
</details>

### <span class="venue-tag venue-journal">IEEE TASE</span> V2-SfMLearner: Learning Monocular Depth and Ego-motion for Multimodal Wireless Capsule Endoscopy  
Long Bai, Beilei Cui, **Liangyu Wang**, Yanheng Li, Shilong Yao, Sishen Yuan, Yanan Wu, Yang Zhang, Max Q.-H. Meng, Zhen Li, Weiping Ding, and Hongliang Ren  
[[paper](https://doi.org/10.1109/TASE.2024.3516968)]

<details>
<summary><strong>Abstract</strong></summary>
Deep learning can predict depth maps and capsule ego-motion from capsule endoscopy videos, aiding in 3D scene reconstruction and lesion localization. However, the collisions of the capsule endoscopies within the gastrointestinal tract cause vibration perturbations in the training data. Existing solutions focus solely on vision-based processing, neglecting other auxiliary signals like vibrations that could reduce noise and improve performance. Therefore, we propose V2-SfMLearner, a multimodal approach integrating vibration signals into vision-based depth and capsule motion estimation for monocular capsule endoscopy. We construct a multimodal capsule endoscopy dataset containing vibration and visual signals, and our artificial intelligence solution develops an unsupervised method using vision-vibration signals, effectively eliminating vibration perturbations through multimodal learning. Specifically, we carefully design a vibration network branch and a Fourier fusion module, to detect and mitigate vibration noises. The fusion framework is compatible with popular vision-only algorithms. Extensive validation on the multimodal dataset demonstrates superior performance and robustness against vision-only algorithms. Without the need for large external equipment, our V2-SfMLearner has the potential for integration into clinical capsule robots, providing real-time and dependable digestive examination tools. The findings show promise for practical implementation in clinical settings, enhancing the diagnostic capabilities of doctors. Note to Practitioners—This paper is motivated by the problem of estimating the depth and ego-motion information for the wireless capsule endoscopy in the human gastrointestinal tract to realize accurate, efficient, robust, and real-time inspection. Our estimation method does not engage any external localization equipment. Instead, inspired by the existing research on integrating capsule endoscopy and inertial measurement units, we introduce vibration signals into vision-based depth and ego-motion estimation approaches, improving the accuracy and robustness of the estimation results based on multimodal learning methods. Research on capsule robots or computer vision can readily be combined with our framework for various clinical and industrial applications.
</details>

---

## 2024

### <span class="venue-tag venue-workshop">MobiSys Workshop 2024</span> WiP: Towards Light Adaptation of Large Language Models For Personal Hardware  
**Liangyu Wang**, Junxiao Wang, and Di Wang  
[[paper](https://dl.acm.org/doi/pdf/10.1145/3662006.3662065)]

<details>
<summary><strong>Abstract</strong></summary>
The large language models (LLMs) that everyone is using are not deployed locally. Users need to send relatively private and important data to LLM when using it. Handing over private and important data to LLM will cause people to worry, especially now that many people have begun to use LLM to deal with life and work affairs. Such concerns cannot be easily dispelled by various guarantees and agreements. However, LLMs are often resource-intensive and computationally demanding, making the transition from server-side to device-side difficult because LLM's self-attention module contains a large number of tensor multiplications that are heavy and inefficient for hardware. While previous work proposed approximate neural operators that enable hardware-efficient implementation of multiplication-less neural networks, they introduce new challenges of significant accuracy loss, making these methods inefficient in practice. In this paper, we examine the problem of light adaptation of LLMs. We propose a new neural operator that enables the adapted LLM to obtain original accuracy without fine-tuning or only requiring a few fine-tuning steps, while our neural operator has high hardware inference efficiency.
</details>

---

## 2022

### <span class="venue-tag venue-journal">Electronics 2022</span> Transformer-Based Disease Identification for Small-Scale Imbalanced Capsule Endoscopy Dataset  
Long Bai, **Liangyu Wang**, Tong Chen, Yanheng Zhao, and Hongliang Ren  
[[paper](https://www.mdpi.com/2079-9292/11/17/2747)]

<details>
<summary><strong>Abstract</strong></summary>
Vision Transformer (ViT) is emerging as a new leader in computer vision with its outstanding performance in many tasks (e.g., ImageNet-22k, JFT-300M). However, the success of ViT relies on pretraining on large datasets. It is difficult for us to use ViT to train from scratch on a small-scale imbalanced capsule endoscopic image dataset. This paper adopts a Transformer neural network with a spatial pooling configuration. Transfomer's self-attention mechanism enables it to capture long-range information effectively, and the exploration of ViT spatial structure by pooling can further improve the performance of ViT on our small-scale capsule endoscopy dataset. We trained from scratch on two publicly available datasets for capsule endoscopy disease classification, obtained 79.15% accuracy on the multi-classification task of the Kvasir-Capsule dataset, and 98.63% accuracy on the binary classification task of the Red Lesion Endoscopy dataset.
</details>
