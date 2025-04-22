---
title: "ZO2: Zeroth-Order Offloading"
excerpt: "Fine-tuning 175B parameter LLMs with only 18GB GPU memory<br/><img src='/images/zo2_hero.svg' alt='ZO2 Architecture'>"
collection: projects
---

<style>
.project-header {
  background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
  color: white;
  padding: 2em;
  border-radius: 10px;
  margin-bottom: 2em;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.feature-box {
  background-color: #f8f9fa;
  border-left: 4px solid #4b6cb7;
  padding: 1.5em;
  margin-bottom: 1.5em;
  border-radius: 0 5px 5px 0;
}
.code-sample {
  background-color: #f1f1f1;
  border-radius: 5px;
  padding: 1em;
  overflow-x: auto;
  font-family: monospace;
  margin-bottom: 1.5em;
}
.highlight-text {
  font-weight: bold;
  color: #4b6cb7;
}
</style>

<div class="project-header">
  <h1>ZO2: Zeroth-Order Offloading</h1>
  <p>A framework that enables fine-tuning of extremely large language models (like OPT-175B) on limited GPU memory through zeroth-order optimization and CPU-GPU offloading.</p>
</div>

## Overview

Large Language Models (LLMs) with billions of parameters have shown remarkable capabilities, but fine-tuning these models is challenging due to extensive GPU memory requirements. Traditional first-order optimization methods like SGD require storing activations and gradients during both forward and backward phases, making them impractical for extremely large models on consumer hardware.

ZO2 solves this problem by combining two key innovations:

<div class="feature-box">
  <p><span class="highlight-text">Zeroth-Order Optimization:</span> Instead of using gradients computed through backpropagation, ZO2 uses forward-pass only gradient approximation, eliminating the need to store activations.</p>
</div>

<div class="feature-box">
  <p><span class="highlight-text">CPU-GPU Offloading:</span> ZO2 dynamically shifts model parameters between CPU and GPU as needed, optimizing memory usage and computation flow.</p>
</div>

## Key Features

- **Fine-tune 175B parameter models on a single consumer GPU** with as little as 18GB of memory
- **No accuracy loss** compared to standard zeroth-order methods
- **Minimal time overhead** through optimized CPU-GPU transfer scheduling
- **Low-bit precision support** in AMP mode for efficient data exchange
- **Compatible with popular LLMs** including OPT, LLaMA, and others

## How It Works

ZO2 integrates parameter offloading with zeroth-order optimization's double forward operations:

1. **Dynamic Parameter Management**: Parameters are stored primarily in CPU memory and loaded to GPU only when needed
2. **Optimized Gradient Approximation**: Uses efficient zeroth-order methods that require only forward passes
3. **Intelligent Scheduling**: Minimizes unnecessary data transfers between CPU and GPU

<div class="code-sample">
<pre>
# Example ZO2 usage
from zo2 import ZO2Optimizer, CPUOffloader

model = get_large_language_model()  # Your 175B parameter model
offloader = CPUOffloader(model)
optimizer = ZO2Optimizer(model.parameters(), lr=1e-4)

for inputs, targets in dataloader:
    # Parameters automatically managed between CPU/GPU
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    
    # Zeroth-order gradient computation and update
    optimizer.zero_grad()
    optimizer.step(loss)
</pre>
</div>

## Results

ZO2 enables fine-tuning of the OPT-175B model (175 billion parameters) on a single GPU with only 18GB of memory—an achievement beyond the reach of traditional methods.

![Performance Comparison](/images/zo2_performance.svg)

## Resources

- [Paper](https://arxiv.org/abs/2503.12668)
- [GitHub Repository](https://github.com/liangyuwang/zo2)
- [Documentation](https://github.com/liangyuwang/zo2/wiki)

## Citation

```
@article{wang2025zo2,
  title={ZO2: Scalable Zeroth-Order Fine-Tuning for Extremely Large Language Models with Limited GPU Memory},
  author={Wang, Liangyu and Ren, Jie and Xu, Hang and Wang, Junxiao and Xie, Huanyi and Keyes, David E and Wang, Di},
  journal={arXiv preprint arXiv:2503.12668},
  year={2025}
}
``` 