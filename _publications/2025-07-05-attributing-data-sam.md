---
title: "Attributing Data for Sharpness-Aware Minimization"
collection: publications
category: conferences
permalink: /publication/2025-07-05-attributing-data-sam
date: 2025-07-05
venue: 'arXiv preprint'
paperurl: 'https://arxiv.org/abs/2507.04059'
authors: 'Chenyang Ren, Yifan Jia, Huanyi Xie, Zhaobin Xu, Tianxing Wei, Liangyu Wang, Lijie Hu, and Di Wang'
abstract: 'A method for data attribution in sharpness-aware minimization to identify the contributions of specific training samples.'
---

## Attributing Data for Sharpness-Aware Minimization

[**Paper**](https://arxiv.org/abs/2507.04059)

<details>
<summary><strong>Abstract</strong></summary>
Sharpness-aware Minimization (SAM) improves generalization in large-scale model training by linking loss landscape geometry to generalization. However, challenges such as mislabeled noisy data and privacy concerns have emerged as significant issues. Data attribution, which identifies the contributions of specific training samples, offers a promising solution. However, directly rendering existing data influence evaluation tools such as influence functions (IF) to SAM will be inapplicable or inaccurate as SAM utilizes an inner loop to find model perturbations that maximize loss, which the outer loop then minimizes, resulting in a doubled computational structure. Additionally, this bilevel structure complicates the modeling of data influence on the parameters. In this paper, based on the IF, we develop two innovative data valuation methods for SAM, each offering unique benefits in different scenarios: the Hessian-based IF and the Gradient Trajectory-based IF. The first one provides a comprehensive estimation of data influence using a closed-form measure that relies only on the trained model weights. In contrast, the other IF for SAM utilizes gradient trajectory information during training for more accurate and efficient data assessment. Extensive experiments demonstrate their effectiveness in data evaluation and parameter tuning, with applications in identifying mislabeled data, model editing, and enhancing interpretability.
</details>

**Authors:** Chenyang Ren, Yifan Jia, Huanyi Xie, Zhaobin Xu, Tianxing Wei, Liangyu Wang, Lijie Hu, and Di Wang

**Published in:** arXiv preprint 