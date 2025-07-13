---
title: "V2-SfMLearner: Learning Monocular Depth and Ego-motion for Multimodal Wireless Capsule Endoscopy"
collection: publications
category: conferences
permalink: /publication/2025-01-16-v2-sfmlearner
date: 2025-01-16
venue: 'IEEE Transactions on Automation Science and Engineering'
paperurl: 'https://doi.org/10.1109/TASE.2024.3516968'
authors: 'Long Bai, Beilei Cui, Liangyu Wang, Yanheng Li, Shilong Yao, Sishen Yuan, Yanan Wu, Yang Zhang, Max Q.-H. Meng, Zhen Li, Weiping Ding, and Hongliang Ren'
abstract: 'A multimodal approach integrating vibration signals into vision-based depth and capsule motion estimation for monocular capsule endoscopy.'
---

## V2-SfMLearner: Learning Monocular Depth and Ego-motion for Multimodal Wireless Capsule Endoscopy

[**Paper**](https://doi.org/10.1109/TASE.2024.3516968)

<details>
<summary><strong>Abstract</strong></summary>
Deep learning can predict depth maps and capsule ego-motion from capsule endoscopy videos, aiding in 3D scene reconstruction and lesion localization. However, the collisions of the capsule endoscopies within the gastrointestinal tract cause vibration perturbations in the training data. Existing solutions focus solely on vision-based processing, neglecting other auxiliary signals like vibrations that could reduce noise and improve performance. Therefore, we propose V2-SfMLearner, a multimodal approach integrating vibration signals into vision-based depth and capsule motion estimation for monocular capsule endoscopy. We construct a multimodal capsule endoscopy dataset containing vibration and visual signals, and our artificial intelligence solution develops an unsupervised method using vision-vibration signals, effectively eliminating vibration perturbations through multimodal learning. Specifically, we carefully design a vibration network branch and a Fourier fusion module, to detect and mitigate vibration noises. The fusion framework is compatible with popular vision-only algorithms. Extensive validation on the multimodal dataset demonstrates superior performance and robustness against vision-only algorithms. Without the need for large external equipment, our V2-SfMLearner has the potential for integration into clinical capsule robots, providing real-time and dependable digestive examination tools. The findings show promise for practical implementation in clinical settings, enhancing the diagnostic capabilities of doctors. Note to Practitioners—This paper is motivated by the problem of estimating the depth and ego-motion information for the wireless capsule endoscopy in the human gastrointestinal tract to realize accurate, efficient, robust, and real-time inspection. Our estimation method does not engage any external localization equipment. Instead, inspired by the existing research on integrating capsule endoscopy and inertial measurement units, we introduce vibration signals into vision-based depth and ego-motion estimation approaches, improving the accuracy and robustness of the estimation results based on multimodal learning methods. Research on capsule robots or computer vision can readily be combined with our framework for various clinical and industrial applications.
</details>

**Authors:** Long Bai, Beilei Cui, Liangyu Wang, Yanheng Li, Shilong Yao, Sishen Yuan, Yanan Wu, Yang Zhang, Max Q.-H. Meng, Zhen Li, Weiping Ding, and Hongliang Ren

**Published in:** IEEE Transactions on Automation Science and Engineering 