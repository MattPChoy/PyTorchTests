# Domain Adversarial Neural Networks (DANNs)
## Interesting Resources
 - [GitHub/fungtion/DANN](https://github.com/fungtion/DANN) Python 2.7, PyTorch 1.0
 - [GitHub/NaJaeMin92/pytorch_DANN](https://github.com/NaJaeMin92/pytorch_DANN) Python 3.5, pytorch 0.4.1

## Sections
 1. [Domain Adaptation](#domain-adaptation) (A naive approach to DANNs)
 2. [Domain Adversarial Neural Networks](#domain-adversarial-neural-networks)

# Domain Adaptation
[1] [Introduction to Domain Adversarial Neural Networks - ElderResearch](https://www.elderresearch.com/blog/introduction-to-domain-adversarial-neural-networks/)
 - Domain Adaptation (DA) is the process for enhancing model training when there is a **shift between input distributions** for different datasets
   - This shift is sometimes referred to as a co-variant shift or data shift.
 - Domain Adaptation has two objectives:
   - `Discriminativeness` The ability to discriminate between data coming from different classes within a particular Domain
   - `Domain Invariance` The ability to measure the similarity between data classes across domains.
- In the context of a `Classification Model`, we want to discern the differencees among classes (maintina `Discriminativeness`) but also want to maintain `domain invariance` so that our classifier can generalise across inputs from multiple domains.

### Real-World use of Domain Adaptation.
- The MNIST dataset is a dataset consisting of images of handwriting.
- There exists a dataset called MNIST-M which adds different backgrounds and digit colours.
  - The variations in the MNIST-M dataset lead to different distributions of input features.
  - Therefore, `Domain Adaptation` could be used to generalise a MNIST model to work with MNIST-M
![DANN-MNIST-vs-MNIST_M](/assets/DANN-Blog_Figure2.webp)

## Sample Re-Weighting
> Sample Re-Weighting is a technique commonly used to implement Domain Adaptation.
- When using Sample-Reweighting, we develop a domain classifier with the following steps:
  1. Label all source domain samples `0` and target domain samples `1`
  2. Train a `binary classifier` that can return predicted probabilities $p_i$ (such as logistic regression or random forest) to discriminate between source data and target data
  3. Use resultant probabilities to obtains ample weights for source domain samples when model fitting on the source domain:
  $$ w_i = \frac{p_i}{1-p_i} $$
- This causes the samples that look most like the target samples to get a higher weights
  - This method has some drawbacks
    1. How do we determine how accurate to drive the domain classifier to be?
    - If it is too accurate, it won't be useful (no overlap between the target and domain regions).

# Domain Adversarial Neural Networks
- DANNs allow us to use DA whilst learning label classifications at the same time.
- Employs source data that has class labels (and target data that is unlabelled)
- Goal is to predict the target data by using the source data and the target data in an adversarial training process
## DANN Model Architecture
![DANN-MNIST-vs-MNIST_M](/assets/DANN-Blog_Figure4.webp)
- A DANN Model has three key components:
 1. `Feature Extractor` Produces features that are used as inputs to the `Label Predictor` and `Domain Classifier` in the training process.
 2. `Label Predictor` Predicts the class labels
 3. `Domain Classifier` Predicts the domain of the output

- We essentially want to optimise the Feature Extractor such that we find the best tradeoff between:
  1. Producing features that are domain invariant
  2. Producing features that are useful for the label predictor
- In other words, we want to:
  1. Minimise the loss of the label predictor
  2. Maximise the loss for the Domain Classifier (if the domain classifier cannot distinguish the domains, then the features are said to be `Domain Invariant`)
