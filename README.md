# Reason4Rec

The implementation of our paper "[Reason4Rec: Large Language Models for Recommendation with Deliberative User Preference Alignment](https://arxiv.org/abs/2502.02061)"

## Introduction

- We formulate the *Deliberative Recommendation* task, which pursues LLMs conducting reasoning before making prediction by learning from verbalized user feedback. 
- We propose a *Reasoning-powered Recommender* framework (Reason4Rec) for deliberative user preference alignment, which achieves the reasoning process with step-wise experts associated with specifically designed training strategies. 
- We conduct extensive experiments on three datasets, validating the effectiveness and rationality of the proposed Reason4Rec framework, showing the potential of slow thinking in recommendation.

<div align="center">
    <img src=".\figs\teaser.png" alt="teaser" width="600px" />
    <p style="color: gray;">Figure 1. Comparison between the alignment objective of existing research, which optimizes LLMs to directly predict user feedback; and the objective of Deliberative Recommendation, which optimizes LLMs to conduct explicit reasoning about user preferences before generating the prediction.</p>
</div>

## Framework

Reason4Rec utilizes multi-step reasoning via three collaborative experts with three core reasoning capabilities: Preference Distillation, Preference Matching, and Feedback Prediction. To align the reasoning process with users' true preferences, verbalized user feedback, i.e., reviews, is utilized.

<div align="center">
    <img src=".\figs\framework.png" alt="framework" width="600px" />
    <p style="color: gray;">Figure 2. Illustration of the Reasoning-powered Recommender framework.</p>
</div>

## Getting Started

### Preparation

- We leverage the [Unsloth](https://github.com/unslothai/unsloth) framework to accelerate the training and inference process. To install it, use the following command:

  ``` bash
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 
  ```

  For most installations, this should suffice. Alternatively, you can refer to the [official installation instructions](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions) for further details.

- Install additional dependencies:

  ``` bash
  pip3 install -r requirements.txt
  ```

- Set your openai API_KEY in `utils.py`

### Dataset

We upload both the raw and pre-processed data on Google Drive. You can download them from [here](https://drive.google.com/file/d/1Hcw_c8Qc3H2szKXQ5jR1TE_cCDB4HVYw/view?usp=drive_link) and unzip the files into the `./Data` directory.

### Training

#### Option 1: Train from Scratch

You can train your own Summarizer, Reasoner, and Predictor by running the files in each folder sequentially, following the numerical order of the filenames.

#### Option 2: Continue from Pre-Processed Data

Alternatively, we have made our pre-processed data available, including the summarizer’s generation results (`train_summarizer_generation_results.pkl`) and high-quality reasoning training data (`distilling_high_quality_reasons.pkl`). You can continue training from our pre-processed data, allowing you to focus solely on training the Reasoner and Predictor. This approach saves you the time required to train the Summarizer and Reward Model, and eliminates the need for distilling data from ChatGPT, ultimately reducing both time and cost. 

1. Training the reasoner:

   ``` bash
   python 2_Reasoner/5_construct_reasoner_train_data.py
   bash 2_Reasoner/6_reasoner_train.sh
   ```

2. Training the predictor:

   ``` python
   python 3_Predictor/1_construct_predictor_train_data.py
   bash 3_Predictor/2_predictor_train.sh
   ```

3. Generate Reasons and Predictions for the Test:

   ``` python
   python 2_Reasoner/7_generate_reason_for_test.py
   python 3_Predictor/3_generate_predict_for_test.py
   ```

## Citation

```bibtex
@article{fang2025large,
  title={Reason4Rec: Large Language Models for Recommendation with Deliberative User Preference Alignment},
  author={Fang, Yi and Wang, Wenjie and Zhang, Yang and Zhu, Fengbin and Wang, Qifan and Feng, Fuli and He, Xiangnan},
  journal={arXiv preprint arXiv:2502.02061},
  year={2025}
}
```
