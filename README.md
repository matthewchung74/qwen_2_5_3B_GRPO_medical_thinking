# Qwen2_5_(3B)_GRPO Notebook

<a href="https://github.com/unslothai/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>

This repository contains a Jupyter notebook designed to demonstrate the use of the Unsloth library for implementing Generalized Reinforcement Policy Optimization (GRPO) with custom reward functions. The notebook is specifically tailored for handling unstructured text using a medical dataset, and it leverages advanced reward functions to enhance model performance.

## Overview

The notebook provides a comprehensive guide to setting up and running experiments with GRPO, focusing on the following key areas:

- **Environment Setup**: Installation of necessary Python packages and configuration of the environment for optimal performance
- **Unsloth Integration**: Utilization of the Unsloth library to optimize system performance and enable faster model fine-tuning
- **Custom Reward Functions**: Implementation of reward functions that evaluate semantic correctness and perplexity to guide the model's learning process
- **Medical Dataset Utilization**: Training and evaluation of the model using the [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset
- **Training Environment**: The model was trained on [Runpod](Runpod.io) using an NVIDIA RTX 3090 GPU, providing a robust and efficient training setup. 

## Key Features

- **Advanced Reward Functions**: Custom reward functions that assess the quality of generated text based on semantic similarity and linguistic fluency
- **GRPO with Unsloth**: Leverages the Unsloth library to enhance the GRPO algorithm, improving the model's ability to generate coherent and contextually relevant responses
- **Medical Reasoning**: Focuses on improving the model's performance in medical reasoning tasks, making it suitable for applications in healthcare and related fields

## Getting Started

To get started with this notebook:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Qwen2_5_3B_GRPO.git
   cd Qwen2_5_3B_GRPO
   ```

2. **Run the Notebook**:
   Open the Jupyter notebook and execute the cells to set up the environment, train the model, and evaluate its performance

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Unsloth library for providing tools to optimize reinforcement learning models
- The [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset for enabling medical reasoning experiments

For more information, please refer to the [Unsloth GRPO blog](https://unsloth.ai/blog/grpo) and the [Hugging Face documentation](https://huggingface.co/docs).
