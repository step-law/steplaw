<p align="center">
    <img src="assets/logo.jpeg" alt="Step Law" width="200">
</p>



<p align="center">
    <h1 align="center">
        Predictable Scale: Part I
    </h1>
</p>

<p align="center">
        <a href="https://step-law.github.io/">Home Page</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://wandb.ai/billzid/predictable-scale">Wandb</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://arxiv.org/abs/2503.04715">Paper</a>
</p>


## News
- 🔥 ```2024/03/10``` We have released our fitting code and fitted model parameters! Check them in the `code` and `data` folders.
- 🔥 ```2024/03/09``` We have released our training logs! Access them on [Wandb](https://wandb.ai/billzid/predictable-scale).
- 🔥 ```2024/03/08``` All smooth loss heatmaps have been released on our  [homepage](https://step-law.github.io/#steplawtool).
- 🔥 ```2024/03/08``` We have launched an optimal hyperparameter tool for the community on our [homepage](https://step-law.github.io/#steplawtool).
- 🔥 ```2025/03/07``` We have released our paper on Arxiv: 📄 [Predictable Scale: Part I -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining](https://arxiv.org/abs/2503.04715).


## 🗓 Coming Soon
- [x] Paper
- [x] Smooth loss heatmaps
- [x] Thousands of training logs
- [x] Fitting code
- [ ] Checkpoints


## Introduction

We first present the unified optimal hyperparameter scaling laws, termed Step Law, that generalizes across diverse model shapes, architectures, and data distributions.

Our findings demonstrate remarkable accuracy, with estimated values on test sets deviating by only 0.09% from the globally optimal LLM performance identified through exhaustive search.

This research entails a significant computational investment, utilizing nearly one million NVIDIA H800 GPU hours to train 3,700 LLMs of varying sizes and hyperparameters from scratch, consuming approximately 100 trillion tokens in total. To support reproducibility and advance the field for LLM pre-training, we will progressively release all loss measurements and model checkpoints through our designated repository. The universal, plug-and-play [optimal hyperparameter tool](https://step-law.github.io/#steplawtool) is provided for the community.

## Usage

The repository provides tools and data for predicting optimal learning rate and batch size for LLM pretraining:

### Data Files

The Data folder contains:
- Smooth loss results for both dense and MoE models (two CSV files)
- Structure and training configurations for each model
- `data/1004_fitted_lr_bs_scaling_model_parameters.csv`: Contains fitted model parameters from 1000 bootstrap models for robust prediction of optimal learning rate and batch size. The model follows the form:
  - lr = exp(intercept) * N^coefN * D^coefD
  - bs = exp(intercept) * D^coefD

### Prediction Tool

We provide a simple command line tool to predict optimal learning rate and batch size based on your model parameters:

```bash
python code/fit_tool.py pred-opt-lr-bs [model_params] [data_in_token] [seq_len]
```

Parameters:
- `model_params`: Number of model parameters
- `data_in_token`: Training data size in tokens
- `seq_len`: Sequence length

Example:
```bash
python code/fit_tool.py pred-opt-lr-bs 7e9 1.4e12 2048
```

For more training details and experimental results, please refer to our [Wandb](https://wandb.ai/billzid/predictable-scale) page.

## Citation
If you find our work helpful, feel free to give us a cite :-)

```bibtex
@misc{li2025predictablescalei,
      title={Predictable Scale: Part I -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining}, 
      author={Houyi Li and Wenzheng Zheng and Jingcheng Hu and Qiufeng Wang and Hanshan Zhang and Zili Wang and Yangshijie Xu and Shuigeng Zhou and Xiangyu Zhang and Daxin Jiang},
      year={2025},
      eprint={2503.04715},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.04715}, 
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=step-law/steplaw&type=Date)](https://star-history.com/#step-law/steplaw&Date)