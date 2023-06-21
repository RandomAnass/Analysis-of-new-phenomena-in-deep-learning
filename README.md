# Practical Course:
## Analysis of new phenomena in deep learning: Shift Invariance Can Reduce Adversarial Robustness

![GitHub commit activity](https://img.shields.io/github/commit-activity/y/RandomAnass/Analysis-of-new-phenomena-in-deep-learning)

![GitHub last commit](https://img.shields.io/github/last-commit/RandomAnass/Analysis-of-new-phenomena-in-deep-learning)

![GitHub repo size](https://img.shields.io/github/repo-size/RandomAnass/Analysis-of-new-phenomena-in-deep-learning)


# Reproduction:

In this project we explore the impact of shift invariance on the adversarial robustness of deep learning models.
The data is the next drive link: [https://drive.google.com/file/d/1sNJ0y0-fZpSVKeW_0Djs_j2yAeMZlw_s/view?usp=sharing](https://drive.google.com/file/d/1sNJ0y0-fZpSVKeW_0Djs_j2yAeMZlw_s/view?usp=sharing)

**Note**: In the notebook the green sections and the underlined text mean it's not a reproduction but an addition.

## Table of Contents
- [Documentation](##Documentation)
- [Installation](##Installation)
- [Using LRZ](##Using_LRZ)
- [Bibliography](##Bibliography)



## Documentation
- `Shift_Invariance_Can_Reduce_Adversarial_Robustness.html`: HTML version of the notebook
- `Shift_Invariance_Can_Reduce_Adversarial_Robustness.ipynb`: Reproducibility report: Jupyter Notebook
- `Shift_Invariance_Can_Reduce_Adversarial_Robustness.pdf`: PDF version of the notebook.


## Installation
To use this project: 
- After cloning the repository. Please install the dependencies, using requirements (pip install -r requirements.txt)
- Download and unzip the data from [the google drive link](https://drive.google.com/file/d/1sNJ0y0-fZpSVKeW_0Djs_j2yAeMZlw_s/view?usp=sharing).
- The file tree of the repository should look like this:
```
├── data
│   ├── adversarial_data_2
│   │   └── ... .npz files
│   ├── adversarial_fashion_data
│   │   └── ... .npz files
│   ├── fashion_adversarial_data_2
│   │   └── ... .npz files
│   ├── mnist_adversarial_data
│   │   └── ... .npz files
│   └── padding
│       ├── logs
│       ├── logs_2 : logs for tensorboard
│       └── padding_experiment
│           └── padding models (.h5)
├── functions
│   ├── dots_models.py
│   └── utils.py
├── models
│   ├── fashion_mnist_2
│   ├── fashion_mnist_main
│   ├── mnist_2
│   └── mnist_main
├── plots
│   ├── black_white_dot.png
│   ├── teaser.png
│   └── ...
├── requirements.txt
├── requirements_LRZ.txt
├── scripts
│   ├── DDN_plot.py
│   ├── fashion_mnist_gpu_create_attack_data.py
│   ├── fashion_mnist_gpu_create_models.py
│   ├── full_fashion_mnist_gpu_2.py
│   ├── full_mnist_gpu.py
│   ├── full_mnist_gpu_2.py
│   └── padding_plot.py
├── notebooks
├── paper refactored code (executable changed code)
├── paper original code (orginal code from the paper)
├── Introduction_Adversarial_attacks.ipynb
├── Shift_Invariance_Can_Reduce_Adversarial_Robustness.ipynb
└── Shift_Invariance_Can_Reduce_Adversarial_Robustness.pdf
```
## Using_LRZ
To train the models and produce adversarial data we used [LRZ](https://doku.lrz.de/lrz-ai-systems-11484278.html). After setting up the [Enroot Container Image](https://doku.lrz.de/5-using-nvidia-ngc-containers-on-the-lrz-ai-systems-10746648.html).You can install the dependencies using, `requirements_LRZ.txt` 
To use tensorflow on gpu it's recommended to follow [these steps](https://www.tensorflow.org/install/pip) by setting up a conda envirement.

* We can then clone this repository: https://gitlab.lrz.de/tfai-practikum-2023/g2-robustness-2/shift-invariance-can-reduce-adversarial-robustness.git, or the github repository we used: 
https://github.com/RandomAnass/Analysis-of-new-phenomena-in-deep-learning.git
* `cd Analysis-of-new-phenomena-in-deep-learning`

* `salloc -p lrz-dgx-1-p100x8 --gres=gpu:8 --time=200`
Check [General Description and Resources](https://doku.lrz.de/1-general-description-and-resources-10746641.html) For other options. For smaller tests lrz-v100x2 is used, otherwise the DGX-1 P100 Architecture or DGX-1 V100 Architecture with 4 to 8 gpus.

* `srun --pty bash`

* `conda activate tf`

* `nvidia-smi` (to check the gpus before running the code)

* `python3 script.py`

To update a file we can use `git`, or the command `wget`.
And once the data is produced we can get it from the server using any scp tool. ( We used FileZilla for its simplicity)






## Bibliography

   - [nvcr.io/nvidia/tensorflow:23.04-tf2-py3](http://nvcr.io/nvidia/tensorflow:23.04-tf2-py3)

  - https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow

  -  https://doku.lrz.de/5-using-nvidia-ngc-containers-on-the-lrz-ai-systems-10746648.html
  - Singla, Vasu, et al. Shift Invariance Can Reduce Adversarial Robustness. ArXiv.org,22 Nov 2021 https://arxiv.org/pdf/2103.02695.pdf.

   - Alireza, et al. “Revisiting DeepFool: Generalization and Improvement.” ArXiv.org, 22 Mar. 2023, arxiv.org/abs/2303.12481, https://doi.org/10.48550/arXiv.2303.12481.

   - Rony, Jérôme, et al. “Decoupling Direction and Norm for Efficient Gradient-Based L2 Adversarial Attacks and Defenses.” ArXiv.org, 3 Apr. 2019, arxiv.org/abs/1811.09600.

   - Xiao, Han, et al. “Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms.” ArXiv:1708.07747 [Cs, Stat], 15 Sept. 2017, arxiv.org/abs/1708.07747.

   - Zhang, Richard. “Making Convolutional Networks Shift-Invariant Again.” ArXiv.org, 8 June 2019, arxiv.org/abs/1904.11486.

   - “Welcome to the Adversarial Robustness Toolbox — Adversarial Robustness Toolbox 1.7.0 Documentation.” Adversarial-Robustness-Toolbox.readthedocs.io, adversarial-robustness-toolbox.readthedocs.io/en/latest/.