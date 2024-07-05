# Gradient-based learning for Spiking Neural Networks
This is a part of my project from the University of Nottingham for MSc Machine Learning in Science.

## Requirements
The code has been tested on RTX3050 locally with PyTorch 1.12 and Cuda 11.3.
```Shell
conda create --name snn_eventprop python=3.7
conda activate snn_eventprop
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch 
```
Next, type
```Shell
!pip install tqdm snntorch
```
This multilayer EventProp code is modified from a single layer EventProp from the repo below, please check for more details

```
https://github.com/lolemacs/pytorch-eventprop
```
