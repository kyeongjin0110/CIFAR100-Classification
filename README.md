# CIFAR100-Classification
#### This project is to solve a multi-class classification problem by applying various CNN models based on the CIFAR-100 dataset.

## Main description

#### EfficientNet
EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients.

## Requirements

- Ubuntu 16.04
- CUDA 9.0
- cuDNN 7.0
- Tensorflow 1.8.0

## Testing

EfficientNet
Path: ./EfficientNet

```bash
# for testing EfficientNet
python3 cifar100_efficient.py
```

SimpleCNN
Path: ./SimpleCNN

```bash
# for testing SimpleCNN
python3 cifar100_simple_cnn.py
```