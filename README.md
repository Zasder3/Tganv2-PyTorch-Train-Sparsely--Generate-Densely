# tganv2-pytorch
 PyTorch: Train Sparsely, Generate Densely: Memory-efficient Unsupervised Training of High-resolution Temporal GAN
## Usage
Simply alter the `genSamples` function that was included in `train_movingmnist.py` to write to the necessary folder for video samples and train away. If you would like to train out of the box on UCF-101, all that is required is an implementation of the dataset and the default model initializations will be identical to the paper.
## Results
With only 10k training iterations the model was successfully trained on a 2080Ti in 1 hour 30 minutes. The following sample progression was observed:
### 1st Epoch
![1st epoch result](/tganv2moving/gensamples_id0.gif)
### 5000th Epoch
![5000th epoch result](/tganv2moving/gensamples_id5000.gif)
### 9900th Epoch
![9900th epoch result](/tganv2moving/gensamples_id9900.gif)
## Limitations/Changes Made
The following implementation observed most of the practices used in the paper. Instead of [Chainer's unpool](https://docs.chainer.org/en/stable/reference/generated/chainer.functions.upsampling_2d.html) I utilized the `nn.Upsample` for increasing resolution. The loss is also slightly different to that of the paper but consistent with their implementation files (it should be noted that loss function did not visibly change any performance and the possibility to change it exists within the trainer).
While the paper quotes a $$D(x'_1, \cdots x'_N) = \sigma(\sum^N D'_n(x'_n))$$
