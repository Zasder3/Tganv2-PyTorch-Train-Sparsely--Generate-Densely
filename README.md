# tganv2-pytorch
PyTorch: Train Sparsely, Generate Densely: Memory-efficient Unsupervised Training of High-resolution Temporal GAN
![Model Overview](https://github.com/pfnet-research/tgan2/raw/master/images/architecture.jpg)
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

While the paper quotes a ![Discriminator prediction fn](http://latex2png.com/pngs/a085db829b7a3d4cf80642478737df20.png). The current github implementation supposes a ![Discriminator alt fn](http://latex2png.com/pngs/2973ff42d02806e2bc7a7f80ed57363b.png) phi being a softplus.

## Sources and Thanks
The authors of the paper were very kind and helpful in my implementation port to PyTorch. The paper may be found at: [Train Sparsely, Generate Densely: Memory-efficient Unsupervised Training of High-resolution Temporal GAN](https://arxiv.org/abs/1811.09245)

Don't forget to cite them with:
```
@journal{TGAN2020,
    author = {Saito, Masaki and Saito, Shunta and Koyama, Masanori and Kobayashi, Sosuke},
    title = {Train Sparsely, Generate Densely: Memory-efficient Unsupervised Training of High-resolution Temporal GAN},
    booktitle = {International Journal of Computer Vision},
    year = {2020},
    month = may,
    doi = {10.1007/s11263-020-01333-y},
    url = {https://doi.org/10.1007/s11263-020-01333-y},
}
```
