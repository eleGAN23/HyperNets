# Hyper
Hypercomplex Neural Networks with PyTorch: this repository would be a container for hypercomplex neural network modules to facilitate research in this topic.

## Lightweight Convolutional Neural Networks By Hypercomplex Parameterization

### [Eleonora Grassucci](https://scholar.google.it/citations?user=Jcv0TgQAAAAJ&hl=it&authuser=1), [Aston Zhang](https://www.astonzhang.com/), and [Danilo Comminiello](https://danilocomminiello.site.uniroma1.it/)

### Abstract

Hypercomplex neural networks have proved to reduce the overall number of parameters while ensuring valuable performances by leveraging the properties of Clifford algebras. Recently, hypercomplex linear layers have been further improved by involving efficient parameterized Kronecker products. In this paper, we define the parameterization of hypercomplex convolutional layers to develop lightweight and efficient large-scale convolutional models. Our method grasps the convolution rules and the filters organization directly from data without requiring a rigidly predefined domain structure to follow. The proposed approach is flexible to operate in any user-defined or tuned domain, from 1D to nD regardless of whether the algebra rules are preset.
Such a malleability allows processing multidimensional inputs in their natural domain without annexing further dimensions, as done, instead, in quaternion neural networks for 3D inputs like color images.
As a result, the proposed method operates with 1/n free parameters as regards its analog in the real domain. We demonstrate the versatility of this approach to multiple domains of application by performing experiments on various image datasets as well as audio datasets in which our method outperforms real and quaternion-valued counterparts.


### Parameterized Hypercomplex Convolutional (PHC) Layer

The core of the approach is the sum of Kronecker products which grasps the convolution rule and the filters organization directly from data. The higlights of our approach is defined in:

  ```python
  def kronecker_product1(self, a, s):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(s.shape[-4:-2]))
    siz2 = torch.Size(torch.tensor(s.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1) * s.unsqueeze(-4).unsqueeze(-6)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out
   
  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
    input = input.type(dtype=self.weight.type())      
    return F.conv2d(input, weight=self.weight, stride=self.stride, padding=self.padding)

   ```

### Usage

### Tutorials

The folder `tutorials` contain a set of tutorial to understand the Parameterized Hypercomplex Multiplication (PHM) layer and the Parameterized Hypercomplex Convolutional (PHC) layer. We develop simple toy examples to learn the matrices A that define algebra rules in order to demonstrate the effectiveness of the proposed approach.

### Experiments on Image Classification

To reproduce image classification experiments, please refer to the `image-classification` folder.

* ```pip install -r requirements.txt```.

* Choose the configurations in `configs` and run the experiment:

```python main.py --TextArgs=config_name.txt```.

The experiment will be directly tracked on [Weight&Biases](https://wandb.ai/).

### Experiments on Sound Event Detection

To reproduce sound event detection experiments, please refer to the `sound-event-detection` folder.

* ```pip install -r requirements.txt```.

We follow the instructions in the original repository for the [L3DAS21](https://github.com/l3das/L3DAS21) dataset:

* Download the dataset:

```python download_dataset.py --task Task2 --set_type train --output_path DATASETS/Task2```

```python download_dataset.py --task Task2 --set_type dev --output_path DATASETS/Task2```

* Preprocess the dataset:

```python preprocessing.py --task 2 --input_path DATASETS/Task2 --num_mics 1 --frame_len 100```

Specify `num_mics=2` and `output_phase=True` to perform experiments up to 16-channel inputs.

* Run the experiment:

```python train_baseline_task2.py```

Specify the hyperparameters options.
We perform experiments with `epochs=1000`, `batch_size=16` and `input_channels=4/8/16` on a single Tesla V100-32GB GPU. 

* Run the evaluation:

```python evaluate_baseline_task2.py```

Specify the hyperparameters options.

### More will be added

Work in progress!

### Similar reporitories

Work in progress!

### Cite
