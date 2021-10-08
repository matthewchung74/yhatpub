<div align="center">

<img src="images/logo_dark.png" width="400px">

**The lightweight app to run your AI models. Use this app, so you don't have to build your own.**
<br>
<p align="center">
  <a href="https://yhat.pub/">Website</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">Quick Start</a> •
  <a href="#examples">Fast.ai Examples</a>
</p>

[![Discord](https://img.shields.io/badge/discord-chat-green.svg?logo=slack)](https://discord.gg/e37qeAGv)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
  
</div>

###### \*Codecov is > 20%+ but will be getting better with time :)

______________________________________________________________________

## YHat.pub Design Philosophy

YHat.pub uses these principles:

- Familiar tools (Colab, Github).
- Minimal code. (Write a predict function with python decorators)
- Turn that predict function into a webapp.
- Make that webapp so easy, a panda could use it.
<p float="center">
  <img src="https://cdn.uconnectlabs.com/wp-content/uploads/sites/46/2019/04/GitHub-Mark.png" width="150px" height="150px">
  <img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="150px" height="150px">
  <img src="https://www.csestack.org/wp-content/uploads/2019/09/Python-Decorators-Explained.png" width="240px" height="150px">
  <img src="https://media3.giphy.com/media/o7OChVtT1oqmk/200w.webp?cid=ecf05e47pd0unq8m3c9hvz1wlevvnomhb3hqyqw08w2b6cbu&rid=200w.webp&ct=g" width="200px" height="150px">    
</p>

______________________________________________________________________

## How To Use

### Step 1: Install

Find matt_yhatpub on discord and ask to get early access. 
<br>
<br>
[![Discord](https://img.shields.io/badge/discord-chat-green.svg?logo=slack)](https://discord.gg/e37qeAGv)


### Step 2: Upload your model

Train your model and upload it for public accessibility. This example uses Google Drive, but anywhere is fine.
<br>
<br>
<p float="center">
  <img src="/images/save_gdrive.gif">
</p>

### Step 3: Write your predict function in a colab.

This example uses `fastai`, but should work with any framework. The entire notebook is here <a href="https://github.com/yhatpub/yhatpub/blob/notebook/notebooks/fastai/lesson2.ipynb">Colab notebook</a>. Feel free to start a new colab notebook and follow along.

```bash
#This example installs pytorch, fastai and yhat_params, which is used to decorate your `predict` function.
!pip install -q --upgrade --no-cache-dir torch torchvision torchaudio
!pip install -q --upgrade --no-cache-dir fastai

!pip install -Uqq --no-cache-dir git+https://github.com/yhatpub/yhat_params.git@main
```

<!-- following section will be skipped from PyPI description -->

<details>
  <summary>Other installation options</summary>
    <!-- following section will be skipped from PyPI description -->

#### Install with optional dependencies

```bash
pip install pytorch-lightning['extra']
```

#### Conda

```bash
conda install pytorch-lightning -c conda-forge
```

#### Install stable 1.4.x

the actual status of 1.4 \[stable\] is following:

![CI basic testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20basic%20testing/badge.svg?branch=release%2F1.4.x&event=push)
![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=release%2F1.4.x&event=push)
![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=release%2F1.4.x&event=push)
![TPU tests](https://github.com/PyTorchLightning/pytorch-lightning/workflows/TPU%20tests/badge.svg?branch=release%2F1.4.x&event=push)
![Docs check](https://github.com/PyTorchLightning/pytorch-lightning/workflows/Docs%20check/badge.svg?branch=release%2F1.4.x&event=push)

Install future release from the source

```bash
pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@release/1.4.x --upgrade
```

#### Install bleeding-edge - future 1.5

Install nightly from the source (no guarantees)

```bash
pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip
```

or from testing PyPI

```bash
pip install -iU https://test.pypi.org/simple/ pytorch-lightning
```

</details>
<!-- end skipping PyPI description -->

### Step 1: Add these imports

```python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
```

### Step 2: Define a LightningModule (nn.Module subclass)

A LightningModule defines a full *system* (ie: a GAN, autoencoder, BERT or a simple Image Classifier).

```python
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

**Note: Training_step defines the training loop. Forward defines how the LightningModule behaves during inference/prediction.**

### Step 3: Train!

```python
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

autoencoder = LitAutoEncoder()
trainer = pl.Trainer()
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
```

## Advanced features

Lightning has over [40+ advanced features](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags) designed for professional AI research at scale.

Here are some examples:

<div align="center">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/features_2.jpg" max-height="600px">
</div>

<details>
  <summary>Highlighted feature code snippets</summary>

```python
# 8 GPUs
# no code changes needed
trainer = Trainer(max_epochs=1, gpus=8)

# 256 GPUs
trainer = Trainer(max_epochs=1, gpus=8, num_nodes=32)
```

<summary>Train on TPUs without code changes</summary>

```python
# no code changes needed
trainer = Trainer(tpu_cores=8)
```

<summary>16-bit precision</summary>

```python
# no code changes needed
trainer = Trainer(precision=16)
```

<summary>Experiment managers</summary>

```python
from pytorch_lightning import loggers

# tensorboard
trainer = Trainer(logger=TensorBoardLogger("logs/"))

# weights and biases
trainer = Trainer(logger=loggers.WandbLogger())

# comet
trainer = Trainer(logger=loggers.CometLogger())

# mlflow
trainer = Trainer(logger=loggers.MLFlowLogger())

# neptune
trainer = Trainer(logger=loggers.NeptuneLogger())

# ... and dozens more
```

<summary>EarlyStopping</summary>

```python
es = EarlyStopping(monitor="val_loss")
trainer = Trainer(callbacks=[es])
```

<summary>Checkpointing</summary>

```python
checkpointing = ModelCheckpoint(monitor="val_loss")
trainer = Trainer(callbacks=[checkpointing])
```

<summary>Export to torchscript (JIT) (production use)</summary>

```python
# torchscript
autoencoder = LitAutoEncoder()
torch.jit.save(autoencoder.to_torchscript(), "model.pt")
```

<summary>Export to ONNX (production use)</summary>

```python
# onnx
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmpfile:
    autoencoder = LitAutoEncoder()
    input_sample = torch.randn((1, 64))
    autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
    os.path.isfile(tmpfile.name)
```

</details>

### Pro-level control of training loops (advanced users)

For complex/professional level work, you have optional full control of the training loop and optimizers.

```python
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # access your optimizers with use_pl_optimizer=False. Default is True
        opt_a, opt_b = self.optimizers(use_pl_optimizer=True)

        loss_a = ...
        self.manual_backward(loss_a, opt_a)
        opt_a.step()
        opt_a.zero_grad()

        loss_b = ...
        self.manual_backward(loss_b, opt_b, retain_graph=True)
        self.manual_backward(loss_b, opt_b)
        opt_b.step()
        opt_b.zero_grad()
```

______________________________________________________________________

## Advantages over unstructured PyTorch

- Models become hardware agnostic
- Code is clear to read because engineering code is abstracted away
- Easier to reproduce
- Make fewer mistakes because lightning handles the tricky engineering
- Keeps all the flexibility (LightningModules are still PyTorch modules), but removes a ton of boilerplate
- Lightning has dozens of integrations with popular machine learning tools.
- [Tested rigorously with every new PR](https://github.com/PyTorchLightning/pytorch-lightning/tree/master/tests). We test every combination of PyTorch and Python supported versions, every OS, multi GPUs and even TPUs.
- Minimal running speed overhead (about 300 ms per epoch compared with pure PyTorch).

______________________________________________________________________

## Examples

###### Hello world

- [MNIST hello world](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html)

###### Contrastive Learning

- [BYOL](https://lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#byol)
- [CPC v2](https://lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#cpc-v2)
- [Moco v2](https://lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#moco-v2)
- [SIMCLR](https://lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#simclr)

###### NLP

- [GPT-2](https://lightning-bolts.readthedocs.io/en/latest/convolutional.html#gpt-2)
- [BERT](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/text-transformers.html)

###### Reinforcement Learning

- [DQN](https://lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html#dqn-models)
- [Dueling-DQN](https://lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html#dueling-dqn)
- [Reinforce](https://lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html#reinforce)

###### Vision

- [GAN](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/basic-gan.html)

###### Classic ML

- [Logistic Regression](https://lightning-bolts.readthedocs.io/en/latest/classic_ml.html#logistic-regression)
- [Linear Regression](https://lightning-bolts.readthedocs.io/en/latest/classic_ml.html#linear-regression)

______________________________________________________________________

## Community

The lightning community is maintained by

- [10+ core contributors](https://pytorch-lightning.readthedocs.io/en/latest/governance.html) who are all a mix of professional engineers, Research Scientists, and Ph.D. students from top AI labs.
- 480+ active community contributors.

Want to help us build Lightning and reduce boilerplate for thousands of researchers? [Learn how to make your first contribution here](https://devblog.pytorchlightning.ai/quick-contribution-guide-86d977171b3a)

Lightning is also part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/) which requires projects to have solid testing, documentation and support.

### Asking for help

If you have any questions please:

1. [Read the docs](https://pytorch-lightning.rtfd.io/en/latest).
1. [Search through existing Discussions](https://github.com/PyTorchLightning/pytorch-lightning/discussions), or [add a new question](https://github.com/PyTorchLightning/pytorch-lightning/discussions/new)
1. [Join our slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ).

### Funding

[We're venture funded](https://techcrunch.com/2020/10/08/grid-ai-raises-18-6m-series-a-to-help-ai-researchers-and-engineers-bring-their-models-to-production/) to make sure we can provide around the clock support, hire a full-time staff, attend conferences, and move faster through implementing features you request.

______________________________________________________________________

## Grid AI

Grid AI is our platform for training models at scale on the cloud!

**Sign up for our FREE community Tier [here](https://www.grid.ai/pricing/)**

To use grid, take your regular command:

```
python my_model.py --learning_rate 1e-6 --layers 2 --gpus 4
```

And change it to use the grid train command:

```
grid train --grid_gpus 4 my_model.py --learning_rate 'uniform(1e-6, 1e-1, 20)' --layers '[2, 4, 8, 16]'
```

The above command will launch (20 * 4) experiments each running on 4 GPUs (320 GPUs!) - by making ZERO changes to
your code.
