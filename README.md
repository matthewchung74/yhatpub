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
- Turn that predict function into a webapp, hosted at <a href="https://yhat.pub">YHat.pub</a>.
- Make the webapp so easy, a panda could use it.
<p float="center">
  <img src="https://cdn.uconnectlabs.com/wp-content/uploads/sites/46/2019/04/GitHub-Mark.png" width="150px" height="150px">
  <img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="150px" height="150px">
  <img src="https://www.csestack.org/wp-content/uploads/2019/09/Python-Decorators-Explained.png" width="240px" height="150px">
  <img src="https://media3.giphy.com/media/o7OChVtT1oqmk/200w.webp?cid=ecf05e47pd0unq8m3c9hvz1wlevvnomhb3hqyqw08w2b6cbu&rid=200w.webp&ct=g" width="200px" height="150px">    
</p>

______________________________________________________________________

## Just want to run a model already?

Choose a model from below and run it:
- [Bear Classifier](https://yhat.pub/model/0edef73f-710a-4fa1-9481-b3b394713595) with model from [fastai lesson 2](https://github.com/fastai/fastbook/blob/master/02_production.ipynb)
- [Pet Breeds Classifier](https://yhat.pub/model/0edef73f-710a-4fa1-9481-b3b394713595) with model from [fastai lesson 5](https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb)
- [Multiclassifier](https://yhat.pub/model/0edef73f-710a-4fa1-9481-b3b394713595) with model from [fastai lesson 6](https://github.com/fastai/fastbook/blob/master/06_multicat.ipynb)
- [Regression (Poses)](https://yhat.pub/model/0edef73f-710a-4fa1-9481-b3b394713595) with model from [fastai lesson 6](https://github.com/fastai/fastbook/blob/master/06_multicat.ipynb)

## How To Upload your own Model?

### Step 1: Install

Find matt_yhatpub on discord and ask to get early access. 
<br>
<br>
[![Discord](https://img.shields.io/badge/discord-chat-green.svg?logo=slack)](https://discord.gg/e37qeAGv)

Got early access and just want to run a model? Then try the <a href="https://yhat.pub/model/0edef73f-710a-4fa1-9481-b3b394713595">Bear Classifier</a> from <a href="https://github.com/fastai/fastbook/blob/master/02_production.ipynb">fastai</a>.


### Step 2: Upload your model

Train your model and upload it for public accessibility. This example uses Google Drive, but anywhere is fine.
<br>
<br>
<p float="center">
  <img src="/images/save_gdrive.gif">
</p>

### Step 3: Write your predict function in a colab.

This example uses `fastai`, but should work with any framework. The entire notebook is here <a href="https://github.com/yhatpub/yhatpub/blob/notebook/notebooks/fastai/lesson2.ipynb">Colab notebook</a>. Feel free to start a new colab notebook and follow along. 

______________________________________________________________________

The following cell installs pytorch, fastai and yhat_params, which is used to decorate your `predict` function.

```bash
!pip install -q --upgrade --no-cache-dir torch torchvision torchaudio
!pip install -q --upgrade --no-cache-dir fastai

!pip install -Uqq --no-cache-dir git+https://github.com/yhatpub/yhat_params.git@main
```

**Warning** don't place `pip installs` and `imports` in the same cell. The imports might not work correctly if done that way.

```bash
from fastai.vision.all import *
from yhat_params.yhat_tools import inference_test, FieldType, inference_predict
```
Google drive does not allow direct downloads for files over 100MB, so you'll need to follow the snippet below to get the download url. .

```bash
#file copied from google drive
google_drive_url = "https://drive.google.com/file/d/1s-fQPvk8l7CTUiiRvKzecijSluDnoZ27/view?usp=sharing"
import os
os.environ['GOOGLE_FILE_ID'] = google_drive_url.split('/')[5]
os.environ['GDRIVE_URL'] = f'https://docs.google.com/uc?export=download&id={os.environ["GOOGLE_FILE_ID"]}'
!echo "This is the Google drive download url $GDRIVE_URL"
```

`wget` it from google drive. This script places the model in a `model` folder
```bash
!wget -q --no-check-certificate $GDRIVE_URL -r -A 'uc*' -e robots=off -nd
!mkdir -p model
!mv $(ls -S uc* | head -1) ./model/export.pkl
```
verify the model exists. **Warning** YHat is pretty finicky about where you place your models. Make sure you create a `model` directory and download your model(s) there  

```bash
!ls -l ./model/export.pkl
```

The following is the equivalent of torch `torch.load` or ts `model.load_weights`

```bash
learn_inf = load_learner('./model/export.pkl')
```

And write your predict function. Note, you will need to decorate your function with <a href="https://github.com/yhatpub/yhat_params">inference_predict</a> which takes 2 parameters, a `dic` for input and output.

**Info** These parameters are how YHat.pub maps your predict functions input/output of the web interface. The `dic` key is how you access the variable and the value is it's type. You can use autocomplete to see all the input/output types and more documentation on `inference_predict` is available at the link.

```bash
input = {"image": FieldType.PIL}
output = {"text": FieldType.Text}

@inference_predict(input=input, output=output)
def predict(params):
    img = PILImage.create(np.array(params["image"].convert("RGB")))
    result = learn_inf.predict(img)
    return {"text": str(result[0])}
```

For testing, first, import `in_colab` since you only want to run this test in colab. YHat will use this colab in a callable API, so you don't want your test to run every time `predict` is called. Next, import `inference_test` which is a function to make sure your `predict` will run with YHat.

Now, inside a `in_colab` boolean, first get whatever test data you'll need, in this case, an image. Then you'll call your predict function, wrapped inside  `inference_test`, passing in the same params you defined above. If something is missing, you should see an informative error. Otherwise, you'll see something like
`Please take a look and verify the results`

```bash
from yhat_params.yhat_tools import in_colab, inference_test

if in_colab():
    import urllib.request
    from PIL import Image
    urllib.request.urlretrieve("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/GrizzlyBearJeanBeaufort.jpg/220px-GrizzlyBearJeanBeaufort.jpg", "input_image.jpg")
    img = Image.open("input_image.jpg")
    inference_test(predict_func=predict, params={'image': img})
```

If you run into errors, feel free to hop into Discord. 

Otherwise, you'll now want to clear your outputs and save a public repo on Github.

<p float="center">
  <img src="/images/save_github.gif">
</p>

### Step 4: Sign into YHat.pub and start your build.
If it doesn't work, make sure to look out for errros. Click the EXPORT button to download the logs.

<p float="center">
  <img src="/images/upload_build.gif">
</p>

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
