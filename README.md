# Rewrite of the text-recogniser project of <a href="https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs">fsdl</a>.
*Given image input with handwritten text, infer the text and output the associated string.*

This repo aims to reproduce the text-recogniser project from fsdl while adding a few improvements over model architecture as well as practical "hacks" to reduce the cost (**monetary** and **tech debt**) of the project.

The project itself is fairly comprehensive and offers a lot of learning:
* Distributed training with **PyTorch Lightning**
* Data wrangling and data synthesis using the <a href="https://fki.tic.heia-fr.ch/databases/iam-handwriting-database">IAM</a> and <a href="https://www.nist.gov/itl/products-and-services/emnist-dataset">EMNIST</a> datasets.
* Experiment tracking with **wandb**
* Deployment of model as a serverless app with **AWS Lambda**
* Using **Gradio** for hosting a quick MVP app

## My improvements over the fsdl project:
* Removed a lot of argparse boilerplate in favour of `LightningCLI` and YAML config files.
* I use my own ResNet, giving me more customisability, in contrast to using a fixed `resnet18` architecture.
* Instead of 2D $\text{sin}$ and $\text{cos}$ positional embedding, my model learns the positional embeddings. This is similar to GPT-2 training.
* In the transformer decoder, I use `dropout=0` in line with GPT training, although it is easy to set it to other values from the CLI `--model.tf_dropout=DROPOUT`.
* In the fsdl repo they set the `ignore_index=self.padding_token` in the cross entropy loss, which seems to lead to worse performance, so I removed that.
* In fsdl, if the data are not available in the local file system, they are downloaded from `S3`. Hosting data on `S3` costs money, so I set up my google drive API and host my data on my personal google drive which is zero-cost.
	* Since this project is intended to be interacted with either via the Gradio app (if lambda still running) or as a POC, the google drive api is not publicly accessible and I configured authentication to let me interact with it.
	* The credentials for the google drive api should live in the root directory as `token.json` or if the token has expired, flow credentials in the same directory under the name of `google_drive_credentials.json` should exist.
	* The google drive download function is available in the <a href="./model_package/project_utils.py">project utils file</a>.

## TODO:

## Maintainer notes:
* If bumping the version of python support to 3.12 and above, I could swap the use of `boltons.cacheutils.cachedproperty` for `functools.cached_property`. Currently I'm aiming to support python 3.11, which still uses the old version of `functools.cached_property` which has <a href="https://discuss.python.org/t/finding-a-path-forward-for-functools-cached-property/23757">tricky locking</a> in multithreaded apps, so will use the `boltons` version for now.
* For parsing `xml`, I currently use <a href="https://pypi.org/project/defusedxml">defusedxml</a> since the standard lib can be vulnerable to some bad `xml`.

## Progress:
* I tested the lit transformer manually to overfit a single batch and reach zero character error rate on it. That worked.
* The `overfit_batches=1` using the lightning Trainer seems to use a different validation batch than the training batch so you only overfit the training batch and not necessarily the validation batch. There are a bunch of issues on GH about the functionality, seems quite contraversial to some people requesting all kinds of functionalities.
* Managed to enforce equal arg vals for `data.max_length` and `model.max_seq_length` via `link_arguments` of `jsonargparse` after subclassing the `LightningCLI` (equality enforced upon parse). Similarly, linked the model's `input_dims` to the data's `input_dims`. Upon instantiation of `data`, the `data.input_dims` attribute is set and then used as input to the constructor of the model.
* Made the `ModelCheckpoint` callback configurable in the `training/run_experiment.py` script via the `my_model_checkpoint` name in the cli (by subclassing the `LightningCLI`).

## Running experiments:
* First `cd` in the root of the repo (parent of `model_package`).
* Second, run `export PYTHONPATH=.` - to add the current directory to `sys.path`.
* Third, run `wandb login` to login to the wandb with your api key.
* Fourth, run `export WANDB_START_METHOD="thread"` otherwise some weird threading exception occurs. For more info see this <a href="https://github.com/wandb/wandb/issues/3223#issuecomment-1032820724">issue</a>.
* Then run what you wanna run:
	```bash
	$ python training/run_experiment.py fit --config emnistlines_experiment_config.yaml --trainer.overfit_batches=1 --trainer.max_epochs=200 --trainer.check_val_every_n_epoch=50 --data.batch_size=64
	```
* To e.g., continue the training for another 200 epochs, just set the `--trainer.max_epochs=400` and provide a `--ckpt_path` value like so:
	```bash
	$ python training/run_experiment.py fit --config emnistlines_experiment_config.yaml --trainer.overfit_batches=1 --trainer.max_epochs=400 --trainer.check_val_every_n_epoch=50 --data.batch_size=64 --ckpt_path='PathToCkpt'
	```
* To run the `test` subcommand, do:
	```bash
	python training/run_experiment.py test --config emnistlines_experiment_config.yaml --data.batch_size=64 --ckpt_path='PathToCkpt'
	```

## Google Drive API setup (for hosting data):
* Follow instructions <a href="https://developers.google.com/drive/api/quickstart/python">here</a>.
* You download the object as `io.BytesIO` and then call the `getvalue()` on it and save it as binary file with `open(file_path, 'wb')`.
* You download a file from google drive based on the file id which can be found by copying the sharable link of the file in the google drive gui and the long, weird sequence of chars in the link is the file id.
* The relevant google drive id and the name of the file after being saved must be made available e.g., in `root/data/raw/iam/metadata.toml`.
