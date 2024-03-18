# Reproducing the text-recogniser project of <a href="https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs">fsdl</a>.
This repo aims to reproduce the text-recogniser project from fsdl. It is fairly comprehensive and provides a lot of learning
on many aspects: 
* Distributed training with PyTorch Lightning 
* Data wrangling and data synthesis of text and images
* Experiment tracking with wandb
* Deployment of model as a serverless app
* Using Gradio for frontend

* My improvements over the fsdl project:
	* removed a lot of argparse boilerplate in favour of `LightningCLI` and YAML config files.
	* I use my own ResNet, giving me more customisability, in contrast to using a fixed resnet18 architecture.
	* Instead of sin and cos positional embeds, my model learns the embeddings in a `nn.Embedding` layer. This is similar to GPT-2 training.
	* In the transformer decoder, I use `dropout=0` in line with GPT training, although it is easy to set it to other values from CLI.
	* In the fsdl repo they set the `ignore_index=self.padding_token` in the cross entropy loss, which seems to lead to worse performance, so I removed that.

## TODO
* TODO: Currently `data_config` is an arg of the `lit_transformer` constructor. Currently it is configured in the yaml file. I shoult figure out how to do sth like `--data=SomeDataModule` and then the model should get a `data_config` dynamically for that dataset. I think there was some `run=False` arg in the `LightningCLI`, maybe look into that.
* TODO: The EMNIST website broke their link <a href="http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip">http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip</a> and now it doesn't lead to the matlab.zip file, but to some html. I have the zip file locally so should figure out where to upload it so I can download it quicker and more reliably.

## Progress:
* I tested the lit transformer manually to overfit a single batch and reach zero character error rate on it. That worked. 
* The `overfit_batches=1` using the lightning Trainer seems to use a different validation batch than the training batch so you only overfit the training batch and not necessarily the validation batch. There are a bunch of issues on GH about the functionality, seems quite contraversial to some people requesting all kinds of functionalities.

## Running experiments:
> TODO: The EMNIST website broke their link <a href="http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip">http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip</a> and now it doesn't lead to the matlab.zip file, but to some html. I have the zip file locally so should figure out where to upload it so I can download it quicker and more reliably.
* First `cd` in the root of the repo (parent of `model_package`).
* Second, run `export PYTHONPATH=.` - to add the current directory to `sys.path`.
* Third, run `wandb login` to login to the wandb with your api key.
* Fourth, run `export WANDB_START_METHOD="thread"` otherwise some weird threading exception occurs. For more info see this <a href="https://github.com/wandb/wandb/issues/3223#issuecomment-1032820724">issue</a>.
* Then run what you wanna run:
	```bash
	$ python training/run_experiment.py fit --config emnistlines_experiment_config --trainer.overfit_batches=1 --trainer.max_epochs=200 --trainer.every_n_epochs=100 --data.batch_size=64
	```
