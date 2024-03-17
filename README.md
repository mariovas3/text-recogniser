# Reproducing the text-recogniser project of <a href="https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs">fsdl</a>.
This repo aims to reproduce the text-recogniser project from fsdl. It is fairly comprehensive and provides a lot of learning
on many aspects: 
* Distributed training with PyTorch Lightning 
* Data wrangling and data synthesis of text and images
* Experiment tracking
* Deployment of model as a serverless app
* Using Gradio for frontend

## TODO
* I played around with LightningCLI, should figure out how to use YAML configs for the inputs to the lit transformer constructor.
	* Dig into Hydra.
* I tested the lit transformer manually to overfit a single batch and reach zero character error rate on it. The `overfit_batches=1` using the lightning Trainer seems to use a different validation batch than the training batch looks like weird behaviour -> should investigate.
* TODO: The EMNIST website broke their link <a href="http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip">http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip</a> and now it doesn't lead to the matlab.zip file, but to some html. I have the zip file locally so should figure out where to upload it so I can download it quicker and more reliably.

## Running experiments:
> TODO: The EMNIST website broke their link <a href="http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip">http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip</a> and now it doesn't lead to the matlab.zip file, but to some html. I have the zip file locally so should figure out where to upload it so I can download it quicker and more reliably.
* First `cd` in the root of the repo (parent of `model_package`).
* Second, run `export PYTHONPATH=.` - to add the current directory to `sys.path`.
* Then run what you wanna run.
	* e.g., `python model_package/data/emnist.py`
	* `python model_package/data/emnist_lines.py --with_start_and_end_tokens`
