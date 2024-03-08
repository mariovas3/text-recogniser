# Reproducing the text-recogniser project of <a href="https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs">fsdl</a>.
This repo aims to reproduce the text-recogniser project from fsdl. It is fairly comprehensive and provides a lot of learning
on many aspects: 
* Distributed training with PyTorch Lightning 
* Data wrangling and data synthesis of text and images
* Experiment tracking
* Deployment of model as a serverless app
* Using Gradio for frontend

## Running experiments:
* First `cd` in the root of the repo (parent of `model_package`).
* Second, run `export PYTHONPATH=.` - to add the current directory to `sys.path`.
* Then run what you wanna run.
	* e.g., `python model_package/data/emnist.py`
	* `python model_package/data/emnist_lines.py --with_start_and_end_tokens`
