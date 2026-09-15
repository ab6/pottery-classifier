# Pottery Classifier

![Python](https://img.shields.io/badge/python-3.13-blue)
![fastai](https://img.shields.io/badge/fastai-2.8-orange)
![GitHub last commit](https://img.shields.io/github/last-commit/ab6/pottery-classifier)
![License](https://img.shields.io/badge/license-TBD-lightgrey)

An image classifier that identifies the type of a pottery piece (bowl, jug, mug, plate, saucer, or teapot) from a photo, plus a small Voilà web app for trying it on your own images.

## Project Overview

This project follows Lesson 2 of the [fast.ai *Practical Deep Learning for Coders*](https://course.fast.ai/) course and its companion book, [*Deep Learning for Coders with fastai and PyTorch*](https://github.com/fastai/fastbook). It walks the full workflow from that lesson on a new domain, handmade pottery:

1. Check and load a labeled image dataset
2. Fine-tune a pretrained ResNet-18 with fastai
3. Review errors with a confusion matrix, top losses, and the fastai image cleaner
4. Export the trained model (`export.pkl`)
5. Wrap the exported model in a simple upload-and-classify widget app that can be served with Voilà

The model predicts one of six classes: `bowl`, `jug`, `mug`, `plate`, `saucer`, `teapot`.

## Installation and Setup

### Codes and Resources Used

- **Editor:** Jupyter Notebook
- **Python version:** 3.13 (pinned in `.python-version`)
- **Environment and package manager:** [uv](https://docs.astral.sh/uv/)

Clone the repo and install the dependencies:

```bash
git clone https://github.com/ab6/pottery-classifier.git
cd pottery-classifier
uv sync
```

Open the notebooks:

```bash
uv run jupyter notebook
```

Run the web app:

```bash
uv run voila potClassWebApp.ipynb
```

> **Note:** `export.pkl` is loaded with `load_learner`, which uses Python's `pickle`. Only load model files from sources you trust.

### Python Packages Used

- **General purpose:** `pathlib` (via fastai), `ipykernel`, `jupyter`
- **Data manipulation:** fastai `DataBlock` / `DataLoaders`, `PIL` (via fastai)
- **Data visualization:** fastai `show_batch`, `ClassificationInterpretation` (matplotlib under the hood)
- **Machine learning:** `fastai` (>= 2.8.8), `fastbook` (>= 0.0.29), PyTorch / torchvision (via fastai)
- **Web app:** `ipywidgets` (via fastai/fastbook), `voila` (>= 0.5.12)

The full list, with exact versions, is in `pyproject.toml` and `uv.lock`.

## Data

### Source Data

The dataset is in `data/`, with one folder per class. The folder name is the label.

| Class   | Images |
|---------|-------:|
| bowl    | 1,771 |
| jug     | 1,671 |
| mug     | 1,380 |
| plate   | 1,298 |
| saucer  |   793 |
| teapot  | 1,152 |
| **Total** | **8,065** |

<!-- TODO: add where the images came from (source link, who took them, and any usage terms). -->
**Source:** The dataset was created by Daniel Carreira and can be found at https://github.com/daniel-carreira/painted-ceramic-dataset

### Data Preprocessing

- **Image check:** `verify_images` was run on every file. No corrupt images were found.
- **Labels:** taken from the parent folder name (`parent_label`).
- **Train/validation split:** random 80/20 split (`RandomSplitter(valid_pct=0.2, seed=42)`).
- **Resizing and augmentation (training):** `RandomResizedCrop(224, min_scale=0.5)` on each image, plus fastai's default `aug_transforms()` on each batch (flips, rotation, zoom, lighting, warp).
- **Cleaning:** `ImageClassifierCleaner` was used to review the highest-loss images for mislabeled or irrelevant pictures.

## Code structure

```
potteryClassifier/
├── data/                       # Labeled images, one subfolder per class
│   ├── bowl/
│   ├── jug/
│   ├── mug/
│   ├── plate/
│   ├── saucer/
│   └── teapot/
├── potteryClassifier.ipynb     # Data loading, training, evaluation, cleaning, export
├── potClassWebApp.ipynb        # Upload-and-classify widget app (run with Voilà)
├── export.pkl                  # Exported fastai learner used by the web app
├── main.py                     # Placeholder entry point from `uv init`
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependency versions
├── .python-version             # Python version pin (3.13)
├── .gitignore
└── README.md
```

- **`potteryClassifier.ipynb`** is the main notebook. It builds the `DataBlock`, fine-tunes the model, shows the evaluation plots, runs the cleaner, exports the model, and builds the widgets step by step.
- **`potClassWebApp.ipynb`** is the trimmed-down app. It loads `export.pkl` and shows an upload button, a **Classify** button, a thumbnail of the image, and the predicted class with its probability.

## Results and evaluation

**Model:** ResNet-18 pretrained on ImageNet, fine-tuned with `learn.fine_tune(4)` (1 frozen epoch, then 4 unfrozen epochs).
**Metric:** error rate on the 20% validation set.

| Stage | Epoch | Train loss | Valid loss | Error rate |
|-------|:-----:|-----------:|-----------:|-----------:|
| Frozen   | 0 | 1.108 | 0.448 | 14.6% |
| Unfrozen | 0 | 0.573 | 0.351 | 10.8% |
| Unfrozen | 1 | 0.440 | 0.302 | 9.7% |
| Unfrozen | 2 | 0.327 | 0.271 | 8.6% |
| Unfrozen | 3 | 0.260 | 0.268 | **8.2%** |

The final model reaches about **91.8% accuracy** on the validation set. Each unfrozen epoch took about 50 seconds on a MacBook Pro.

Validation loss was still dropping slightly at the last epoch while training loss fell faster, so a few more epochs may help a little before the model starts to overfit.

Evaluation also included:

- **Confusion matrix** (`interp.plot_confusion_matrix()`) to see which classes get mixed up
- **Top losses** (`interp.plot_top_losses(5)`) to inspect the most confident mistakes

<!-- TODO: save and embed the confusion matrix and top-losses plots, e.g. ![Confusion matrix](images/confusion_matrix.png) -->

**Sample predictions**

- A teapot image from the dataset: `teapot` with probability 0.995
- An uploaded mug photo: `mug` with probability 0.9945

## Future work

- Train longer, or try a larger backbone (ResNet-34/50, ConvNeXt), and compare error rates
- Balance the classes (saucer has the fewest images) with more data or weighted sampling
- Add classes for other forms (vases, planters, pitchers)
- Deploy the app publicly, for example on Hugging Face Spaces with Gradio
- Save the model weights in a safer format than pickle
- Move training code from the notebooks into scripts so runs are easy to repeat
- Store the images and model with Git LFS or outside the repo to keep it small

## Acknowledgments/References

- Jeremy Howard and Sylvain Gugger, [*Deep Learning for Coders with fastai and PyTorch*](https://github.com/fastai/fastbook), Chapter 2
- [fast.ai *Practical Deep Learning for Coders*](https://course.fast.ai/), Lesson 2
- [fastai library](https://docs.fast.ai/)
- README layout based on [pragyy/datascience-readme-template](https://github.com/pragyy/datascience-readme-template)

## License

_TODO: choose a license for the code (e.g., MIT) and add a `LICENSE` file._

_TODO: state the license or usage terms for the images in `data/`._
