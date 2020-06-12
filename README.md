# PHASEN

This is an unofficial implentation of [PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network](https://arxiv.org/abs/1911.04697).

### Datasets

You could download datasets from [https://datashare.is.ed.ac.uk/handle/10283/1942](https://datashare.is.ed.ac.uk/handle/10283/1942).

`./tools/resample.py` could be used to resample `.wav` file, and `./tools/clip.py` could be used to clip wave segment into 1 second. Before training model, make sure the datasets has been resampled to 16kHz and cliped into 1 second.

### Run

The model implentation is in `./phasen.py`.

You could use `./run.py` to train and test. The model parameters will be stored in directory `./model`.

Before you use any `.py` file in this repo, make sure you have changed the parameters in that file, like `epochs`, `learning_rate`, etc.

### Requirements

* Python
* Pytorch
* Librosa
* Numpy
* Soundfile

### References

[https://github.com/huyanxin/phasen](https://github.com/huyanxin/phasen)
[https://hiedean.github.io/2020/04/10/PHASEN](https://hiedean.github.io/2020/04/10/PHASEN/)
