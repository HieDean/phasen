this directory is used to place datasets.

```
- datasets
    - noisy_testset_wav
    - clean_testset_wav
    - noisy_trainset_wav
    - clean_trainset_wav
```

you could use `clip.py` to clip the speech file, and this operation will generate `_clip` directory.

```
- datasets
    - noisy_testset_wav
    - clean_testset_wav
    - noisy_trainset_wav
    - clean_trainset_wav
    - noisy_testset_wav_clip
    - clean_testset_wav_clip
    - noisy_trainset_wav_clip
    - clean_trainset_wav_clip
```

you could use `denoise_test.py` to denoise the `noisy_testset_wav` directory , and this operation will generate `eval` directory.

```
- datasets
    - noisy_testset_wav
    - clean_testset_wav
    - noisy_trainset_wav
    - clean_trainset_wav
    - noisy_testset_wav_clip
    - clean_testset_wav_clip
    - noisy_trainset_wav_clip
    - clean_trainset_wav_clip
    - eval
```