Cross-Domain Echo Controller
=========================


This repository contains python/tensorflow code to reproduce the experiments presented in our paper
[Acoustic Echo Cancellation with Cross-Domain Learning](https://doi.org/10.1109/ICASSP.2019.8683517).
It uses the state-space partitioned-block-based acoustic echo controller (https://doi.org/10.1109/ICASSP.2014.6853806) 
to model the linear echo path, and a time-domain neural network to model non-linear and late echo artifacts.



Requirements
------------

The data loader uses the 'soundfile' package to read/write wavs:
```
pip install soundfile
```



Preriquisites
-------------

For training, you need pre-recorded echo samples which are separated into near-end, far-end and doubletalk wav-files.
We used the database for the 'AEC Challenge Interspeech 2021': https://www.microsoft.com/en-us/research/academic-program/acoustic-echo-cancellation-challenge-interspeech-2021/

To use your own database, set up the 'dataset_dir' variable in './loaders/aec_loader.py' accordingly.

Note that we use the doubletalk files only for testing (i.e. the provided blind test set).
For training, mix the near-end microphone signal with our own doubletalk and background noise.
This allows to filter and modify each component independently, resulting in a greater variability in the test data as discussed in the paper.
For the close-talking speaker (doubletalk), we use the WSJ0 database: https://catalog.ldc.upenn.edu/LDC93S6A
For the background noise, we use random youtube noise tracks. Alternatively, NOIZEUS may be used: https://ecs.utdallas.edu/loizou/speech/noizeus/




Training
--------

Prior to training, a cache of 10,000 randomly mixed doubletalk files is generated. This is done with:
```
cd loaders
python generate_cache.py
```
The size of the training cache can be modified by setting the variable 'train_set_length' in './loaders/generate_cache.py' accordingly.



To train the model (NAEC), use:
```
cd experiments
python tdnaec_best.py train
```



Testing
----------

To test the model on the blind test set, use:
```
cd experiments
python tdnaec_best.py test
```




Performance
-----------

To evaluate the performance of our model in terms of P.808 Mean Opinion Score (MOS) using the script 'decmos.py', 
which is provided at https://github.com/microsoft/AEC-Challenge
And in terms of the ERLE, as shown in the paper.

ERLE (far-end) 43.65dB
MOS (averaged) 4.04



Citation
--------

Please cite our work as 

```
@INPROCEEDINGS{pfeifenberger2021cdec,
  author={L. {Pfeifenberger} and M. {Zöhrer} and F. {Pernkopf}},
  booktitle={Interspeech}, 
  title={Acoustic Echo Cancellation with Cross-Domain Learning}, 
  year={2021},
  volume={},
  number={},
  pages={},
}

```


