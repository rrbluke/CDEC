# CDEC
Cross-Domain Echo Controller
=========================


This repository contains python/tensorflow code to reproduce the experiments presented in our paper
[Acoustic Echo Cancellation with Cross-Domain Learning](https://doi.org/10.1109/ICASSP.2019.8683517).
It is based on the state-space partitioned-block-based acoustic echo controller (https://doi.org/10.1109/ICASSP.2014.6853806),
and a tome-domain neural network to remove non-linear and residual echo artifacts.



Requirements
------------

The data loader uses the 'soundfile' package to read/write wavs:
```
pip install soundfile
```


Preriquisites
-------------

We use the training data provided for the Acoustic Echo Cancellation Challenge of the Interspeech 2021: 
https://www.microsoft.com/en-us/research/academic-program/acoustic-echo-cancellation-challenge-interspeech-2021/, which contains near-end, far-end and doubletalk wav-files. 

For training, we only use the separated far-end and near-end echo files. 
We generate doubletalk by mixing the near-end echo with a desired speech signal from the WSJ0 database: https://catalog.ldc.upenn.edu/LDC93S6A
Further, we add background noise from various youotube sources or the NOIZEUS database: https://ecs.utdallas.edu/loizou/speech/noizeus/
This allows to freely mix, shift and filter the individual signal components, as discussed in the paper.
To use your own databases, you need to change the corresponding paths in './loaders/generate_cache.py' and './loaders/aec_loader.py'



Prior to training, you need to create a cache which will perform the linear AEC on 10,000 randomly selected mixtures. This is done with:
```
cd loaders
python generate_cache.py
```

To change the cache size, the variable 'self.train_set_length = 10000' in './loaders/generate_cache.py' needs to be changed accordingly.




Training
--------

To train the CDEC model, use:
```
cd experiments
python tdnaec_best.py train
```




Test
----------

To test the model on the blind test set, use:
```
cd experiments
python tdnaec_best.py test
```




Performance
-----------

The performance of the CDEC is evaluated using the script 'decmos.py' which is provided at https://github.com/microsoft/AEC-Challenge
It provides the P.808 Mean Opinion Score (MOS) for the following cases

| single-talk near-end  | single-talk far-end | doubletalk echo | doubletalk other | average |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 4.01 | 4.52 | 3.90 | 3.72 | 4.04 |

The Echo Return Loss Enhancement (ERLE) for the single-talk far-end case is 43.65 dB





Citation
--------

Please cite our work as 

```
@INPROCEEDINGS{8683517,
  author={L. {Pfeifenberger} and M. {Zöhrer} and F. {Pernkopf}},
  booktitle={Interspeech}, 
  title={Acoustic Echo Cancellation with Cross-Domain Learning}, 
  year={2021},
  volume={},
  number={},
  pages={},
}```

