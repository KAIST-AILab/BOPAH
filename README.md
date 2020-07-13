# Batch Reinforcement Learning with Hyperparameter Gradients

This repository is the official implementation of [Batch Reinforcement Learning with Hyperparameter Gradients](http://ailab.kaist.ac.kr/papers/LLVKK2020).

## Requirements

To install requirements:

```sh
conda env create -f environment.yml
conda activate batchrl
```

To download the batch trajectories used in the paper, please run the following:

```sh
python download_dataset.py
```

## Finite MDP experiments

To run the finite MDP experiments in the paper, run this command:

```sh
python finite_run.py
```

## Continuous control experiments

To run the Mujoco continuous control experiments in the paper, run this command:

```sh
python cont_run.py
```


## References

If this repository helps you in your academic research, you are encouraged to cite our paper. Here is an example bibtex:
```bibtex
@inproceedings{lee2020batch,
	title={Batch Reinforcement Learning with Hyperparameter Gradients},
	author={Byung-Jun Lee* and Jongmin Lee* and Peter Vrancx and Dongho Kim and Kee-Eung Kim},
	booktitle={Proceedings of the 37th International Conference on Machine Learning},
	year={2020}
}
```
