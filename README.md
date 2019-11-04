# Free Form Deformation Network: Tensorflow 2 Implementation

## Setup

Install `tensorflow >= 2.0` by your preferred means (GPU version strongly advised).

```bash
git clone https://github.com/jackd/kblocks.git
git clone https://github.com/jackd/collection-utils.git
git clone https://github.com/jackd/shape-tfds.git
git clone https://github.com/jackd/ffd-tf2.git
pip install -e kblocks
pip install -e collection-utils
pip install -e shape-tfds
pip install -e ffd-tf2
```

## Running

CLI is handled in [kblocks](https://github.com/jackd) which is a thin wrapper around [gin-config](https://github.com/google/gin-config). If you're new to [gin](https://github.com/google/gin-config) it might be a bit intimidating, but for rapid prototyping I'm yet to find anything remotely close.

```bash
cd ffd-tf2/configs
# default synset is telephone
python -m kblocks '$KB_CONFIG/fit' base.gin
python -m kblocks '$KB_CONFIG/fit' entropy.gin --bindings='synset="plane"'
```

## Reference

If you find this code useful in your research, please cite the [following paper](https://128.84.21.199/abs/1803.10932).

```bib
@article{jack2018learning,
  title={Learning Free-Form Deformations for 3D Object Reconstruction},
  author={Jack, Dominic and Pontes, Jhony K and Sridharan, Sridha and Fookes, Clinton and Shirazi, Sareh and Maire, Frederic and Eriksson, Anders},
  journal={arXiv preprint arXiv:1803.10932},
  year={2018}
}
```

### Differences from Paper

1. Global pooling of image model features replaces 1x1 conv/flattening.

## TODO

- FFD tests
- earthmover distance
- mesh FFDs (don't forget to subdivide template meshes!)
- voxels / IoU
- documentation
