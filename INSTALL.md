### installing CUDA, cuDNN, Tensorflow with cuDNN support:

```sh
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-driver nvidia-cuda-toolkit libcupti-dev
git clone https://github.com/NVIDIA/cuda-samples.git && cd cuda-samples
```

nvcc does not work with v(gcc) > 5, and deb installation path has standard prefix:
```sh
sudo apt-get install gcc-5 g++-5
make CUDA_PATH=/usr HOST_COMPILER=g++-5 EXTRA_NVCCFLAGS="-L /usr/lib/x86_64-linux-gnu"
./bin/x86_64/linux/release/deviceQuery
```

Then follow [cuDNN installation guide](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installcuda).
Caveat: Ubuntu 18.04 is still not supported, but we can use the 16.04 deb files `libcudnn7{,-dev,-doc}`,
so register as Nvidia developer, [download cuDNN](https://developer.nvidia.com/rdp/cudnn-download).

Note: cuDNN depends on specific CUDA toolkit versions (without explicit dependency)!
```sh
sudo dpkg -i libcudnn7-dev_7.1.4.18-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev-dev_7.1.4.18-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc-dev_7.1.4.18-1+cuda9.0_amd64.deb
cp -r /usr/src/cudnn_samples_v7/mnistCUDNN/ . && cd mnistCUDNN
make CUDA_PATH=/usr HOST_COMPILER=g++-5 EXTRA_NVCCFLAGS="-L /usr/lib/x86_64-linux-gnu"
./mnistCUDNN
```

Next, install Tensorflow-GPU according to [build instructions for CUDA](https://www.tensorflow.org/install/install_sources),
but ignoring statements about `LD_LIBRARY_PATH` etc.:
```sh
...
cd tensorflow; git checkout r1.8 # or whatever
./configure
```

This must be done both for `/usr/bin/python` (if required) and `/usr/bin/python3`:
- for CUDA, enter `y`, version `9.1` and path `/usr` (will be guessed wrongly)
- for cuDNN 7 likewise, version `7.1` and path `/usr`
- Compute-capability `6.1` for _Quadro P1000_ according to https://developer.nvidia.com/cuda-gpus (will be guessed wrongly, too)
- for C compiler `/usr/bin/x86_64-linux-gnu-gcc-5` (will be guessed wrongly), 
  since `>5` does not work (but even `>5.4` does not work, see below)
- to get `/usr` working at all, `third_party/gpus/cuda_configure.bzl` must be patched!
- moreoever, there seems to be a [bug in GCC 5.5](https://github.com/tensorflow/tensorflow/issues/10220#issuecomment-352110064) (with intrinsic functions for AVX512 instruction set). With that workaround the following build does run through:
```sh
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
. env/bin/activate
pip install /tmp/tensorflow_pkg/tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl
# (or "pip install --ignore-installed --upgrade ...")
pip install pycudnn
deactivate
...
. env3/bin/activate
pip install /tmp/tensorflow_pkg/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
pip install pycudnn
deactivate
```

### from Keras FAQ: How can I obtain reproducible results using Keras during development?

```python
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
```

