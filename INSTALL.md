### installing CUDA, cuDNN, Tensorflow with cuDNN support:

(as per [CUDA installation instructions](https://developer.nvidia.com/cuda-downloads):)

Download and install CUDA driver and toolkit:

```sh
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.168-418.67_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-*/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
export PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}
cuda-install-samples-10.1.sh cuda-samples/ && cd cuda-samples && make
```

However, beware that this method installs CUDA toolkit into `/usr/local/cuda`, while the other Nvidia repos use `/usr`, which does not work when building Tensorflow (see below)!

Next, install [NCCL](https://developer.nvidia.com/nccl/nccl-download):

```sh
sudo dpkg -i nccl-repo-ubuntu1804-2.4.7-ga-cuda10.1_1-1_amd64.deb
sudo apt install libnccl2=2.4.7-1+cuda10.1 libnccl-dev=2.4.7-1+cuda10.1
# compensate for NCCL in /usr instead of /usr/local/cuda (like CUDA toolkit):
sudo tar --strip-components=1 -C /usr/local/cuda/targets/x86_64-linux/ -xvf nccl_2.4.7-1+cuda10.1_x86_64.txz
```

Then follow [cuDNN installation guide](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installcuda)
to install `libcudnn7{,-dev,-doc}`, so register as Nvidia developer, [download cuDNN](https://developer.nvidia.com/rdp/cudnn-download).

Note: cuDNN depends on specific CUDA toolkit versions (without explicit dependency)!
```sh
sudo dpkg -i libcudnn7_7.6.0.64-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.0.64-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-doc_7.6.0.64-1+cuda10.1_amd64.deb
cp -r /usr/src/cudnn_samples_v7 . && cd cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
# compensate for cuDNN in /usr instead of /usr/local/cuda (like CUDA toolkit):
sudo tar -C /usr/local/ -zxvf cudnn-10.1-linux-x64-v7.6.0.64.tgz
```

Next, install Tensorflow-GPU according to [build instructions for CUDA](https://www.tensorflow.org/install/install_sources),
but ignoring statements about `LD_LIBRARY_PATH` etc.:
```sh
# virtualenv...
# python requirements:
pip install -U pip six numpy wheel setuptools mock
pip install -U keras_applications==1.0.6 --no-deps
pip install -U keras_preprocessing==1.0.5 --no-deps
pip install -U protobuf # not mentioned in build instructions, only Python 2
pip install -U enum34 # not mentioned in build instructions, only Python 2
cd tensorflow; git checkout r1.13 # or whatever
./configure
```

This must be done both for `/usr/bin/python` (if required) and `/usr/bin/python3`:
- for CUDA, enter `y`, version `10.1` (guessed wrongly) and path `/usr/local/cuda`
- for cuDNN 7 likewise, version `7.6.0` and path `/usr/local/cuda`
- for NCCL, enter version `2.4.7`
- Compute-capability `6.1` for _Quadro P1000_ according to https://developer.nvidia.com/cuda-gpus
- for C compiler, keep default `/usr/bin/gcc`
- to get `/usr/local/cuda` prefix from CUDA Toolkit repo for Ubuntu mixed with `/usr` prefix from cuDNN repo for Ubuntu working at all with Tensorflow build, the following symlinks have to be created:
```sh
# compensate for missing symlinks for minor version:
for file in /usr/local/cuda/lib64/*.so.10; do sudo ln -rs $file ${file}.1; done
# compensate for libcublas in /usr instead of /usr/local/cuda (like CUDA toolkit):
sudo ln -s $(dpkg-query -L libcublas-dev | fgrep /include) /usr/local/cuda/include
sudo ln -s $(dpkg-query -L libcublas10 | fgrep .so.) /usr/local/cuda/lib64
# same for NCCL (if not using the tgz above):
sudo ln -s $(dpkg-query -L libnccl-dev | fgrep /include) /usr/local/cuda/include
sudo ln -s $(dpkg-query -L libnccl2 | fgrep .so.) /usr/local/cuda/lib64
# same for cuDNN (if not using the tgz above):
sudo ln -s $(dpkg-query -L libcudnn7 | fgrep .so.) /usr/local/cuda/lib64
```

With these workarounds the following build does run through:
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

