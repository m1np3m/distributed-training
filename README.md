# Introduction

Why we need DT ?. The best way to describe it is the way to address the not-fit-in-ram problem as it’s mechanism is to train multi models (workers) on shared dataset.

# **Choose the right strategy**

These are two common ways of distributing training with data parallelism:

- *Synchronous training*, where the steps of training are synced across the workers and replicas, such as [`tf.distribute.MirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy), [`tf.distribute.TPUStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/TPUStrategy), and [`tf.distribute.MultiWorkerMirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy). All workers train over different slices of input data in sync, and aggregating gradients at each step.
- *Asynchronous training*, where the training steps are not strictly synced, such as [`tf.distribute.experimental.ParameterServerStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy). All workers are independently training over the input data and updating variables asynchronously.

# Prerequisites
1. You need to setup a cluster, if you want to use at local then you could go for `minikube` and `Kind`.
2. Create `kubeflow` namespace, and deploy `training operator`.
3. Install training operator: ```kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0" -n kubeflow```
4. Python SDK for Kubeflow Training Operator: ```pip install kubeflow-training```


# Install
python3 -m pip install -r requirments.txt

# Run
python3 file.py