# FLAC: Federated Learning with Autoencoder Compression and Convergence Guarantee

This is the implementation of the FLAC introduced in the paper submitted to GLOBECOM2022. 

## Abstract
Federated Learning (FL) is considered the key approach for privacy-preserving, distributed machine learning (ML) systems. However, due to the transmission of large ML models from users to the server in each iteration of FL, communication on resource-constrained networks is currently a fundamental bottleneck in FL, restricting the ML model complexity and user participation. One of the notable trends to reduce the communication cost of FL systems is gradient compression, in which techniques in the form of sparsification or quantization are utilized. However, these methods are pre-fixed and do not capture the redundant, correlated information across parameters of the ML models, user devices' data, and iterations of FL. Further, these methods do not fully take advantage of the error-correcting capability of the FL process. In this paper, we propose the Federated Learning with Autoencoder Compression (FLAC) approach that utilizes the redundant information and error-correcting capability of FL to compress user devices' models for uplink transmission. FLAC trains an autoencoder to encode and decode users' models at the server in the Training State, and then, sends the autoencoder to user devices for compressing local models for future iterations during the Compression State. To guarantee the convergence of the FL, FLAC dynamically controls the autoencoder error by switching between the Training State and Compression State to adjust its autoencoder and its compression rate based on the error tolerance of the FL system. We theoretically prove that FLAC converges for FL systems with strongly convex ML models and non-i.i.d. data distribution. Our extensive experimental results over three datasets with different network architectures show that FLAC can achieve compression rates ranging from 83x to 875x while staying near 7 percent of the accuracy of the non-compressed FL systems.

## System Model
<p float="right">
  <img src="/images/compression.jpg" width="500" title="Compression"/>
  
  <img src="/images/training.jpg" width="500" title="Training" />

</p>

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Flower, Pytorch, and Tensorboard.

```bash
pip install Flower
pip install Pytorch
```

## Usage

```bash
python server.py

```
Then, run the run-clients.sh file using you root directory and select your data set.
```bash
run-clients.sh

```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache](https://www.apache.org/legal/src-headers.html)
