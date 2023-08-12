# GCNNet
GCNNet : 3D Point Cloud Classification &amp; Localization Using Graph-CNN
## Getting Started

### Prerequisites
   
```
conda install tensorflow-gpu==1.15.0
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
Also, please install other dependencies from the requirements beforehand. 

### KITTI Dataset

Kitti 3D object detection and birds eye view datasets were used in this work. Go to the dataset file and read the instructions provided, it is recommended to follow the file structure given there.

### Download GCNNet

Clone the repository recursively:
```
https://github.com/kageyama-shigeoo/GCNNet.git --recursive
```
## Run a checkpoint
Run an inference on the validation split:
```
bash run_on_validation.sh
```
Run an inference on the test dataset:
```
bash run_on_test.sh
```
## Training
We put training parameters in a train_config file. To start training, we need both the train_config and config.In order to start training, both the train_config and config are needed, both of these files can be found in configs. Train_configs are labeled "..train_train_config" and configs are labeled as ".._train_config".

For example:
```
python train.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config
```
Some common parameters which the reader might want to change:
```
train_dir     The directory where checkpoints and logs are stored.
train_dataset The dataset split file for training. 
NUM_GPU       The number of GPUs to use. We used two GPUs for the reference model. 
              If you want to use a single GPU, you might also need to reduce the batch size by half to save GPU memory.
              Similarly, you might want to increase the batch size if you want to utilize more GPUs. 
              Check the train.py for details.               
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

