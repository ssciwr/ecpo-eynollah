# Training Eynollah with labeled data

## References from Eynollah:

* [How to train an Eynollah model](https://github.com/qurator-spk/eynollah/blob/main/docs/train.md#train-a-model)
* [Install related libs and copy pre-trained model](https://github.com/qurator-spk/eynollah/tree/main/train)

## Notes on training with GPUs

Eynollah has limited support on GPU. The training process runs on single GPU only.

The following config works when running on a HPC cluster with a `conda` environment .

* `Python` 3.10
* `CUDA` 11.8
* `cuDNN` 8.6
* `TensorFlow` 2.12.0

As `cuDNN` is not available with `conda`, we have to download the file manually.

### Step-by-step installation

* Create a conda environment
    ```bash
    conda create --name eynollah-gpu python=3.10
    ```

* Activate the environment
    ```bash
    conda activate eynollah-gpu
    ```
* Install CUDA 11.8
    ```bash
    conda install -c conda-forge cudatoolkit=11.8

    # check info
    echo $CONDA_PREFIX
    # something like "your_home/anaconda3/envs/eynollah-gpu"

    ```
* Download cuDNN 8.6 from [cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive) and copy it to the cluster
* Extract the downloaded cuDNN file
    ```bash
    tar -xf cudnn-linux-x86_64-8.6.0.x_cuda11-archive.tar.xz
    ```
* Copy lib files into CUDA directory
    ```bash
    cp cudnn-*/include/cudnn*.h $CONDA_PREFIX/include/
    cp cudnn-*/lib/libcudnn* $CONDA_PREFIX/lib/
    chmod a+r $CONDA_PREFIX/include/cudnn*.h
    chmod a+r $CONDA_PREFIX/lib/libcudnn*

    # ensure linker sees cuDNN
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
    # verify cuDNN files exist
    ls $CONDA_PREFIX/lib | grep cudnn
    ```
* Install TensorFlow 2.12.0
    ```bash
    python3 -m pip install tensorflow[and-cuda]==2.12.0
    ```
* Verify TensorFlow sees cuDNN
    ```bash
    python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('CUDA:', tf.sysconfig.get_build_info()['cuda_version']); print('cuDNN:', tf.sysconfig.get_build_info()['cudnn_version']); print('GPUs:', tf.config.list_physical_devices('GPU'))"

    # expected
    # TF: 2.12.0
    # CUDA: 11.8
    # cuDNN: 8
    # GPUs [PhysicalDevice...]
    ```
* Install `eynollah` package
    ```bash
    # move to eynollah folder if needed
    cd eynollah

    pip install -e .[training]
    ```

## Training results

### First trial

#### Config
```json
{
    "backbone_type" : "transformer",
    "task": "segmentation",
    "n_classes" : 5,
    "n_epochs" : 10,
    "input_height" : 448,
    "input_width" : 896,
    "weight_decay" : 1e-6,
    "n_batch" : 10,
    "learning_rate": 1e-4,
    "patches" : true,
    "pretraining" : true,
    "augmentation" : true,
    "flip_aug" : false,
    "blur_aug" : false,
    "scaling" : true,
    "degrading": false,
    "brightening": false,
    "binarization" : false,
    "scaling_bluring" : false,
    "scaling_binarization" : false,
    "scaling_flip" : false,
    "rotation": false,
    "rotation_not_90": false,
    "transformer_num_patches_xy": [14, 7],
    "transformer_patchsize_x": 2,
    "transformer_patchsize_y": 2,
    "transformer_projection_dim": 64,
    "transformer_mlp_head_units": [128, 64],
    "transformer_layers": 8,
    "transformer_num_heads": 4,
    "transformer_cnn_first": true,
    "blur_k" : ["blur","guass","median"],
    "scales" : [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.4],
    "brightness" : [1.3, 1.5, 1.7, 2],
    "degrade_scales" : [0.2, 0.4],
    "flip_index" : [0, 1, -1],
    "thetha" : [10, -10],
    "continue_training": false,
    "index_start" : 0,
    "dir_of_start_model" : " ",
    "weighted_loss": false,
    "is_loss_soft_dice": false,
    "data_is_provided": true,
    "dir_train": "/export/data/tle/eynollah/train",
    "dir_eval": "/export/data/tle/eynollah/eval",
    "dir_output": "/export/data/tle/eynollah/out_gpu"
}
```

#### Running statistic

* GPU: compgpu12, GPU 5
* Memory usage: 17.5 GB
* GPU-Util: max 92%
* Training time: 
* Loss:
* Accuracy:

