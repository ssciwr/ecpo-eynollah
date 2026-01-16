# Training Eynollah with labeled data

## References from Eynollah

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
* Download cuDNN 8.6 (for CUDA 11.x) from [cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive) (login is needed) and copy it to the cluster
* Extract the downloaded cuDNN file
    ```bash
    tar -xf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
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

## Prepare training data

To train Eynollah models, we used labeled data obtained from LabelStudio (see [step 3 of groundtruth strategy](./groundtruth.md#step-3-exporting-from-labelstudio-to-eynollah)). *It is important to note that we also refined the annotations to ensure that all text blocks are clearly separated from one another in the ground truth.*

The original images and their corresponding label files are divided into `train` and `eval` directories using our `prepare-data` command. Due to the nature of the dataset, the images are first grouped by year and page type (ads-heavy vs. text-heavy) before being split into the `train` and `eval` folders.

```bash
prepare-data --img-dir <dir_to_org_imgs> --labeled-dir <dir_to_labeled_imgs> --train-ratio 0.8 --out-train-dir <train_dir> --out-eval-dir <eval_dir>
```

According to the instructions from Eynollah for [page segmentation](https://github.com/qurator-spk/eynollah/blob/main/docs/train.md#parameter-configuration-for-segmentation-or-enhancement-usecases), the `train` or `eval` directory should have the following structure:

```
.
└── train             # train or eval directory
   ├── images         # directory of images
   └── labels         # directory of labels
```

The `images` directory contains the original images used for training or evaluation, while the `labels` directory contains the corresponding labeled files.

These labeled files are generated from LabelStudio using our `labelstudio2png` command. In the output, each pixel is assigned to one of five classes:

```
background: 0
text: 1
image: 2
heading: 3
separator: 4
```

## Train Eynollah models

### Training config

We used the following configuration for training Eynollah models. Detailed explanations of each parameter can be found in the [Eynollah documentation](https://github.com/qurator-spk/eynollah/blob/main/docs/train.md#parameter-configuration-for-segmentation-or-enhancement-usecases).


```json
{
    "backbone_type" : "transformer",
    "task": "segmentation",
    "n_classes" : 5,
    "n_epochs" : 10,
    "input_height" : 864,
    "input_width" : 1216,
    "weight_decay" : 1e-6,
    "n_batch" : 4,
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
    "transformer_num_patches_xy": [38, 27],
    "transformer_patchsize_x": 1,
    "transformer_patchsize_y": 1,
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
    "dir_output": "/export/data/tle/eynollah/out"
}
```

In the Jingbao collection, the original image size is `4832 × 3424`, which is too large for full-image model training. Therefore, we chose a sliding window of size `1216 × 864`, ensuring that each window covers at least one text block while complying with Eynollah's input size requirements:

```
input_height = transformer_num_patches_y * transformer_patchsize_y * 32
input_weight = transformer_num_patches_x * transformer_patchsize_x * 32
```

The configuration above is used for training with scaling as augmentation.

### Training results

**Note**: We slightly modified the source code of Eynollah’s `inference` feature (specifically the `visualize_model_output()` function) to adjust the color coding to our preferences.

```python
def visualize_model_output(self, prediction, img, task):
    if task == "binarization":
        ...
    else:
        ...
        rgb_colors = {
                "0": [255, 255, 255],  # this is BGR, not RGB
                "1": [60, 76, 231],
                "2": [219, 152, 52],
                "3": [34, 126, 230],
                "4": [182, 89, 155],
                ...
        }
        ...
        layout_only = layout_only.astype(np.uint8)  # instead of int32
        img = img.astype(np.uint8)  # instead of int32

        added_image = cv2.addWeighted(img, 0.5, layout_only, 0.5, 0)  # instead of 0.5 and 0.1
```

#### Scaling only
* Run on `hgscomp01`
* GPU1
* Memory usage: 32.45 GB
* GPU-Util: max 99%
* Training time: 4:51:29
* Training loss: 0.1589
* Training accuracy: 0.9598
* Inference results: [heiBOX folder](https://heibox.uni-heidelberg.de/library/44f70fc2-27c0-41c8-8b85-63efb83f2ffe/ECPO_data/eynollah_inference_20260115/scale-only)
* Trained model: [heiBOX link](https://heibox.uni-heidelberg.de/f/b3525ec2f7384054be30/)

#### Scaling and binarization
* Run on `compgpu14`
* GPU1
* Memory usage: 32.45 GB
* GPU-Util: max 99%
* Training time: 8:16:36
* Training loss: 0.3221
* Training accuracy: 0.9491
* Inference results: [heiBOX folder](https://heibox.uni-heidelberg.de/library/44f70fc2-27c0-41c8-8b85-63efb83f2ffe/ECPO_data/eynollah_inference_20260115/scale-bin)
* Trained model: [heiBOX link](https://heibox.uni-heidelberg.de/f/e65fe27aa79d4a3ab54c/)

#### Scaling, binarization, and rotation (not 90°)
* Run on `compgpu14`
* GPU1
* Rotation : [45, 35, 25, 15, -15, -25, -35, -45]
* Memory usage: 32.45 GB
* GPU-Util: max 99%
* Training time: 11:13:31
* Training loss: 0.1495
* Training accuracy: 0.9586
* Inference results: [heiBOX folder](https://heibox.uni-heidelberg.de/library/44f70fc2-27c0-41c8-8b85-63efb83f2ffe/ECPO_data/eynollah_inference_20260115/scale-bin-rotate)
* Trained model: [heiBOX link](https://heibox.uni-heidelberg.de/f/c2d510d7de0e401aa000/)

#### Scaling, binarization, and rotation (90°)
* Run on `compgpu14`
* GPU1
* Rotation : 90°
* Memory usage: 32.45 GB
* GPU-Util: max 99%
* Training time: TBU.
* Training loss: TBU.
* Training accuracy: TBU.
* Inference results: TBU.
* Trained model: TBU.


#### Impression on current results (without 90° rotation)
* Overall image recognition quality is quite good
* Heading detection remains inconsistent (mixed of good and bad results)
* Results across different trained model variants show some differences, but they are not significant
    * The `scaling-binarization` model performs slightly better than the `scaling-only` model
    * The `scaling-binarization-rotation model (non-90° rotation)` shows some improvement in heading masking but performs worse in separating text blocks
    * `scaling-binarization-rotation (90° rotation)` - TBU.
* In general, many text blocks are not fully separated, particularly on advertisement-heavy pages compared to text-heavy pages
    * This may be related to the column-detection approach used by Eynollah?
    * Additional preprocessing steps in Eynollah’s data preparation pipeline might help improve this issue?