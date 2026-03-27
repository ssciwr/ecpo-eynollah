# Training Eynollah with labeled data

## Jingbao collection

### Data with ground truth

There are 100 labeled images in total, extracted from 3 years:
* 1920/04
* 1930/04
* 1939/04

However, there are 18 images with the same file names of the others, even though their annotations are slightly different. To make it simple, we overwrote the duplicated files with the same file names, which resulted in 82 unique labeled images. These 82 images are used for training and evaluation.

Depending on the year and page type (ads-heavy vs. text-heavy), the images are groupped into categories before spliting into `train` and `eval` folders with a specified ratio (e.g. 80% for training and 20% for evaluation).

In detail, the characteristics of images (with ground truth) in the Jingbao collection are as follows:

```
Total labeled no-dup files: 82
Total groups: 8
Group sizes (without dup files):
('1939', '6-7'): 10
('1939', '5-8'): 11
('1939', '2-3'): 11
('1939', '1-4'): 10
('1920', '1-4'): 10
('1920', '2-3'): 10
('1930', '2-3'): 10
('1930', '1-4'): 10
```

* 1920/04: 20 images (10 ads-heavy, 10 text-heavy)
    + no text in gutter area
    + page 0001to0004: ads-heavy with several big font headings and some pictures
    + page 0002to0003: text-heavy, up to 1 image

* 1930/04: 20 images (10 ads-heavy, 20 text-heavy)
    + ads in gutter area for both ads-heavy and text-heavy pages
        + heavy-text ads for ads-heavy pages
        + light-to-simple-text ads for text-heavy pages
    + page 0001to0004: ads-heavy with several big font headings and some pictures
    + page 0002to0003: text-heavy, 5 to 7 images
* 1939/04: 40 images with no clear distinction between ads-heavy and text-heavy pages
    + all pages have text-heavy parts and ads in the gutter area. Text in the gutter area is also heavy.
    + page 0001to0004: ads in top quarter of page 0001 and bottom quarter of page 0002, while the rest of the pages are text-heavy with some images and headings
    + page 0002to0003: ads in the bottom quarter (or third) of both pages, while the rest of the pages are text-heavy with some headings, no images
    + page 0005to0008: ads appear randomly, occupying around 50% of the page area, while the rest of the pages are text-heavy with some images and headings
    + page 0006to0007: ads in the bottom quarter (or third) of both pages, while the rest of the pages are text-heavy with some images and headings

### Data without ground truth

I only inspected 46 Jingbao images without ground truth so far.

* Same attributes as 1930/04: 1919/04, 1920/01, 1921/02, 1922/02, 1923/02, 1924/01, 1925/01, 1926/01, 1927/01, 1928/01, 1929/01, 1930/01-02, 1931/01, 1932/01, 1933/01, 1934/02, 1935/01, 1936/05, 1937/01
* Same attributes as 1939/04: 1938/02, 1939/01, 1940/01

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

These labeled files are generated from LabelStudio using our `labelstudio2png` command. In the output, each pixel is assigned to one of five classes:

```
background: 0
text: 1
image: 2
heading: 3
separator: 4
```

The original images and their corresponding labeled files are divided into `train` and `eval` directories using our `prepare-data` command. Due to the nature of the dataset, the images are first grouped by year and page type (ads-heavy vs. text-heavy) before being split into the `train` and `eval` folders.

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

#### Scaling only (with pre-train option)
* Run on `hgscomp01`
* GPU1
* Memory usage: 32.45 GB
* GPU-Util: max 99%
* Training time: 4:51:29
* Training loss: 0.1589
* Training accuracy: 0.9598
* Inference results: [heiBOX folder](https://heibox.uni-heidelberg.de/d/91c8c83ebb334003b21c/)
* Trained model: [heiBOX link](https://heibox.uni-heidelberg.de/f/b3525ec2f7384054be30/)

#### Scaling and binarization (with pre-train option)
* Run on `compgpu14`
* GPU1
* Memory usage: 32.45 GB
* GPU-Util: max 99%
* Training time: 8:16:36
* Training loss: 0.3221
* Training accuracy: 0.9491
* Inference results: [heiBOX folder](https://heibox.uni-heidelberg.de/d/1707b288158449529659/)
* Trained model: [heiBOX link](https://heibox.uni-heidelberg.de/f/e65fe27aa79d4a3ab54c/)

#### Scaling, binarization, and rotation (not 90°) (with pre-train option)
* Run on `compgpu14`
* GPU1
* Rotation : [45, 35, 25, 15, -15, -25, -35, -45]
* Memory usage: 32.45 GB
* GPU-Util: max 99%
* Training time: 11:13:31
* Training loss: 0.1495
* Training accuracy: 0.9586
* Inference results: [heiBOX folder](https://heibox.uni-heidelberg.de/d/6ce6f074eaa84591bb10/)
* Trained model: [heiBOX link](https://heibox.uni-heidelberg.de/f/c2d510d7de0e401aa000/)

#### Scaling, binarization, and rotation (90°) (with pre-train option)
* Run on `compgpu14`
* GPU1
* Rotation : 90°
* Memory usage: 32.45 GB
* GPU-Util: max 99%
* Training time: 9:04:56
* Training loss: 0.1238
* Training accuracy: 0.9599
* Inference results: [heiBOX folder](https://heibox.uni-heidelberg.de/d/abdef92df89748958c03/)
* Trained model: [heiBOX link](https://heibox.uni-heidelberg.de/f/35e907b0012c47259374/)

#### Scaling and binarization (without pre-train option)
* Run on `compgpu14`
* GPU1
* Memory usage: 32.45 GB
* GPU-Util: max 99%
* Training time: 8:50:29
* Training loss: 0.0556
* Training accuracy: 0.9788
* Inference results: [heiBOX folder](https://heibox.uni-heidelberg.de/d/59a90f62088a4c34879d/)
* Trained model: [heiBOX link](https://heibox.uni-heidelberg.de/f/d2ba5e27e74c4b8fb628/)


#### Impression on current results
* Overall image recognition quality is quite good
* Heading detection remains inconsistent (mixed of good and bad results)
* Results across different trained model variants show some differences, but they are not significant
    * The `scaling-binarization` model performs slightly better than the `scaling-only` model
    * The `scaling-binarization-rotation (non-90° rotation)` model shows some improvement in heading masking but performs worse in separating text blocks
    * The `scaling-binarization-rotation (90° rotation)` model seems less effective for image detection than the `scaling-binarization` model
* In general, many text blocks are not fully separated, particularly on advertisement-heavy pages compared to text-heavy pages
    * This may be related to the column-detection approach used by Eynollah?
    * Additional preprocessing steps in Eynollah’s data preparation pipeline might help improve this issue?


#### Fine-tuning vs. training from scratch (with scaling and binarization)
* The inference results of these two models are quite similar.
* For the Jingbao test set
    - The model trained from scratch does not outperform the fine-tuned model
    - In some cases, the trained-from-scratch model separates text blocks better, while in other cases the fine-tuned model performs better. This behavior is also observed for images and headings.
    - One noticeable difference is that the trained-from-scratch model produces more noise outside the main frame area compared to the fine-tuned model.
* For non-Jingbao sample images
    - Both the trained-from-scratch and fine-tuned models show limited performance, with text blocks not being separated correctly.


### Training with artificial boundaries, 10 pixels buffering

#### Scaling and binarization (with pre-train option)
* Run on `compgpu9`
* GPU5
* Memory usage: 32.45 GB
* GPU-Util: max 99% (but not often. I'm wondering if the training process was not fully using the GPU)
* Training time: 1 day, 22:06:42 (don't know why it took much longer than before)
* Training loss: 0.0532
* Training accuracy: 0.9795
* Inference results: [heiBOX folder](https://heibox.uni-heidelberg.de/d/32a2f27a832843eb8484/)
* Trained model: [heiBOX link](https://heibox.uni-heidelberg.de/f/c956e55a939744b39d91/)

#### Scaling and binarization (without pre-train option)
* Run on `compgpu10`
* GPU7
* Memory usage: 32.45 GB
* GPU-Util: max 99% (but also not often)
* Training time: 2 days, 22:11:33 (much longer than before. Maybe because all GPUs that share the same CPU are using with maximum GPU-Util at the same time?)
* Training loss: 0.0812
* Training accuracy: 0.9688
* Inference results: [heiBOX folder](https://heibox.uni-heidelberg.de/d/f1ae6d46f24e4f40a153/)
* Trained model: [heiBOX link](https://heibox.uni-heidelberg.de/f/c0d062c5717d400ebbad/)

#### First impression
* The retrained model (model without pre-train option) creates a lot of false positives (noise) outside the main frame area, which is not the case for the fine-tuned model.
* The aritifical boundaries with 10 pixels buffering do not improve the separation of text blocks significantly
    * At some spots, the boundaries seem to help separate text blocks better, but in other cases they do not make a noticeable difference, or even confused the model more.
* Both models yield not promising results on non-Jingbao sample images.


### Training with artificial boundaries, 10 pixels buffering, no heading class

#### Scaling and binarization (with pre-train option)
* Run on `hgscomp01`
* GPU0
* Memory usage: 32.45 GB
* GPU-Util: max 99%
* Training time: 7:47:22
* Training loss: 0.0441
* Training accuracy: 0.9831
* Inference results: [heiBOX folder](https://heibox.uni-heidelberg.de/d/f6684a567b564418b550/)
* Trained model: [heiBOX link](https://heibox.uni-heidelberg.de/f/4c142a162f654383b8b9/)

#### First impression
It seems this model yields worse results in separating text blocks than other model. Reasons might include:
* The text class occupies most of pixels within the image, dominating other classes in model's prediction
* In our annotation, boundaries between texts and headings are usually thin and some headings streach over multiple text block. This might challenge the text block separation in test images.


### Training with artificial boundaries, 10 pixels buffering, with heading class, bigger input size

Some options in the configuration are updated as follows
```
input_height: 1728
input_width: 2432
transformer_num_patches_xy: [38, 27]
transformer_patchsize_x: 2
transformer_patchsize_y: 2
```

#### Scaling and binarization (with pre-train option)
* Run on `hgscomp01`
* GPU0
* Memory usage: 64.45 GB
* GPU-Util: max 99%
* Training time: x:xx:xxx
* Training loss: x.xx
* Training accuracy: x.xxx
* Inference results: [heiBOX folder]()
* Trained model: [heiBOX link]()

#### First impression
TBU.