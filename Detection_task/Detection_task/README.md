## Dependencies

We use PyTorch, TorchVision, PIL, and some transformations from [albumentations](https://github.com/albumentations-team/albumentations) library.
```bash
conda install -c pytorch torchvision captum
conda install -c albumentations albumentations
```

## Training
You should provide a choice for he model as argument ["RetinaNet", "FasterRCNN"]

To launch the training script:
```bash
python train.py --model "RetinaNet" --depth 101 --epochs 10
```

## Inference
This script evaluates the performance of the model on a given dataset

```bash
python evaluate.py --model "RetinaNet" --depth 101 --weights models/RetinaNet.PTH --dataset "test"
```
You can also plot a prediction for a given image by providing its index in the dataset by adding an index as integer
(be careful to not exceed the length of the dataset)

```bash
python evaluate.py --model "RetinaNet" --depth 101 --weights models/RetinaNet.PTH --dataset "test" --img_idx idx
```
You will get a plot like :

![example prediction](./figures/Figure_1.png)

## Data

The data folders are in the same format as provided in the IDRID challenge
You can find the data here (https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid).

