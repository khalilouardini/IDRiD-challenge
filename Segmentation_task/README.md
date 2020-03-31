
## Dataset
Download segmentation dataset from: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

The dataset should be stored in a "Segmentation.nosync" folder (or you should change the main_path variable in main.py to the desired folder).

Your dataset should be divided into 4 folders:
- test_images: folder containing all the raw test images
- test_masks: folder containing 5 folder respectively called 'EX', 'HE', 'MA', 'OD', 'SE' with the test masks for each segmentation task. (some images can have no corresponding mask for a certain task)
- train_images: folder containing all the raw train images
- train_masks: folder containing 5 folder respectively called 'EX', 'HE', 'MA', 'OD', 'SE' with the train masks for each segmentation task. (some images can have no corresponding mask for a certain task)

## Train and test model
You can use either the Segmentation.ipynb file (preferably you should store the 'Segmentation.nosync' folder in your google drive and run the notebook using Google Colab after having imported/mounted your drive in your session) or the main.py file.

## Evaluate our model
To run inference with our model download the weights in the .pth file from: https://drive.google.com/open?id=1rmor-yqJew9E16NqFkzRWcx3zck0kD75
and run test_model.py. Make sure to change 'save_path' variable to the path where you downloaded the weights .pth file.




