import pandas as pd
import os
import copy

idrid_labels = pd.read_csv('../B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv')
dr_labels = pd.read_csv('../Kaggle_DR/trainLabels.csv')
merged_labels = copy.deepcopy(idrid_labels)
merged_path = '../B. Disease Grading/2. Groundtruths/merged_labels.csv'
columns = list(idrid_labels.columns)

kaggle_dir = '../Kaggle_DR/data'
for dir in os.listdir(kaggle_dir):
    dir_path = os.path.join(kaggle_dir, dir)
    img_names = os.listdir(dir_path)
    for img_name in img_names:
        img_path = os.path.join(dir_path, img_name)
        img_id = img_name.split('.')[0]
        target = dr_labels[dr_labels['image'] == img_id]
        row = [0]*len(columns)
        row[0] = img_id
        row[1] = target['level'].values[0]
        to_append = pd.Series(row, index=merged_labels.columns)
        merged_labels = merged_labels.append(to_append, ignore_index=True)

merged_labels.to_csv(merged_path, index=False)
print("Done")