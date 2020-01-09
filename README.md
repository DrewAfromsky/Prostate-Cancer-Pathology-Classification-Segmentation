# Gleason Grading - Prostate Cancer Tissue Microarray Classfication

![Alt text](/Prostate-Cancer-Pathology-Classification-Segmentation/img.png?raw=true "Optional Title")


The Dataset can be downloaded at the following link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP&version=1.0

### Files and their purposes
1. Image_Patches.py - utility python script for creating central image regions of size 250x250 from the original TMA spots of size 3100x3100
2. Mask_Patches.py - utility python script for creating central image regions of size 250x250 from the masks of size 3100x3100
3. Train_&_Masks.ipynb runs 1. and 2.
4. gleason_labels_3.csv - a csv file containing the single class labels for training/validation data (central image regions containing single class labels, not multiple labels)
5. mask_to_label.py - a utility function that converts the masks to the appropriate class
6. Gleason_Training.ipynb - the main notebook containing the code for training and validation. Contains the code for obtaining all figures (i.e. confusion matrix, training/validation loss and accuracy, and pixel-wise segmentation)
 
### Key Steps for Pathology Classification were as follows:

1. Acquire TMA Spot pathology dataset
2. Create image patches for patch-based training
3. Create mask patches that correspond to training data
4. Train network and save best model

The inspiration paper for this project can be found here: https://www.nature.com/articles/s41598-018-30535-1






