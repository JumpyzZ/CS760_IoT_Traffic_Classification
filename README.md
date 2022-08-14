# CS760_IoT_Traffic_Classification

Dataset link: http://205.174.165.80/IOTDataset/CIC_IOT_Dataset2022/CICIOT/

UNSW dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset    --- only requries 2 GB 

N_BaIot dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/00442/ --- requries 14+ GB


## Preprocessing

### PCA - N_BaIot

To get pcaed data with device name and traffic type:

0. Make sure RAM is enough

1. Download full dataset, unzip all rar files, rename folder '00442' to 'N_BaIot' and replace the 'N_BaIot' folder you colned.
```
wget -r --no-parent https://archive.ics.uci.edu/ml/machine-learning-databases/00442/
```

2. Run pca_N_BaIot.py, this should give a full pcaed data
```
cd ..../CS760_IoT_Traffic_Classification
python pca_N_BaIot.py
```

### PCA - UNSW

There are serveral train and test datasets available of reduced size, this should not take longer than 30 seconds to preform PCA on
