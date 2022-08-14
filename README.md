# CS760_IoT_Traffic_Classification

Dataset link: http://205.174.165.80/IOTDataset/CIC_IOT_Dataset2022/CICIOT/

UNSW dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset    --- only requries 2 GB 


## Preprocessing

### PCA - N_BaIot

To get pcaed data with device name and traffic type:

0. Make sure RAM is enough(python will use 14GB+)

1. Download full dataset, unzip all rar files, rename folder '00442' to 'N_BaIot' and replace the 'N_BaIot' folder you colned.
```
wget -r --no-parent https://archive.ics.uci.edu/ml/machine-learning-databases/00442/
```

2. Run pca.py
```
cd ..../CS760_IoT_Traffic_Classification
python pca.py
```
