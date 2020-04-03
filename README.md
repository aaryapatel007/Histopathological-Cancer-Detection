# Histopathological-Cancer-Detection
Created an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans. The data for this competition is a slightly modified version of the PatchCamelyon (PCam) benchmark dataset

## Dataset

The dataset used for the research is a slightly modified    version    of    the    PatchCamelyon    (PCam)    [21],[22].The original PCam dataset contains duplicate images due to its probabilistic sampling, however, this   version   does   not   contain   duplicates.   The   dataset is open-source and can be downloaded from (<https://www.kaggle.com/c/histopathologic-cancer-detection/data>).   The   dataset   has   more   than   220K   RGB images  with  a  dimension  of  96x96x3.  The  given  problem  is  the  binary  classification  problem  where  the  associated label has two class labels i.e. tumor and non-tumor tissues. A  positive  label  indicates  that  the  center  32x32px  region  of a  patch  contains  at  least  one  pixel  of  tumor  tissue.
