import glob
import os
import pandas as pd
import random

CsvFromPGFolder = r"E:\dataset\page_graph"
CsvFromPGFiles = glob.glob(os.path.join(CsvFromPGFolder, r"*.graphml\text_dataset.csv"))
random.shuffle(CsvFromPGFiles)

CsvFromImageFolder = r"E:\dataset\*_text"
CsvFromImageFiles = glob.glob(os.path.join(CsvFromImageFolder, r"*.csv"))
random.shuffle(CsvFromImageFiles)

CsvOutputFile = r".\dataset_new.csv"
CsvOutput = pd.DataFrame(columns=["Text", "Label"])

for CsvFromPGFile in CsvFromPGFiles:
    print("Dealing file: {}".format(CsvFromPGFile))
    CsvFromPG = pd.read_csv(CsvFromPGFile, header=None, names=["Text", "Label"])
    CsvOutput = pd.concat([CsvOutput, CsvFromPG], ignore_index=True)

    # if CsvOutput.shape[0] >= 100000:
    #     break

PGFile_data_size = CsvOutput.shape[0]

for CsvFromImageFile in CsvFromImageFiles:
    print("Dealing file: {}".format(CsvFromImageFile))
    CsvFromImage = pd.read_csv(CsvFromImageFile, header=0, index_col=0).rename(columns={"text": "Text", "is ad": "Label"})
    CsvOutput = pd.concat([CsvOutput, CsvFromImage], ignore_index=True)

    if CsvOutput.shape[0] > PGFile_data_size * 1.2:
        break

    # if CsvOutput.shape[0] >= 150000:
    #     break

CsvOutput.to_csv(CsvOutputFile, index=False)
