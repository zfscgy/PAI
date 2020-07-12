import pandas as pd


data_root = "Test/TestDataset/Data/"
# data column 0 - 71
# label column 72
dataset = pd.read_csv(data_root + "credit_default.csv")
data_size = len(dataset.index)

index = ["ID-%08d" % i for i in range(data_size)]
dataset["index"] = index
dataset = dataset.set_index("index")
dataset_0_29 = dataset.iloc[:, :30].sample(frac=1)
dataset_30_71 = dataset.iloc[:, 30:72].sample(frac=1)
labelset = dataset.iloc[:, 72:].sample(frac=1)
dataset_0_29.to_csv(data_root + "Splitted_Indexed_Data/credit_default_data1.csv", index=True, header=False)
dataset_30_71.to_csv(data_root + "Splitted_Indexed_Data/credit_default_data2.csv", index=True, header=False)
labelset.to_csv(data_root + "Splitted_Indexed_Data/credit_default_label.csv", index=True, header=False)