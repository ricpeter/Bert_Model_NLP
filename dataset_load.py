df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
batch_1 = df[:2000]
batch_1[1].value_counts()
