import numpy as np
import pandas as pd


path = "../../input/goodbooks-10k/"

books = pd.read_csv(path + "books.csv")
book_tags = pd.read_csv(path + "book_tags.csv")
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
tags = pd.read_csv(path + "tags.csv")
to_read = pd.read_csv(path + "to_read.csv")

train["book_id"] = train["book_id"].astype(np.str)
test["book_id"] = test["book_id"].astype(np.str)
books["book_id"] = books["book_id"].astype(np.str)
