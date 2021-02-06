from typing import Dict, List, Tuple

import scipy
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares as ALS
from tqdm import tqdm


from data.dataset import train, test, books


def als_model() -> Tuple[Dict[int, List], Dict[int, List]]:
    global train, test

    sol = test.groupby(["user_id"])["book_id"].agg({"unique"}).reset_index()
    gt = {}

    for user in tqdm(sol["user_id"].unique()):
        gt[user] = list(sol[sol["user_id"] == user]["unique"].values[0])

    read_list = train.groupby(["user_id"])["book_id"].agg({"unique"}).reset_index()
    popular_rec_model = books.sort_values(by="books_count", ascending=False)[
        "book_id"
    ].values[0:500]

    user2idx = {l: i for i, l in enumerate(train["user_id"].unique())}
    book2idx = {l: i for i, l in enumerate(train["book_id"].unique())}
    idx2user = {i: user for user, i in user2idx.items()}
    idx2book = {i: item for item, i in book2idx.items()}

    data = train[["user_id", "book_id"]].reset_index(drop=True)
    useridx = data["useridx"] = train["user_id"].apply(lambda x: user2idx[x]).values
    bookidx = data["bookidx"] = train["book_id"].apply(lambda x: book2idx[x]).values
    rating = np.ones(len(data))

    purchase_sparse = scipy.sparse.csr_matrix(
        (rating, (useridx, bookidx)), shape=(len(set(useridx)), len(set(bookidx)))
    )

    als_model = ALS(factors=20, regularization=0.01, iterations=100)
    als_model.fit(purchase_sparse.T)
    total_rec_list = {}
    for user in tqdm(data["useridx"].unique()):
        rec_list = []

        # 기존에 만든 Book ID를 변경
        seen = read_list[read_list["user_id"] == idx2user[user]]["unique"].values[0]
        recs = als_model.recommend(user, purchase_sparse, N=250)
        recs = [idx2book[x[0]] for x in recs][0:250]

        for rec in recs:
            if rec not in seen:
                rec_list.append(rec)

        if len(rec_list) < 200:
            for i in popular_rec_model[0:200]:
                if rec not in seen:
                    rec_list.append(rec)

        total_rec_list[idx2user[user]] = rec_list[0:200]
    return total_rec_list, gt
