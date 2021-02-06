from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.dataset import train, test, books


def baseline_model() -> Tuple[Dict[int, List], Dict[int, List]]:
    global train, test

    sol = test.groupby(["user_id"])["book_id"].agg({"unique"}).reset_index()
    gt = {}

    for user in tqdm(sol["user_id"].unique()):
        gt[user] = list(sol[sol["user_id"] == user]["unique"].values[0])

    rec_df = pd.DataFrame()
    rec_df["user_id"] = train["user_id"].unique()
    popular_rec_model = books.sort_values(by="books_count", ascending=False)[
        "book_id"
    ].values[0:500]

    # 통계기반 모델 데이터셋
    train = pd.merge(
        train, books[["book_id", "authors", "ratings_count"]], how="left", on="book_id"
    )
    agg = train.groupby(["user_id", "authors"])["authors"].agg({"count"}).reset_index()
    agg = agg.sort_values(by="count", ascending=False)

    author_books = (
        books[["book_id", "authors", "ratings_count"]]
        .sort_values(by=["authors", "ratings_count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    author_rec_model = agg.merge(author_books, how="left", on=["authors"])

    # 내가 읽을 책의 목록 추출
    read_list = train.groupby(["user_id"])["book_id"].agg({"unique"}).reset_index()
    total_rec_list = {}
    for user in tqdm(rec_df["user_id"].unique()):
        rec_list = []
        author_rec_model_ = author_rec_model[author_rec_model["user_id"] == user][
            "book_id"
        ].values
        seen = read_list[read_list["user_id"] == user]["unique"].values[0]
        for rec in author_rec_model_:
            if rec not in seen:
                rec_list.append(rec)
        if len(rec_list) < 200:
            for i in popular_rec_model[0:200]:
                if i not in seen:
                    rec_list.append(i)

        total_rec_list[user] = rec_list[0:200]

    return total_rec_list, gt
