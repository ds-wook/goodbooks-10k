# %%

import os
import sys
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%

path = "../input/goodbooks-10k/"

books = pd.read_csv(path + "books.csv")
book_tags = pd.read_csv(path + "book_tags.csv")
ratings = pd.read_csv(path + "ratings.csv")
tags = pd.read_csv(path + "tags.csv")
to_read = pd.read_csv(path + "to_read.csv")

# %%
books.head()
# %%
books.columns
# %%
books["small_image_url"].values[0]
# %%

books = books[
    ["book_id", "authors", "title", "ratings_count", "average_rating", "language_code"]
].reset_index(drop=True)


# %%


agg = books.groupby("authors")["authors"].agg({"count"})

fig, ax = plt.subplots(figsize=(12, 8))
sns.distplot(agg["count"], ax=ax, kde=False)
plt.title("Number of the Author's Book")
plt.xlabel("Book Count")
plt.ylabel("Author Count")
plt.show()


# %%

print(f"책의 숫자: {books['book_id'].nunique()}")
print(f"저자의 숫자: {books['authors'].nunique()}\n")
pd.DataFrame(agg["count"].describe()).T


# %%

agg.sort_values(by="count", ascending=False)


# %%


fig, ax = plt.subplots(figsize=(12, 8))
sns.distplot(books["average_rating"], ax=ax, kde=False, bins=30)
plt.title("Number of the Author's Book")
plt.xlabel("Average Rating")
plt.ylabel("Author Count")
plt.show()


# %%

books[books["average_rating"] <= 3].shape[0]
# %%

books.sort_values(by="average_rating", ascending=False).head()
# %%


fig, ax = plt.subplots(figsize=(12, 8))
sns.distplot(books["ratings_count"], ax=ax, kde=False)
plt.title("Number of the Author's Book")
plt.xlabel("Rating Count")
plt.ylabel("Author Count")
plt.show()


# %%

pd.DataFrame(books["ratings_count"].describe()).T
# %%


fig, ax = plt.subplots(figsize=(12, 8))
sns.distplot(books[books["ratings_count"] < 1000000]["ratings_count"], ax=ax, kde=False)
plt.title("Number of the Author's Book")
plt.xlabel("Rating Count")
plt.ylabel("Author Count")
plt.show()


# %%

agg = pd.DataFrame(books["language_code"].value_counts()).reset_index()
agg.columns = ["language_code", "count"]
agg.head()

# %%

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x="language_code", y="count", data=agg, ax=ax)
plt.title("Number of the Author's Book")
plt.xlabel("Rating Count")
plt.ylabel("Author Count")
plt.show()

# %%

ratings

# %% [markdown]
# ratings에는 있지만, books에는 없는 책의 id의 수를 계산
len(set(ratings["book_id"].unique()).difference(set(books["book_id"].unique())))

# %%

book_tags = pd.merge(tags, book_tags, how="left", on="tag_id")
book_tags.head()
# %%

agg = book_tags.groupby("tag_name")["count"].agg({"sum"}).reset_index()
agg = agg.sort_values(by="sum", ascending=False).reset_index(drop=True)
agg.head()

# %%

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x="tag_name", y="sum", data=agg.loc[:20], ax=ax)
plt.title("Top 20: Tag Cont")
plt.xlabel("Tag name")
plt.ylabel("Tag count")
plt.xticks(rotation=60)
plt.show()

# %%

agg = ratings.groupby(["user_id"])["book_id"].agg({"count"}).reset_index()
agg

# %%


fig, ax = plt.subplots(figsize=(12, 8))
sns.distplot(agg["count"], ax=ax, kde=False, bins=30)
plt.title("Average Number of the Read Count")
plt.xlabel("Read Count")
plt.ylabel("User Count")
plt.show()


# %%

agg = ratings.groupby(["book_id"])["book_id"].agg({"count"}).reset_index()
fig, ax = plt.subplots(figsize=(12, 8))
sns.distplot(agg["count"], ax=ax, kde=False, bins=30)
plt.title("Average Readed Count")
plt.xlabel("Readed Count")
plt.ylabel("User Count")
plt.show()


# %%

books[books["book_id"].isin([1, 2, 3, 4, 5, 6, 7, 8])].head()
# %%

agg = (
    ratings[ratings["book_id"].isin([1, 2, 3, 4, 5, 6, 7, 8])]
    .groupby(["user_id"])["book_id"]
    .agg({"nunique"})
)
agg = agg.reset_index()
agg = agg.groupby(["nunique"])["user_id"].agg({"count"}).reset_index()

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x="nunique", y="count", data=agg, ax=ax)
plt.title("Harry Potter's Reading Count")
plt.xlabel("Series Count")
plt.ylabel("Reading Person Count")
plt.xticks(rotation=60)
plt.show()


# %%

agg["ratio"] = agg["count"] / agg["count"].sum()
agg[['nunique', "ratio"]].T

# %%
