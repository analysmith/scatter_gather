import numpy as np
import csv
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
from scipy.sparse import csr_matrix
import json
from typing import *

glove = defaultdict(lambda : np.random.random(100))

def load_papers(use_abstracts=False) -> Set[str]:
    count = 0
    titles = set()
    abstracts = set()
    with open("all_papers.csv", encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for p in reader:
            try:
                titles.add(p["Title"])
                if "Abstract Note" in p and p["Abstract Note"]:
                    abstracts.add( p["Title"] + " " + p["Abstract Note"])
                else:
                    abstracts.add(p["Title"] + " " + "No Abstract")
                count += 1
            except:
                pass
    print("Number of papers:", count)
    print("Number of unique titles:", len(titles))
    if use_abstracts:
        return abstracts
    return titles


def load_glove_embeddings():
    global glove
    print("Loading glove embeddings")
    if len(glove) == 0:
        with open("glove.6B.100d.txt", encoding="utf-8") as f:
            for line in f:
                values = re.split(" ", line)
                word = values[0]
                vector = np.array([float(x) for x in values[1:]])
                glove[word] = vector
    print("glove embeddings loaded")


def ngram_cluster(texts: Set[str]) -> Dict[int, List[str]]:
    texts = list(sorted(texts))
    cluster2texts = defaultdict(lambda : [])
    print("Processing texts")
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" "), ngram_range=(1,4), min_df=2, max_df=1.0)
    text_mat = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=int(len(texts) / 10))
    clusters = kmeans.fit_predict(text_mat)
    for idx, c in enumerate(clusters):
        cluster2texts[c].append(texts[idx])
    return cluster2texts


def glove_cluster(texts: Set[str]) -> Dict[int, List[str]]:
    texts = list(sorted(texts))
    cluster_seed = np.random.random(100)
    cluster2texts = defaultdict(lambda : [])
    embedded_text = []
    print("Processing texts")
    for text in texts:
        cbow = cluster_seed
        num_words = 0
        for w in text.lower().split(" "):
            if w in glove:
                cbow  += glove[w.lower()]
            num_words += 1
        embedded_text.append(cbow / num_words)
    kmeans = KMeans(n_clusters=int(len(texts) / 10))
    clusters = kmeans.fit_predict(csr_matrix(np.array(embedded_text)))
    for idx, c in enumerate(clusters):
        cluster2texts[c].append(texts[idx])
    return cluster2texts

def scatter_gather(titles: Set[str]):
    ungathered_titles = set([t for t in titles])
    gather_bag: List[Tuple(str, List[str])] = []
    gather_set = set()
    round = 0
    while True:
        do_scatter: str = input("Scatter? (y/n):")
        if do_scatter.lower() == "y" or do_scatter.lower() == "yes":
            print("Round %s" % round)
            clusters = glove_cluster(ungathered_titles)
            for k, ts in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
                print("Cluster:", k)
                for t in ts[:5]:
                    print("\t" + t)
                print()
            indices: str = input("Which ones will you gather?")
            for idx_pair in indices.split(";"):
                idx_str, cluster_name = idx_pair.split("*")
                idx: int = int(idx_str)
                gather_bag.append((cluster_name, [t for t in clusters[idx]]))
                gather_set.update(clusters[idx])
                ungathered_titles = ungathered_titles - set(clusters[idx])
        else:
            break
    save: str = input("Save file? (y/n):")
    if save.lower() == "y" or save.lower() == "yes":
        fname: str = input("filename (default=scatter_gather_out.json):")
        if fname.strip() == "":
            fname = "scatter_gather_out.json"
        with open(fname, "w") as f:
            json.dump(gather_bag, f)

if __name__ == "__main__":
    load_glove_embeddings()
    paper_titles = load_papers(use_abstracts=True)
    scatter_gather(paper_titles)