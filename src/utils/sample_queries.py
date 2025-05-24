import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def subsample_json(input_path, query_field, output_path, k=1000):
    data = json.load(open(input_path))
    if isinstance(data, dict) and query_field in data and isinstance(data[query_field], list):
        items = data[query_field]
        wrapper = data
        key = query_field
    else:
        items = data
        wrapper = None
        key = None
    n_items = len(items)
    if n_items > k:
        queries = [item[query_field] for item in items]
        tfidf = TfidfVectorizer().fit_transform(queries)
        kmeans = KMeans(n_clusters=k, random_state=42).fit(tfidf)
        distances = kmeans.transform(tfidf)
        selected_indices = [int(distances[:, j].argmin()) for j in range(k)]
        subsampled = [items[i] for i in selected_indices]
    else:
        subsampled = items
    if wrapper is not None:
        wrapper[key] = subsampled
        output = wrapper
    else:
        output = subsampled
    json.dump(output, open(output_path, 'w'), indent=2)


