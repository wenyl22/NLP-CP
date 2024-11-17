from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction
def self_bleu(conclusions: list[int]) -> float:
    smoothie = SmoothingFunction().method1
    scores = []
    for i in range(len(conclusions)):
        references = conclusions[:i] + conclusions[i+1:]
        score = sentence_bleu(references, conclusions[i], smoothing_function=smoothie)
        scores.append(score)
    return np.mean(scores)

def edit_distance_grouping(conclusions: list[str], threshold: float = 0.8) -> list[list[str]]:
    def compute_sequence_match(conclusion1, conclusion2):
        return SequenceMatcher(None, conclusion1, conclusion2).ratio()

    def parallel_sequence_matching(conclusions: list[str], threshold: float = 0.8) -> list[list[str]]:
        n = len(conclusions)
        similarity_matrix = Parallel(n_jobs=-1)(delayed(compute_sequence_match)(conclusions[i], conclusions[j]) for i in range(n) for j in range(i+1, n))
        full_similarity_matrix = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                full_similarity_matrix[i, j] = similarity_matrix[idx]
                full_similarity_matrix[j, i] = similarity_matrix[idx]
                idx += 1 
        full_similarity_matrix = np.exp(full_similarity_matrix) / np.sum(np.exp(full_similarity_matrix), axis=1, keepdims=True)
        # sum the diagonal term
        return np.sum(full_similarity_matrix.diagonal()) / n

    return parallel_sequence_matching(conclusions, threshold)

def distinct_n_grams(conclusions: list[list[int]], n: int) -> float:
    n_grams = set()
    tot = 0
    for conclusion in conclusions:
        # print("HAHAHA", conclusion)
        for i in range(len(conclusion) - n + 1):
            n_grams.add(tuple(conclusion[i:i+n]))
            tot += 1
    return len(n_grams) / tot

def cosine_similarity_grouping(embeddings: list[list[float]], threshold: float = 0.8) -> list[list[list[float]]]:
    matrix = cosine_similarity(embeddings, embeddings)
    # softmax along each row
    matrix = np.exp(matrix) / np.sum(np.exp(matrix), axis=1)
    # sum the diagonal term
    return np.sum(matrix.diagonal()) / len(embeddings)

def spectral_clustering(embeddings: list[list[float]], k: int) -> float:
    pass

def k_means_grouping(embeddings: list[list[float]], k: int) -> float:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    clusters = [[] for _ in range(k)]
    for label, embedding in zip(labels, embeddings):
        clusters[label].append(embedding)

    # compute average intertia of each cluster
    ret = 0
    for i in range(k):
        center = np.mean(clusters[i], axis=0)
        ret += np.sum((clusters[i] - center) ** 2)
    return ret / k