"""
Author: happygirlzt
Date: 19th Feb 2021
"""

import pickle

saved_classifier = '...'
clf = pickle.loads(saved_classifier)

def compute_similarity_bucket():
    """
    compute the similarity between a bug report and a bucket
    """
    max_prob = 0
    tests = [get_features(q, report) for report in bucket]

    for test in tests:
        probility = svm_predict(test)
        max_prob = max(probility, max_prob)

    return max_prob


def generate_candidate_reports(n):
    candidates = []
    heapify(candidates)

    for bucket in self.buckets:
        similarity = self.compute_similarity_bucket()
        master_bucket.similarity = similarity
        heappush(candidates, master_bucket)

    return nlargest(n, candidates)
