import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

dataset = datasets.load_digits()
data = dataset.data
labels = dataset.target

# Find the best representant of each class
# Find the Non-Trivial class by comparing the best representant with the ones of the others classes

# Creating a dict that have an empty list to every label
samples_by_class = dict(map(lambda x: (x, []), list(range(10))))

for sample, label in zip(data, labels):
    samples_by_class[label].append(sample)

euc_dist = lambda sample1, sample2: sum((sample1-sample2)**2) ** 0.5

for label, samples in samples_by_class.items():

    representant = np.array([])
    min_dist = 1e20             # init with with a big value

    # the ith sample will be compared with every other jth sample
    for i in range(len(samples)):
        cur_dist = 0
        for j in range(len(samples)):
            if i == j: continue

            # taking the euclidean distance between the ith samples and every other
            # we want to find the sample that has the minimum distance to every other member of its class
            cur_dist += euc_dist(samples[i], samples[j])

        # get the representant and its dist sum
        if min_dist > cur_dist:
            representant, min_dist = samples[i], cur_dist

    samples_by_class[label] = { 'samples': samples, 'representant': representant }

# from PIL import Image
# for c in samples_by_class.values():
#     im = Image.fromarray(c['representant'].reshape((8, 8,)))
#     im.show()

# The non-trivial class is the one that has the shortest distance to every other class(easiest to classify as another class)
non_trivial_label = -1
min_dist = 1e20
for i in range(10):
    cur_dist = 0
    for j in range(10):
        cur_dist += euc_dist(samples_by_class[i]['representant'], samples_by_class[j]['representant']) if i != j else 0

    if min_dist > cur_dist:
        min_dist = cur_dist
        non_trivial_label = i

print('The non-trivial label is: ', non_trivial_label)
