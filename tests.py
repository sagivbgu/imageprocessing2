from utils import generate_pair, generate_triple
import numpy as np


def test_generate_pair():
    image = np.float32([
        [255, 0, 0],
        [0, 255, 0],
        [0, 255, 255]
    ])
    pairs = generate_pair(image, val=255)
    assert list(pairs) == [((0, 0), (1, 1)),
                           ((0, 0), (1, 2)),
                           ((0, 0), (2, 2)),

                           ((1, 1), (1, 2)),
                           ((1, 1), (2, 2)),

                           ((1, 2), (2, 2))]


def test_generate_triple():
    image = np.float32([
        [255, 0, 255],
        [0, 255, 0],
        [0, 255, 255]
    ])
    triples = generate_triple(image, val=255)
    assert list(triples) == [((0, 0), (2, 0), (1, 1)),
                             ((0, 0), (2, 0), (1, 2)),
                             ((0, 0), (2, 0), (2, 2)),
                             ((0, 0), (1, 1), (1, 2)),
                             ((0, 0), (1, 1), (2, 2)),
                             ((0, 0), (1, 2), (2, 2)),

                             ((2, 0), (1, 1), (1, 2)),
                             ((2, 0), (1, 1), (2, 2)),
                             ((2, 0), (1, 2), (2, 2)),

                             ((1, 1), (1, 2), (2, 2))]
