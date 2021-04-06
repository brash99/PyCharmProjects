import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kmeans import *
from test_kmeans import *
from testing import *

x = TestKMeans()

#x.test_sse_00()
#x.test_sse_01()
#x.test_sse_02()

#x.test_labels_00()
#x.test_labels_01()
#x.test_labels_02()

#x.test_recompute_centroids_00()
#x.test_recompute_centroids_01()
#x.test_recompute_centroids_02()

#x.test_2D_2K_00()
#x.test_2D_2K_01()
#x.test_3D_2K_00()
x.test_3D_3K_00()