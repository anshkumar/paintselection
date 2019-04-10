# Paintselection and GrabCut (C++, Python and Javascript)

## Features
### Multilevel narrow band graph cut [1]

The multilevel banded Graph Cuts consists of three steps: coarsening, initial segmentation and uncoarsening. The coarsening is done directly on the image. This can be done with any standard multiresolution image technique. An example of simple procedure that can be used is downsampling. The original image is not considered part of the memory consumption overhead of this step. All these steps are depicted in Figure

![alt text](https://github.com/anshkumar/paintselection/blob/master/img/gAVoN.png)

During the first stage, several many smaller images {I_1, I_2, ..., I_k} are constructed based in the original image I0 such that the size constraint M(k) < M(k−1) is satisfied for each image dimension n = {1, 2, ..., k} and each image level k = {1, 2, ..., k}. Obviously, the image seeds are also reduced and the multilevel graphs are constructed directly based upon these low resolution images.

The second stage is the segmentation of the coarsest image I_k where k is the largest level defined in this instance problem. A graph G_k = (V_k,E_k) is defined for I_k and their minimum cut is obtained. This minimum cut yields a segmentation of the image I_k.

During the final step, a binary boundary image J_k is constructed, representing all these image elements that are identified by the nodes of the cut C_k, k ∈ {1, 2, ..., k}. The boundary image is projected onto a higher resolution boundary image J(k−1) at level k − 1. The resulting boundary image J(k−1) contains a narrow band that limits the candidate boundaries of the object elements to be extracted from I(k−1). The band width can be controlled by an optional dilation parameter d > 0. If d is small, the method may not be able to recover the full details of the objects with high shape complexity or large curvature. Moreover, if d is large, the computational benefits of banded graph cuts are reduced and the wider band may also introduce potential outliers far away from the desired object boundaries.

### Mulithreaded image segmentation [3]

Fast image segmentation using MaxFlow/MinCut Boykov-Kolmogorov algorithm

Opencv grabcut algorithm is pretty good, but slow for large images. One bottleneck is the function GCGraph::MaxFlow, which implements the max-flow/min-cut Boykov-Kolmogorov algorithm.A simple parallel version of this algorithm is praposed , providing a significant speedup.

    The class GCGraph (cf. file gcgraph.hpp) is extended with an overloaded function maxFlow(int r). It computes the max flow corresponding to a subgraph of the initial graph.

    A function constructGCGraph_slim (cf. grabcut.cpp) is implemented building a partially reduced graph. The reduction is based on a paper of Scheuermann and Rosenhahn : https://pdfs.semanticscholar.org/92df/9a469fe878f55cd0ef3d55477a5f787c47ba.pdf

    A mulithreaded version of the function estimateSegmentation() (cf. file grabcut.cpp) is implemented, using our overloaded function maxFlow(). Threads run on disjoint subgraphs, corresponding to disjoint subregions of the image, thus no synchronization is needed. The residual graph is updated and the partial flows are added. A second parallel run with slightly shifted regions is performed to process inter-region edges. A last call to maxFlow() on the whole residual graph achieves the segmentation.

Source Files:

dist/sources/modules/imgproc/include/opencv2/imgproc.hpp

dist/sources/modules/imgproc/src/gcgraph.hpp

dist/sources/modules/imgproc/src/grabcut.cpp

dist/sources/modules/js/src/embindgen.py


Example Files:

grabcut.cpp

grabcut.html

grabcut.py

paintselection.cpp

paintselection.html

References:
1) Liu, Jiangyu, Jian Sun, and Heung-Yeung Shum. "Paint selection." ACM Transactions on Graphics (ToG) 28.3 (2009): 69.
2) Rother, Carsten, Vladimir Kolmogorov, and Andrew Blake. "Grabcut: Interactive foreground extraction using iterated graph cuts." ACM transactions on graphics (TOG). Vol. 23. No. 3. ACM, 2004.
3) https://github.com/bvirxx/opencv3
