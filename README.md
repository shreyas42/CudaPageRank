# CudaPageRank
## Project for the High Performance Computer Architecture course.
### A simple implementation of Google's PageRank algorithm running on a GPU.
### Link to the dataset: https://snap.stanford.edu/data/web-Google.html


### Design:
The graph is stored internally as a vertex and edge list representation. The ranks are updated using the Power method over a user defined number of iterations.
