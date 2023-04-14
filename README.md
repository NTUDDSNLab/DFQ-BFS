# DFQ-BFS: A Decentralized Frontier Queue for Improving Scalability of Breadth-First-Search on GPUs

## 1. Getting started Instructions.
- Clone this project
`git clone git@github.com:NTUDSNLab/DFQ-BFS.git`
- Hardware:
    - `CPU x86_64` (Test on Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz)
    - `NVIDIA GPU (arch>=86)` with device memory >= 12GB.(Support NVIDIA RTX3080(sm_86). Note that we mainly evaluate our experience on RTX3090. The execution time could be different with different devices.
- OS & Compler:
    - `Ubuntu 18.04`
    - `CUDA = 11.6`
    - `nvcc = 11.6` 
- Important Files/Directories
    - `data/`: contains all datasets that we want to compare with.
    - `bin/`: contains all binaries from different designs (including baseline with name started with `1_`) that we want to compare with. Note that all the source codes of each binary can be found in the `design/` directory.
    - `design/`: contains all the source codes organized by tree structure, each sub-directory name describing the design choice.
    - `plot.py`: The python script that traversal all the datasets in `data/` with all binaries in `bin/`, and also plot their runtime speedup with a histogram.

    **noting that the first two lines
    \# Nodes: #node  Edges: #edge
    \# FromNodeId ToNodeId
    are necessary!!!**
    the following is the example:
    
        ```
        # Nodes: 239 Edges: 502
        # FromNodeId    ToNodeId
        0       1
        0       2
        0       3
        ...
        ```

## 2. Environment Setup

### 1) Pick up the implementation you want in `design/` and compile it
```
cd design/implementation/you/want/
nvcc -O3 --compiler-options -Wall -Xptxas -v bfs.cu -o bfs
```

### 2) copy it into `design/bin`
```
cp bfs design/bin
```

### 3) unzip dataset under 'data/' or download it from [SNAP](http://snap.stanford.edu/data/index.html)

```
tar xvf data.tar
```

### 4) run the python script
```
python plot.py
```

### 5) check the result png file
