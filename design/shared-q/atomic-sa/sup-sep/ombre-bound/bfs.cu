#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <bits/stdc++.h>
#include <cooperative_groups.h>
using namespace std;
using namespace cooperative_groups;
namespace cg = cooperative_groups;

// shared-q/atomic-sa/sup-sep/ombre-bound/

#define NUM_SEP_SHARED_Q      11
#define NUM_SEP_GLOBAL_Q       7
#define NUM_SEP_GLOBAL_Q_      6
#define NUM_SEP_ALL_Q         18
#define NUM_SEP_ALL_Q_        17

#define FTR_SIZ_SHARED_Q    1116
#define FTR_SIZ_GLOBAL_Q 2097152

#define TILE_UPPER_ADJUST 3
#define TILE_ENLARGE_MASK 0x00000003

typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

__device__ int ftr_GQ[NUM_SEP_GLOBAL_Q][FTR_SIZ_GLOBAL_Q], ftr_size_GQ[NUM_SEP_GLOBAL_Q], NODE_i;
__global__ void CUDA_BFS_KERNEL_CG_NEW(Node *node, int *edge, short *cost, bool *done, int *NUM_NODES)
{
    __shared__ int ftr_SQ[NUM_SEP_SHARED_Q][FTR_SIZ_SHARED_Q], ftr_size_SQ[NUM_SEP_SHARED_Q], node_i;
    grid_group grid = this_grid();
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int num_thds = blockDim.x * gridDim.x;
    for(short iter = 0; !(*done) || !iter; iter++){
        if(               threadIdx.x < NUM_SEP_SHARED_Q){
            ftr_size_SQ[threadIdx.x] = 0; // (threadIdx.x == NUM_SEPQ_) ? 0 : ~(0xffffffff << threadIdx.x);
            if(threadIdx.x == 0){
                node_i = blockIdx.x * blockDim.x;
            }
        }
        if(!blockIdx.x && threadIdx.x < NUM_SEP_GLOBAL_Q){
            ftr_size_GQ[threadIdx.x] = 0; // (threadIdx.x == NUM_SEPQ_) ? 0 : ~(0xffffffff << threadIdx.x);
            if(threadIdx.x == 0){
                NODE_i = gridDim.x * blockDim.x;
            }
        }
        grid.sync();
        *done = true;
        while(node_i < *NUM_NODES){
            int idx = node_i + threadIdx.x;
            if(idx < *NUM_NODES && cost[idx] == iter && node[idx].length){
                short log_range = 0;
                for(int range = node[idx].length - 1; range && log_range != NUM_SEP_ALL_Q_; range = range >> 1) log_range++;
                (log_range < NUM_SEP_SHARED_Q) ? \
                    ftr_SQ[log_range                   ][atomicAdd(&(ftr_size_SQ[log_range                   ]), 1)] = idx : \
                    ftr_GQ[log_range - NUM_SEP_SHARED_Q][atomicAdd(&(ftr_size_GQ[log_range - NUM_SEP_SHARED_Q]), 1)] = idx;
            }
            __syncthreads();
            int mask = 0xffffffff, j_start_S = threadIdx.x, j_stripe_S = blockDim.x;
            for(int log_range = 0; log_range < NUM_SEP_SHARED_Q; log_range++){
                if(ftr_size_SQ[log_range] >= FTR_SIZ_SHARED_Q - blockDim.x){
                    for(int j = j_start_S; j < ftr_size_SQ[log_range]; j += j_stripe_S){
                        int ftr = ftr_SQ[log_range][j];
                        int id_ = (threadIdx.x & (~mask));
                        for(int eid = node[ftr].start + id_, i_end = node[ftr].start + node[ftr].length; eid < i_end; eid += (~mask + 1)){
                            int nid = edge[eid];
                            if(cost[nid] < 0){
                                cost[nid] = cost[ftr] + 1;
                                *done = false;
                            }
                        }
                    }
                    __syncthreads();
                    if(threadIdx.x == 0){
                        ftr_size_SQ[log_range] = 0;
                    }
                    __syncthreads();
                }
                if(log_range & TILE_ENLARGE_MASK){
                    j_start_S  = j_start_S  >> 1;
                    j_stripe_S = j_stripe_S >> 1;
                    mask       = mask       << 1;
                }
            }
            if(threadIdx.x == 0){
                node_i = atomicAdd(&NODE_i, blockDim.x);
            }
            __syncthreads();
        }
        int j_start_S = threadIdx.x, j_stripe_S = blockDim.x, mask = 0xffffffff;
        int j_start_G = id >> (NUM_SEP_SHARED_Q - TILE_UPPER_ADJUST), j_stripe_G = num_thds >> (NUM_SEP_SHARED_Q - TILE_UPPER_ADJUST);
        for(int log_range = 0; log_range < NUM_SEP_SHARED_Q; log_range++){
            for(int j = j_start_S; j < ftr_size_SQ[log_range]; j += j_stripe_S){
                int ftr = ftr_SQ[log_range][j];
                int id_ = (threadIdx.x & (~mask));
                for(int eid = node[ftr].start + id_, i_end = node[ftr].start + node[ftr].length; eid < i_end; eid += (~mask + 1)){
                    int nid = edge[eid];
                    if(cost[nid] < 0){
                        cost[nid] = cost[ftr] + 1;
                        *done = false;
                    }
                }
            }
            if(log_range & TILE_ENLARGE_MASK){
                j_start_S  = j_start_S  >> 1;
                j_stripe_S = j_stripe_S >> 1;
                mask       = mask       << 1;
            }
        }
        grid.sync();
        for(int log_range = 0; log_range < NUM_SEP_GLOBAL_Q_; log_range++){
            for(int j = j_start_G; j < ftr_size_GQ[log_range]; j += j_stripe_G){
                int ftr = ftr_GQ[log_range][j];
                int id_ = (id & (~mask));
                for(int eid = node[ftr].start + id_, i_end = node[ftr].start + node[ftr].length; eid < i_end; eid += (~mask + 1)){
                    int nid = edge[eid];
                    if(cost[nid] < 0){
                        cost[nid] = cost[ftr] + 1;
                        *done = false;
                    }
                }
            }
            /*if(log_range)*/{
                j_start_G  = j_start_G  >> 1;
                j_stripe_G = j_stripe_G >> 1;
                mask       = mask       << 1;
            }
        }
        for(int j = 0; j < ftr_size_GQ[NUM_SEP_GLOBAL_Q_]; j++){
            int ftr = ftr_GQ[NUM_SEP_GLOBAL_Q_][j];
            for(int eid = node[ftr].start + id, i_end = node[ftr].start + node[ftr].length; eid < i_end; eid += num_thds){
                int nid = edge[eid];
                if(cost[nid] < 0){
                    cost[nid] = cost[ftr] + 1;
                    *done = false;
                }
            }
        }
        grid.sync();
    }
}


// The BFS frontier corresponds to all the nodes being processed at the current level.



int main(int argc, char* argv[])
{
    printf("\033[0;1;33m"); cout << argv[1] + 15; printf("\033[0;1m\n");
	ifstream fin;
    int _, *NUM_NODES, NUM_EDGES, SOURCE;
    cudaMallocManaged(&NUM_NODES, sizeof(int));
    fin.open(argv[1]);
    fin >> *NUM_NODES >> NUM_EDGES >> SOURCE;

	Node* node;
	int* edge;
	short* cost;
	int* count;
    cudaMallocManaged(&node, sizeof(Node)*(*NUM_NODES));
    cudaMallocManaged(&edge, sizeof(int)*NUM_EDGES);
    cudaMallocManaged(&cost, sizeof(short)*(*NUM_NODES));
    cudaMallocManaged(&count, sizeof(int));
    for(int i=0;i<*NUM_NODES;i++) fin >> node[i].start >> node[i].length;
    for(int i=0;i<NUM_EDGES;i++) fin >> edge[i] >> _;
    fin.close();

    for(string is_exit = ""; is_exit != "y"; )
    {
        do{
            if(is_exit == ""){
                cout << "# Nodes : " << *NUM_NODES << endl;
                cout << "# Edges : " <<  NUM_EDGES << endl;
                cout << "Source  : " <<     SOURCE << endl;
            }
            else{
                cout << "Source  : ";
                cin >> SOURCE;
            }
        } while(SOURCE >= *NUM_NODES || SOURCE < 0);

        for(int i=0;i<*NUM_NODES;i++) cost[i] = (i == SOURCE) ? 0 : -1;

        vector<int> F;
        F.push_back(SOURCE);
        bool *X = (bool*)malloc(sizeof(bool)*(*NUM_NODES));
        for(int i=0;i<*NUM_NODES;i++) X[i] = (i == SOURCE);
        int *C = (int*)malloc(sizeof(int)*(*NUM_NODES));
        for(int i=0;i<*NUM_NODES;i++) C[i] = 0;
        // vector<vector<int>> frontier_every_round;
        // int maxi = 0;
        
        for(;!F.empty();)
        {
            vector<int> F_next;
            // maxi = F.size()>maxi?F.size():maxi;
            // vector<int> tmp;
            for(int i=F.size()-1;i>=0;i--)
            {
                int id = F[i];
                // tmp.push_back(id);
                int start = node[id].start;
                int end = start + node[id].length;
                for (int j = start; j < end; j++) 
                {
                    int nid = edge[j];
                    if (X[nid] == false)
                    {
                        X[nid] = true;
                        C[nid] = C[id] + 1;
                        F_next.push_back(nid);
                    }
                }
                F.pop_back();
            }
            F = F_next;
            // sort(tmp.begin(), tmp.end());
            // frontier_every_round.push_back(tmp);
        }

        int numBlocksPerSM = 1;
        int numThreads = 1024;
        bool* done;
        cudaMallocManaged(&done, sizeof(bool));
        (*done) = false;
        (*count) = 0;
        int device = -1;
        cudaGetDevice(&device);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, CUDA_BFS_KERNEL_CG_NEW, numThreads, 0);
        void *kernelArgs[] = {&node, &edge, &cost, &done, &NUM_NODES};
        dim3 num_blocks(deviceProp.multiProcessorCount * numBlocksPerSM, 1, 1);
        dim3 block_size(numThreads, 1, 1);
        // cout << "num of Nodes: " << *NUM_NODES << endl;
        // cout << "num_blocks: " << deviceProp.multiProcessorCount * numBlocksPerSM << endl;
        // cout << "block_size: " << numThreads << endl;

        cudaMemPrefetchAsync(node, sizeof(Node)*(*NUM_NODES), device, NULL);
        cudaMemPrefetchAsync(edge, sizeof(int)*NUM_EDGES, device, NULL);
        cudaMemPrefetchAsync(cost, sizeof(short)*(*NUM_NODES), device, NULL);
        cudaMemPrefetchAsync(done, sizeof(bool), device, NULL);
        cudaMemPrefetchAsync(NUM_NODES, sizeof(int), device, NULL);
    
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        auto a = cudaLaunchCooperativeKernel((void*)CUDA_BFS_KERNEL_CG_NEW, num_blocks, block_size, kernelArgs);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);

        int near_nodes = 0, near_errors = 0, dist = 0;
        for (int i = 0; i<*NUM_NODES; i++)
            if(X[i]){
                near_nodes++;
                if(cost[i] != C[i]) near_errors++;
                else dist = max(dist, cost[i]);
            }
        cout << "- Dist  = " << dist << endl;
        cout << "- Touch = " << (double)near_nodes / (double)(*NUM_NODES) * 100 << " % (" << near_nodes << " / " << *NUM_NODES << ")" << endl;
        (near_errors) ? printf("- Error = \033[31m%d\033[0;1m\n", near_errors) : printf("- Error = 0\n");
        cout << "- Time  = " << time << "ms" << endl;

        if(argv[argc - 1][0] == '$')
        {
            printf("\033[5mExit? [y/n]\033[0;1m ");
            cin >> is_exit;
            printf("\033[1A\033[K");
        }
        else is_exit = "y";
    }

    printf("\033[0m");
    cudaFree(NUM_NODES);
    cudaFree(node);
    cudaFree(edge);
    cudaFree(cost);
    cudaFree(count);
}