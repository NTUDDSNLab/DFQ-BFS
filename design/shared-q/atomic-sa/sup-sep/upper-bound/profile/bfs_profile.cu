#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include "workload.cuh"
using namespace std;
using namespace cooperative_groups;
namespace cg = cooperative_groups;

#define NUM_SEPQ        18
#define NUM_SEPQ_       17
#define FTR_SIZE  16777216

typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

__device__ int ftr_Q[NUM_SEPQ][FTR_SIZE], ftr_size_Q[NUM_SEPQ];
__global__ void CUDA_BFS_KERNEL_CG_NEW(Node *node, int *edge, short *cost, bool *done, int *NUM_NODES, workload *work)
{
    grid_group grid = this_grid();
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int num_thds = blockDim.x * gridDim.x;
    for(short iter = 0; !(*done) || !iter; iter++){
        if(!blockIdx.x && threadIdx.x < NUM_SEPQ){
            ftr_size_Q[threadIdx.x] = (threadIdx.x == NUM_SEPQ_) ? 0 : ~(0xffffffff << threadIdx.x);
        }
        grid.sync();
        *done = true;
        for(int idx = id; idx < *NUM_NODES; idx += num_thds){
            if(idx < *NUM_NODES && cost[idx] == iter && node[idx].length){
                short log_range = 0;
                for(int range = (node[idx].length) ? (node[idx].length - 1) : node[idx].length; range && log_range != NUM_SEPQ_; range = range >> 1) log_range++;
                ftr_Q[log_range][atomicAdd(&(ftr_size_Q[log_range]), 1)] = idx;
                node_prop(work, iter);
            }
        }
        grid.sync();
        for(int log_range = 0, mask = 0xffffffff, j_start = id, j_stripe = num_thds; log_range < NUM_SEPQ_; log_range++){
            for(int j = j_start + (~(0xffffffff << log_range)); j < ftr_size_Q[log_range]; j += j_stripe){
                int ftr = ftr_Q[log_range][j];
                int id_ = (id & (~mask));
                for(int eid = node[ftr].start + id_, i_end = node[ftr].start + node[ftr].length; eid < i_end; eid += (~mask + 1)){
                    visit();
                    int nid = edge[eid];
                    if(cost[nid] < 0){
                        cost[nid] = cost[ftr] + 1;
                        *done = false;
                    }
                }
            }
            /*if(log_range)*/{
                j_start  = j_start  >> 1;
                j_stripe = j_stripe >> 1;
                mask     = mask     << 1;
            }
        }
        for(int j = 0; j < ftr_size_Q[NUM_SEPQ_]; j++){
            int ftr = ftr_Q[NUM_SEPQ_][j];
            for(int eid = node[ftr].start + id, i_end = node[ftr].start + node[ftr].length; eid < i_end; eid += num_thds){
                int nid = edge[eid];
                visit();
                if(cost[nid] < 0){
                    cost[nid] = cost[ftr] + 1;
                    *done = false;
                }
            }
        }
        grid.sync();
        writeback(work, iter);
    }
}


// The BFS frontier corresponds to all the nodes being processed at the current level.



int main(int argc, char* argv[])
{
    //printf("\033[0;1;33m"); cout << argv[1] + 15; printf("\033[0;1m\n");
	ifstream fin;
    int _, *NUM_NODES, NUM_EDGES, SOURCE;
    cudaMallocManaged(&NUM_NODES, sizeof(int));
    fin.open(argv[1]);
    fin >> *NUM_NODES >> NUM_EDGES >> SOURCE;

	Node* node;
	int* edge;
	short* cost;
	int* count;
    workload *work = workloadInit();
    cudaMallocManaged(&node, sizeof(Node)*(*NUM_NODES));
    cudaMallocManaged(&edge, sizeof(int)*NUM_EDGES);
    cudaMallocManaged(&cost, sizeof(short)*(*NUM_NODES));
    cudaMallocManaged(&count, sizeof(int));
    for(int i=0;i<*NUM_NODES;i++) fin >> node[i].start >> node[i].length;
    for(int i=0;i<NUM_EDGES;i++) fin >> edge[i] >> _;
    fin.close();

    for(string is_exit = ""; is_exit != "y"; )
    {
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
        void *kernelArgs[] = {&node, &edge, &cost, &done, &NUM_NODES, &work};
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

        
        is_exit = "y";
    }
    print_log(work);

    cudaFree(NUM_NODES);
    cudaFree(node);
    cudaFree(edge);
    cudaFree(cost);
    cudaFree(count);
}