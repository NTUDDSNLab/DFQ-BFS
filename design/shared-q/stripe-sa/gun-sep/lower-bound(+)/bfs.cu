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

// global-q/stripe-sa/gun-sep/lower-bound/

#define FTR_SIZE_S 33554432
#define FTR_SIZE_M 16777216
#define FTR_SIZE_L  8388608

#define TILE_LOG_S   0
#define TILE_SIZ_S   1
#define TILE_RNG_S  32

#define TILE_LOG_M   5
#define TILE_SIZ_M  32
#define TILE_RNG_M 256

#define TILE_LOG_L   8
#define TILE_SIZ_L 256

typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

__device__ inline void VISIT_tiled(thread_group tile, Node *node, int *edge, short *cost, bool *done, int ftr)
{
    int start = node[ftr].start;
    int end = start + node[ftr].length;
    
    for(int i = start+tile.thread_rank(); i < end; i += tile.size()){
        int nid = edge[i];
        if(cost[nid] < 0){
            cost[nid] = cost[ftr] + 1;
            *done = false;
        }
    }
}

__device__ inline void VISIT_non_tiled(short log_size, Node *node, int *edge, short *cost, bool *done, int ftr)
{
    int start = node[ftr].start;
    int end = start + node[ftr].length;
    for(int i = start + ((threadIdx.x + blockIdx.x * blockDim.x) & ~(0xffffffff << log_size)); i < end; i += 1 << log_size){
        int nid = edge[i];
        if(cost[nid] < 0){
            cost[nid] = cost[ftr] + 1;
            *done = false;
        }
    }
}

__device__ inline void VISIT_all_tiled(Node *node, int *edge, short *cost, bool *done, int ftr)
{
    int start = node[ftr].start;
    int end = start + node[ftr].length;
    for(int i = start + threadIdx.x + blockIdx.x * blockDim.x; i < end; i += blockDim.x * gridDim.x){
        int nid = edge[i];
        if(cost[nid] < 0){
            cost[nid] = cost[ftr] + 1;
            *done = false;
        }
    }
}

__device__ int ftr_SQ[FTR_SIZE_S], ftr_MQ[FTR_SIZE_M], ftr_LQ[FTR_SIZE_L];
__device__ int ftr_size_SQ, ftr_size_MQ, ftr_size_LQ;
__global__ void CUDA_BFS_KERNEL_CG_NEW(Node *node, int *edge, short *cost, bool *done, int *NUM_NODES)
{
    grid_group grid = this_grid();
    thread_group tile_S = tiled_partition(grid, TILE_SIZ_S);
    
    for(short iter = 0; !(*done) || !iter; iter++){
        if(threadIdx.x + blockIdx.x * blockDim.x == 0){
            ftr_size_SQ = ftr_size_MQ = ftr_size_LQ = 0;
        }
        grid.sync();
        *done = true;
        // stripe scan sa
        for(int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < *NUM_NODES; idx += blockDim.x * gridDim.x){
            if(idx < *NUM_NODES && cost[idx] == iter && node[idx].length){
                (node[idx].length <= TILE_RNG_S) ? ftr_SQ[atomicAdd(&ftr_size_SQ, 1)] = idx : \
                (node[idx].length <= TILE_RNG_M) ? ftr_MQ[atomicAdd(&ftr_size_MQ, 1)] = idx : \
                                                   ftr_LQ[atomicAdd(&ftr_size_LQ, 1)] = idx ;
            }
        }
        grid.sync();
        for(int j = ((threadIdx.x + blockIdx.x * blockDim.x) >> TILE_LOG_S); j < ftr_size_SQ; j += (blockDim.x * gridDim.x) >> TILE_LOG_S){
            VISIT_tiled(tile_S, node, edge, cost, done, ftr_SQ[j]);
        }
        for(int j = ((threadIdx.x + blockIdx.x * blockDim.x) >> TILE_LOG_M); j < ftr_size_MQ; j += (blockDim.x * gridDim.x) >> TILE_LOG_M){
            VISIT_non_tiled(TILE_LOG_M, node, edge, cost, done, ftr_MQ[j]);
        }
        for(int j = ((threadIdx.x + blockIdx.x * blockDim.x) >> TILE_LOG_L); j < ftr_size_LQ; j += (blockDim.x * gridDim.x) >> TILE_LOG_L){
            VISIT_non_tiled(TILE_LOG_L, node, edge, cost, done, ftr_LQ[j]);
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