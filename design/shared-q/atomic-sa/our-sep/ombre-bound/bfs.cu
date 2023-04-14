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

// shared-q/atomic-sa/our-sep/ombre-bound/

#define TILE_LOG_A      3
#define TILE_SIZ_A      8
#define TILE_RNG_A      8
#define FTR_SIZE_A   4608

#define TILE_LOG_B      5
#define TILE_SIZ_B     32
#define TILE_RNG_B     64
#define FTR_SIZE_B   3584

#define TILE_LOG_C      7
#define TILE_SIZ_C    128
#define TILE_RNG_C   1024
#define FTR_SIZE_C   2560

#define TILE_LOG_D     10
#define TILE_SIZ_D   1024
#define TILE_RNG_D  16384
#define FTR_SIZE_D   1530

#define TILE_LOG_E     13
#define TILE_SIZ_E   8192
#define TILE_RNG_E 167936
#define FTR_SIZE_E  16384

#define FTR_SIZE_F  16384

typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

__device__ inline void VISIT_tiled(thread_group tile, Node *node, int *edge, short *cost, bool *done, int ftr)
{
    int start = node[ftr].start;
    int end = start + node[ftr].length;

    int idx = start + tile.thread_rank();
    int nid = edge[idx];
    if(cost[nid] < 0 && idx < end){
        cost[nid] = cost[ftr] + 1;
        *done = false;
    }
}

__device__ inline void VISIT_blk_tiled(short log_size, Node *node, int *edge, short *cost, bool *done, int ftr)
{
    int start = node[ftr].start;
    int end = start + node[ftr].length;
    for(int i = start + (threadIdx.x & ~(0xffffffff << log_size)); i < end; i += 1 << log_size){
        int nid = edge[i];
        if(cost[nid] < 0){
            cost[nid] = cost[ftr] + 1;
            *done = false;
        }
    }
}

__device__ inline void VISIT_grid_tiled(short log_size, Node *node, int *edge, short *cost, bool *done, int ftr)
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

__device__ int ftr_EQ[FTR_SIZE_E], ftr_FQ[FTR_SIZE_F];
__device__ int ftr_size_EQ, ftr_size_FQ, NODE_i;
__global__ void CUDA_BFS_KERNEL_CG_NEW(Node *node, int *edge, short *cost, bool *done, int *NUM_NODES)
{
    __shared__ int ftr_AQ[FTR_SIZE_A], ftr_BQ[FTR_SIZE_B], ftr_CQ[FTR_SIZE_C], ftr_DQ[FTR_SIZE_D];
    __shared__ int ftr_size_AQ, ftr_size_BQ, ftr_size_CQ, ftr_size_DQ, node_i;
    grid_group grid = this_grid();
    thread_group tile_S = tiled_partition(grid, TILE_SIZ_A);
    
    for(short iter = 0; !(*done) || !iter; iter++){
        if(threadIdx.x == 0){
            ftr_size_AQ = ftr_size_BQ = ftr_size_CQ = ftr_size_DQ = 0;
            node_i = blockIdx.x * blockDim.x;
        }
        if(threadIdx.x + blockIdx.x * blockDim.x == 0){
            ftr_size_EQ = ftr_size_FQ = 0;
            NODE_i = gridDim.x * blockDim.x;
        }
        grid.sync();
        *done = true;
        while(node_i < *NUM_NODES){
            int idx = node_i + threadIdx.x;
            if(idx < *NUM_NODES && cost[idx] == iter && node[idx].length){
                (node[idx].length <= TILE_RNG_A) ? ftr_AQ[atomicAdd(&ftr_size_AQ, 1)] = idx : \
                (node[idx].length <= TILE_RNG_B) ? ftr_BQ[atomicAdd(&ftr_size_BQ, 1)] = idx : \
                (node[idx].length <= TILE_RNG_C) ? ftr_CQ[atomicAdd(&ftr_size_CQ, 1)] = idx : \
                (node[idx].length <= TILE_RNG_D) ? ftr_DQ[atomicAdd(&ftr_size_DQ, 1)] = idx : \
                (node[idx].length <= TILE_RNG_E) ? ftr_EQ[atomicAdd(&ftr_size_EQ, 1)] = idx : \
                                                   ftr_FQ[atomicAdd(&ftr_size_FQ, 1)] = idx ;
            }
            __syncthreads();
            if(ftr_size_AQ >= FTR_SIZE_A - blockDim.x){
                for(int j = threadIdx.x >> TILE_LOG_A; j < ftr_size_AQ; j += blockDim.x >> TILE_LOG_A){
                    VISIT_tiled(tile_S, node, edge, cost, done, ftr_AQ[j]);
                }
                __syncthreads();
                if(threadIdx.x == 0){
                    ftr_size_AQ = 0;
                }
                __syncthreads();
            }
            if(ftr_size_BQ >= FTR_SIZE_B - blockDim.x){
                for(int j = threadIdx.x >> TILE_LOG_B; j < ftr_size_BQ; j += blockDim.x >> TILE_LOG_B){
                    VISIT_blk_tiled(TILE_LOG_B, node, edge, cost, done, ftr_BQ[j]);
                }
                __syncthreads();
                if(threadIdx.x == 0){
                    ftr_size_BQ = 0;
                }
                __syncthreads();
            }
            if(ftr_size_CQ >= FTR_SIZE_C - blockDim.x){
                for(int j = threadIdx.x >> TILE_LOG_C; j < ftr_size_CQ; j += blockDim.x >> TILE_LOG_C){
                    VISIT_blk_tiled(TILE_LOG_C, node, edge, cost, done, ftr_CQ[j]);
                }
                __syncthreads();
                if(threadIdx.x == 0){
                    ftr_size_CQ = 0;
                }
                __syncthreads();
            }
            if(ftr_size_DQ >= FTR_SIZE_D - blockDim.x){
                for(int j = threadIdx.x >> TILE_LOG_D; j < ftr_size_DQ; j += blockDim.x >> TILE_LOG_D){
                    VISIT_blk_tiled(TILE_LOG_D, node, edge, cost, done, ftr_DQ[j]);
                }
                __syncthreads();
                if(threadIdx.x == 0){
                    ftr_size_DQ = 0;
                }
                __syncthreads();
            }
            if(threadIdx.x == 0){
                node_i = atomicAdd(&NODE_i, blockDim.x);
            }
            __syncthreads();
        }
        for(int j = threadIdx.x >> TILE_LOG_A; j < ftr_size_AQ; j += blockDim.x >> TILE_LOG_A){
            VISIT_tiled(tile_S, node, edge, cost, done, ftr_AQ[j]);
        }
        for(int j = threadIdx.x >> TILE_LOG_B; j < ftr_size_BQ; j += blockDim.x >> TILE_LOG_B){
            VISIT_blk_tiled(TILE_LOG_B, node, edge, cost, done, ftr_BQ[j]);
        }
        for(int j = threadIdx.x >> TILE_LOG_C; j < ftr_size_CQ; j += blockDim.x >> TILE_LOG_C){
            VISIT_blk_tiled(TILE_LOG_C, node, edge, cost, done, ftr_CQ[j]);
        }
        for(int j = threadIdx.x >> TILE_LOG_D; j < ftr_size_DQ; j += blockDim.x >> TILE_LOG_D){
            VISIT_blk_tiled(TILE_LOG_D, node, edge, cost, done, ftr_DQ[j]);
        }
        grid.sync();
        for(int j = (threadIdx.x + blockIdx.x * blockDim.x) >> TILE_LOG_E; j < ftr_size_EQ; j += (blockDim.x * gridDim.x) >> TILE_LOG_E){
            VISIT_grid_tiled(TILE_LOG_E, node, edge, cost, done, ftr_EQ[j]);
        }
        for(int j = 0; j < ftr_size_FQ; j++){
            VISIT_all_tiled(node, edge, cost, done, ftr_FQ[j]);
        }
        grid.sync();
    }
}

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