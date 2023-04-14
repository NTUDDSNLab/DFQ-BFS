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

// shared-q/scan-free/non-sep/lower-bound/

#define TILE_LOG_A    0
#define TILE_SIZ_A    1
#define FTR_SIZE_A 6138

#define FTR_SIZE_G 16777216

typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

__device__ inline void VISIT_tiled(thread_group tile, Node *node, int *edge, unsigned int *cost, bool *done, int ftr, int *ftr_AQ, int *ftr_size_AQ)
{
    int start = node[ftr].start;
    int end = start + node[ftr].length;
    
    for(int i = start+tile.thread_rank(); i < end; i += tile.size()){
        int nid = edge[i];
        if(atomicCAS(&(cost[nid]), (unsigned int)(0xffffffff), (unsigned int)(cost[ftr] + 1)) == (unsigned int)(0xffffffff)){
            if(node[nid].length) {
                ftr_AQ[atomicAdd(ftr_size_AQ, 1)] = nid;
            }
            *done = false;
        }
    }
}

// __device__ int NODE_i;
__device__ int ftr_GQ[FTR_SIZE_G], NODE_i;
__global__ void CUDA_BFS_KERNEL_CG_NEW(Node *node, int *edge, unsigned int *cost, bool *done, int *NUM_NODES)
{
    __shared__ int ftr_AQ_a[FTR_SIZE_A], ftr_size_AQ_a;
    __shared__ int ftr_AQ_b[FTR_SIZE_A], ftr_size_AQ_b;
    __shared__ int *ftr_AQ_f, *ftr_AQ_t, *ftr_size_AQ_f, *ftr_size_AQ_t, node_i;
    grid_group grid = this_grid();
    thread_group tile_S = tiled_partition(grid, TILE_SIZ_A);
    
    ftr_size_AQ_a = ftr_size_AQ_b = 0;
    __syncthreads();
    for(int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < *NUM_NODES; idx += gridDim.x * blockDim.x){
        if(!cost[idx]){
            NODE_i = 1;
            ftr_GQ[0] = idx;
        }
    }
    grid.sync();

    for(short iter = 0; !(*done) || !iter; iter++){
        if(threadIdx.x == 0){
            // printf("iter: %d\nsize a/b: %d %d\n", iter, ftr_size_SQ_a, ftr_size_SQ_b);
            ftr_size_AQ_f = ((iter % 2) ? &ftr_size_AQ_a : &ftr_size_AQ_b);
            ftr_size_AQ_t = ((iter % 2) ? &ftr_size_AQ_b : &ftr_size_AQ_a);
            *ftr_size_AQ_f = NODE_i / gridDim.x + (blockIdx.x < (NODE_i % gridDim.x));
            *ftr_size_AQ_t = 0;
            ftr_AQ_f = ((iter % 2) ? ftr_AQ_a : ftr_AQ_b);
            ftr_AQ_t = ((iter % 2) ? ftr_AQ_b : ftr_AQ_a);
            // node_i = blockIdx.x * blockDim.x;
        }
        __syncthreads();
        for(int i = threadIdx.x; i < *ftr_size_AQ_f; i += blockDim.x){
            ftr_AQ_f[i] = ftr_GQ[i + (NODE_i / gridDim.x) * blockIdx.x + ((blockIdx.x < (NODE_i % gridDim.x)) ? blockIdx.x : (NODE_i % gridDim.x))];
        }
        grid.sync();
        if(threadIdx.x + blockIdx.x * blockDim.x == 0){
            NODE_i = 0;// gridDim.x * blockDim.x;
        }
        *done = true;
        // while(node_i < *NUM_NODES){
        //     int idx = node_i + threadIdx.x;
        //     if(idx < *NUM_NODES && cost[idx] == iter && node[idx].length){
        //         ftr_AQ[atomicAdd(&ftr_size_AQ, 1)] = idx;
        //     }
        //     __syncthreads();
        //     if(ftr_size_AQ >= FTR_SIZE_A - blockDim.x){
        //         for(int j = threadIdx.x >> TILE_LOG_A; j < ((iter % 2) ? ftr_size_AQ_a : ftr_size_AQ_b); j += blockDim.x >> TILE_LOG_A){
        //             VISIT_tiled(tile_S, node, edge, cost, done, ftr_AQ_f[j], ftr_AQ_t, ((iter % 2) ? &ftr_size_AQ_b : &ftr_size_AQ_a));
        //         }
        //         __syncthreads();
        //         if(threadIdx.x == 0){
        //             ftr_size_AQ = 0;
        //         }
        //         __syncthreads();
        //     }
        //     if(threadIdx.x == 0){
        //         node_i = atomicAdd(&NODE_i, blockDim.x);
        //     }
        //     __syncthreads();
        // }
        grid.sync();
        for(int j = 0; j < *ftr_size_AQ_f; j += blockDim.x >> TILE_LOG_A){
            int idx = j + (threadIdx.x >> TILE_LOG_A);
            if(idx < *ftr_size_AQ_f){
                VISIT_tiled(tile_S, node, edge, cost, done, ftr_AQ_f[idx], ftr_AQ_t, ftr_size_AQ_t);
            }
            __syncthreads();
            if(threadIdx.x == 0){
                node_i = atomicAdd(&NODE_i, *ftr_size_AQ_t);
            }
            __syncthreads();
            for(int i = threadIdx.x; i < *ftr_size_AQ_t; i += blockDim.x){
                ftr_GQ[node_i + i] = ftr_AQ_t[i];
            }
            __syncthreads();
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
	unsigned int* cost;
	int* count;
    cudaMallocManaged(&node, sizeof(Node)*(*NUM_NODES));
    cudaMallocManaged(&edge, sizeof(int)*NUM_EDGES);
    cudaMallocManaged(&cost, sizeof(unsigned int)*(*NUM_NODES));
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

        for(int i=0;i<*NUM_NODES;i++) cost[i] = (i == SOURCE) ? 0 : 0xffffffff;

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
        cudaMemPrefetchAsync(cost, sizeof(unsigned int)*(*NUM_NODES), device, NULL);
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