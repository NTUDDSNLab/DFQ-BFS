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

// shared-q/atomic-sa/our-sep(+)/ombre-bound/

#define TILE_LOG_A      3
#define TILE_SIZ_A      8
#define TILE_RNG_A      8
#define FTR_SIZE_A   3584

#define TILE_LOG_B      4
#define TILE_SIZ_B     16
#define TILE_RNG_B     64
#define FTR_SIZE_B   3584

#define TILE_LOG_C      6
#define TILE_SIZ_C     64
#define TILE_RNG_C   1024
#define FTR_SIZE_C   2560

#define TILE_LOG_D      9
#define TILE_SIZ_D    512
#define TILE_RNG_D  16384
#define FTR_SIZE_D   1536

#define TILE_LOG_E     13
#define TILE_SIZ_E   8192
#define TILE_RNG_E 167936
#define FTR_SIZE_E   3584

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
    // for(int i = start+tile.thread_rank(); i < end; i += tile.size()){
    //     int nid = edge[i];
    //     if(cost[nid] < 0){
    //         cost[nid] = cost[ftr] + 1;
    //         *done = false;
    //     }
    // }
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
// __device__ int num_S, num_M, num_L;
__global__ void CUDA_BFS_KERNEL_CG_NEW(Node *node, int *edge, short *cost, bool *done, int *NUM_NODES/*, int* tile_size*/)
{
    // short log2_tile_size = -1, tile_size_L = (*tile_size)*(*tile_size);
    // for(short tile_size_ = *tile_size; tile_size_; tile_size_ = tile_size_ >> 1) log2_tile_size++;
    __shared__ int ftr_AQ[FTR_SIZE_A], ftr_BQ[FTR_SIZE_B], ftr_CQ[FTR_SIZE_C], ftr_DQ[FTR_SIZE_D];
    __shared__ int ftr_size_AQ, ftr_size_BQ, ftr_size_CQ, ftr_size_DQ, node_i;
    // __shared__ int idx_i_CQ[32], IDX_i_CQ;
    grid_group grid = this_grid();
    thread_group tile_S = tiled_partition(grid, TILE_SIZ_A);
    // int id = threadIdx.x + blockIdx.x * blockDim.x;
    
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
            if(ftr_size_AQ >= FTR_SIZE_A - blockDim.x - (blockDim.x >> TILE_LOG_A)){
                for(int j = threadIdx.x >> TILE_LOG_A; j < ftr_size_AQ; j += blockDim.x >> TILE_LOG_A){
                    VISIT_tiled(tile_S, node, edge, cost, done, ftr_AQ[j]);
                }
                __syncthreads();
                if(threadIdx.x == 0){
                    ftr_size_AQ = 0;
                }
                __syncthreads();
            }
            if(ftr_size_BQ >= FTR_SIZE_B - blockDim.x - (blockDim.x >> TILE_LOG_B)){
                for(int j = threadIdx.x >> TILE_LOG_B; j < ftr_size_BQ; j += blockDim.x >> TILE_LOG_B){
                    VISIT_blk_tiled(TILE_LOG_B, node, edge, cost, done, ftr_BQ[j]);
                }
                __syncthreads();
                if(threadIdx.x == 0){
                    ftr_size_BQ = 0;
                }
                __syncthreads();
            }
            if(ftr_size_CQ >= FTR_SIZE_C - blockDim.x - (blockDim.x >> TILE_LOG_C)){
                for(int j = threadIdx.x >> TILE_LOG_C; j < ftr_size_CQ; j += blockDim.x >> TILE_LOG_C){
                    VISIT_blk_tiled(TILE_LOG_C, node, edge, cost, done, ftr_CQ[j]);
                }
                __syncthreads();
                if(threadIdx.x == 0){
                    ftr_size_CQ = 0;
                }
                __syncthreads();
            }
            if(ftr_size_DQ >= FTR_SIZE_D - blockDim.x - (blockDim.x >> TILE_LOG_D)){
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
    // if(id == 0) printf("%d %d %d\n", num_S, num_M, num_L);
}

int main(int argc, char* argv[])
{
    ;
}