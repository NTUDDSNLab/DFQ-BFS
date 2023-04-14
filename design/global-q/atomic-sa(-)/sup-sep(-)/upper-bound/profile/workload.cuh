using namespace std;
#define maxStep 100
typedef struct{
    int node_prop;
    int visit;
} workload;

__device__ workload works[82*1024];
__device__ inline void node_prop(workload *work, short iter){
    work[iter * 82 * 1024 + threadIdx.x + blockIdx.x * blockDim.x].node_prop++;
}
__device__ inline void visit(){
    works[threadIdx.x + blockIdx.x * blockDim.x].visit++;
}

__device__ inline void writeback(workload *work, short iter){
    work[iter * 82 * 1024 + threadIdx.x + blockIdx.x * blockDim.x].visit += works[threadIdx.x + blockIdx.x * blockDim.x].visit;
    works[threadIdx.x + blockIdx.x * blockDim.x].visit = 0;
}

__host__ workload* workloadInit(){
    workload *work;
    cudaMallocManaged(&work, 82*1024*maxStep*sizeof(workload));
    for(int i = 0; i < 82 * maxStep * 1024; i++){
        work[i].node_prop = 0;
        work[i].visit = 0;

    }
    return work;
}
__host__ void print_log(workload *work){
    int accu = 0;
    cout << 82*1024 << endl;
    for(int i = 0; i < maxStep; i++){
        int max = 0;
        for(int j = 0; j < 82*1024; j++){
            cout << work[82*1024*i + j].visit << ' ';
            if(max < work[82*1024*i + j].visit)
                max = work[82*1024*i + j].visit;
        }
        if(max == 0) break;
        accu += max;
        max = 0;
        cout << endl << accu << endl;
    }
}