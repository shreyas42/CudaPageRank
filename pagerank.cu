#include<cstdio>
#include<cstdlib>
#include<iostream>

#define DFL_LEN 32
#define MAX_THREADS_PER_BLOCK 1024 //supported by hardware, run ./deviceQuery to determine

//cuda error checking 
#define check_error(ans) {cudaCheckError((ans),__FILE__,__LINE__);}

inline void cudaCheckError(cudaError_t e,const char *file,int line,bool abort = true){
    if(e != cudaSuccess){
        fprintf(stderr,"GPUassert: %s\nFile: %s\nLine: %d\n",cudaGetErrorString(e),file,line);
        if(abort) exit(e);
    }
}
//end of error checking

typedef long int g_type;

//struct declarations and global variables begin here


struct vertex{
    g_type number;
    g_type start;
    int n;
};

struct map{
    g_type node;
    g_type index;
};

struct entry{
    g_type edge;
    double val;
};

g_type *edges;
g_type edges_length;
g_type edges_size;
g_type edges_itr;


struct vertex *vertex_list;
g_type vertex_length;
g_type vertex_size;
g_type vertex_itr; 

struct entry *transitions;
struct map *node_map;
double *ranks;

//end of struct and global definitions

//start of interface
int init_edges(){
    if(edges != NULL)
        return 0;
    edges = (g_type *)malloc(DFL_LEN * sizeof(g_type));
    if(edges == NULL){
        fprintf(stderr,"Malloc failed in %s at line %d\n",__FILE__,__LINE__);
        return 0;
    }
    edges_size = DFL_LEN;
    edges_length = 0;
    edges_itr = 0;
    return 1;
}

void delete_edges(){
    edges_length = 0;
    edges_size = DFL_LEN;
    edges_itr = 0;
    if(edges != NULL)
        free(edges);
}

int add_edge(int edge){
    if(edges == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return 0;
    } 
    if(edges_length == edges_size){
        edges_size *= 2;
        edges = (g_type *)realloc(edges,edges_size * sizeof(g_type));
        if(edges == NULL){
            fprintf(stderr,"Malloc failed in %s at line %d\n",__FILE__,__LINE__);
            return 0;
        }
    }
    edges[edges_length] = edge;
    edges_length++;
    return 1; 
}

int get_edge(g_type *e){
    
    if(edges == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return 0; 
    }
    if(e == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return 0;
    }
    g_type val = edges[edges_itr];
    edges_itr++;
    if(edges_itr >= edges_size){
        edges_itr = edges_itr % edges_size;
    }
    *e = val;
    return 1;
}

void reset_edge(){
    edges_itr = 0;
}

void move_edge(g_type index){
    edges_itr = index;
}

int init_vertices(){
    if(vertex_list != NULL)
        return 0;
    vertex_list = (struct vertex *)malloc(DFL_LEN * sizeof(struct vertex));
    if(vertex_list == NULL){
        fprintf(stderr,"Malloc failed in %s at line %d\n",__FILE__,__LINE__);
        return 0;
    }
    vertex_length = 0;
    vertex_size = DFL_LEN;
    vertex_itr = 0;
    return 1;
}

void delete_vertices(){
    vertex_itr = 0;
    vertex_length = 0;
    vertex_size = 0;
    if(vertex_list != NULL)
        free(vertex_list);
}

int add_vertex(struct vertex v){
    if(vertex_list == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return 0;
    }
    if(vertex_length == vertex_size){
        vertex_size *= 2;
        vertex_list = (struct vertex *)realloc(vertex_list,vertex_size * sizeof(struct vertex));
        if(vertex_list == NULL){
            fprintf(stderr,"Malloc failed in %s at line %d\n",__FILE__,__LINE__);
            return 0;
        }
    }
    vertex_list[vertex_length].number = v.number;
    vertex_list[vertex_length].n = v.n;
    vertex_list[vertex_length].start = v.start;
    g_type temp = vertex_length;
    vertex_length++;
    return temp;
}

int get_vertex(struct vertex *v){
    if(vertex_list == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return 0;
    }
    v->number = vertex_list[vertex_itr].number;
    v->start = vertex_list[vertex_itr].start;
    v->n = vertex_list[vertex_itr].n;
    vertex_itr++;
    if(vertex_itr >= vertex_size){
        vertex_itr = vertex_itr % vertex_size;
    }
    return 1;
}

void reset_vertex(){
    vertex_itr = 0;
}

void move_vertex(g_type index){
    vertex_itr = index; 
}

void build_graph(FILE *fp){
    if(vertex_list == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return ;
    }
    if(edges == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return ;
    }
    if(fp == NULL){
        fprintf(stderr,"File pointer error in %s at line %d\n",__FILE__,__LINE__);
        return ;
    }
    g_type from,to;
    int seen = 0;
    g_type cur = -1;
    while(fscanf(fp,"%ld %ld",&from,&to) != -1){
        if(from == vertex_list[cur].number && vertex_length != 0){
            seen = 1;
        }
        else{
            seen = 0;
        }
        if(!seen){
            struct vertex temp;
            temp.number = from;
            temp.start = edges_length;
            temp.n = 0;
            cur = add_vertex(temp);
        }
        add_edge(to);
        vertex_list[cur].n++;
    }
}

void create_map(){
   if(vertex_list == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return ;
    }
    if(edges == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return ;
    }
    if(node_map == NULL){
        node_map = (struct map *)malloc(vertex_length * sizeof(struct map));
        if(node_map == NULL){
            fprintf(stderr,"Malloc failed in %s at line %d\n",__FILE__,__LINE__);
            return ;
        }
    }
    for(int i=0;i<vertex_length;i++){
        node_map[i].node = vertex_list[i].number;
        node_map[i].index = i;
    }
}

g_type search_map(g_type node){
   if(vertex_list == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return -1;
    }
    if(edges == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return -1;
    } 
    if(node_map == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return -1;
    }
    for(int i=0;i<vertex_length;i++){
        if(node_map[i].node == node)
            return node_map[i].index;
    }
    return -1;
}

__device__ g_type search_dmap(struct map *d_map,g_type *d_vlength,g_type node){
    if(d_map == NULL){
        return -1;
    }
    g_type len = *d_vlength;
    for(g_type i=0;i<len;i++){
        if(d_map[i].node == node)
            return d_map[i].index;
    }
    return -1;
}

void delete_map(){
    if(node_map != NULL)
        free(node_map);
}

void init_ranks(){
    if(vertex_list == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return ;
    }
    if(edges == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return ;
    }
    if(ranks == NULL){
        ranks = (double *)malloc(vertex_length * sizeof(double));
        if(ranks == NULL){
            fprintf(stderr,"Malloc failed in %s at line %d\n",__FILE__,__LINE__);
            return ;
        }
    }
    for(int i=0;i<vertex_length;i++){
        ranks[i] = 0.25;
    }
}

void delete_ranks(){
    if(ranks != NULL)
        free(ranks);
}

void init_transitions(){
    if(vertex_list == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return ;
    }
    if(edges == NULL){
        fprintf(stderr,"Null pointer error in %s at line %d\n",__FILE__,__LINE__);
        return ;
    }
    if(transitions == NULL){
        transitions = (struct entry *)malloc(edges_length * sizeof(struct entry));
        if(transitions == NULL){
            fprintf(stderr,"Malloc failed in %s at line %d\n",__FILE__,__LINE__);
            return ;
        }
    }
    for(g_type i=0;i<vertex_length;i++){
        g_type start = vertex_list[i].start;
        g_type j = start;
        int n = vertex_list[i].n;
        while(j < start + n){
            transitions[j].edge = edges[j];
            transitions[j].val = 1.0 / vertex_list[i].n;
            j++;
        }
    }
}

void delete_transitions(){
    if(transitions != NULL)
        free(transitions);
}
//end of interface

//CUDA kernels
__global__ void multiply_kernel(struct vertex *d_vertices,struct entry *d_transitions,double *d_ranks,double *d_res,g_type *d_vlength){
   int threadId = blockDim.x * blockIdx.x + threadIdx.x;
   double b = d_ranks[threadId];
   g_type len = *d_vlength;
   if(threadId < len){
       for(g_type i = d_vertices[threadId].start;i < d_vertices[threadId].start + d_vertices[threadId].n;i++){
           double a = d_transitions[i].val;
           double res = a * b;
           d_res[i] = res; 
       }
   }
}

__global__ void add_kernel(struct vertex *d_vertices,struct entry *d_transitions,double *d_res,struct map *d_map,double *d_tempranks,g_type *d_vlength){
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    g_type len = *d_vlength;
    if(threadId < len){
        for(g_type i = d_vertices[threadId].start;i < d_vertices[threadId].start + d_vertices[threadId].n;i++){
            int index = search_dmap(d_map,d_vlength,d_transitions[i].edge);
            double val = d_res[i];
            double temp = d_tempranks[index];
            __syncthreads();
            temp += val;
            d_tempranks[index] = temp;
            __syncthreads();
        }
    }
}

__global__ void update_kernel(double *d_tempranks,double *d_ranks,g_type *d_vlength){
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    g_type len = *d_vlength;
    if(threadId < len){
        d_ranks[threadId] = d_tempranks[threadId];
    }
}

//end of CUDA kernels

//main program begins here
int main(int argc,char **argv){
    if(argc != 3){
        fprintf(stderr,"Correct usage: %s <pathToGraph> <numIterations>\n",argv[0]);
        exit(1);
    }
    FILE *fp = fopen(argv[1],"r");
    const int iterations = atoi(argv[2]);
    init_vertices();
    init_edges();
    build_graph(fp);
    create_map();
    init_ranks();
    init_transitions();
/*
    printf("Edges\n");
    for(int i=0;i<edges_length;i++){
        printf("%ld\n",edges[i]);
    }
    
    printf("Transitions\n");
    for(int i=0;i<edges_length;i++){
        printf("%ld %f\n",transitions[i].edge,transitions[i].val);
    }
    
    printf("Vertices\n");
    for(int i=0;i<vertex_length;i++){
        printf("NUmber: %ld\tStart: %ld\tN: %d\n",vertex_list[i].number,vertex_list[i].start,vertex_list[i].n);
    }
    printf("Initial ranks\n");
    for(int i=0;i<vertex_length;i++){
        printf("%f\n",ranks[i]);
    }
*/
    //initializing device memory
    g_type *d_elength;
    check_error(cudaMalloc((void **)&d_elength,sizeof(g_type)));
    check_error(cudaMemcpy(d_elength,&edges_length,sizeof(g_type),cudaMemcpyHostToDevice));

    g_type *d_vlength;
    check_error(cudaMalloc((void **)&d_vlength,sizeof(g_type)));
    check_error(cudaMemcpy(d_vlength,&vertex_length,sizeof(g_type),cudaMemcpyHostToDevice));

    struct vertex *d_vertices;
    check_error(cudaMalloc((void **)&d_vertices,vertex_length * sizeof(struct vertex)));
    check_error(cudaMemcpy(d_vertices,vertex_list,vertex_length * sizeof(struct vertex),cudaMemcpyHostToDevice));

    struct entry *d_transitions;
    check_error(cudaMalloc((void **)&d_transitions,edges_length * sizeof(struct entry)));
    check_error(cudaMemcpy(d_transitions,transitions,edges_length * sizeof(struct entry),cudaMemcpyHostToDevice));

    struct map *d_map;
    check_error(cudaMalloc((void **)&d_map,vertex_length * sizeof(struct map)));
    check_error(cudaMemcpy(d_map,node_map,vertex_length * sizeof(struct map),cudaMemcpyHostToDevice));

    double *d_ranks;
    check_error(cudaMalloc((void **)&d_ranks,vertex_length * sizeof(double)));
    check_error(cudaMemcpy(d_ranks,ranks,vertex_length * sizeof(double),cudaMemcpyHostToDevice));

    double *d_res;
    check_error(cudaMalloc((void **)&d_res,edges_length * sizeof(double)));

    double *d_tempranks;
    check_error(cudaMalloc((void **)&d_tempranks,vertex_length * sizeof(double)));

    //pagerank iterations begin here: Power method

    int blocks = 1;
    int threads = vertex_length;
    if(vertex_length > MAX_THREADS_PER_BLOCK){
        blocks = (int)ceil(vertex_length / (double)MAX_THREADS_PER_BLOCK);
        threads = MAX_THREADS_PER_BLOCK;
    }
    
    int counter = 0;
    while(counter < iterations){
        check_error(cudaMemset(d_res,0.0,edges_length * sizeof(double)));
        check_error(cudaMemset(d_tempranks,0.0,vertex_length * sizeof(double)));
        
        multiply_kernel<<<blocks,threads>>>(d_vertices,d_transitions,d_ranks,d_res,d_vlength);
        cudaDeviceSynchronize();
        add_kernel<<<blocks,threads>>>(d_vertices,d_transitions,d_res,d_map,d_tempranks,d_vlength); 
        cudaDeviceSynchronize();
        update_kernel<<<blocks,threads>>>(d_tempranks,d_ranks,d_vlength);
        counter++;
    }
    //end of pagerank iterations

    double *res;
    res = (double *)malloc(vertex_length * sizeof(double));
    check_error(cudaMemcpy(res,d_ranks,vertex_length * sizeof(double),cudaMemcpyDeviceToHost));
    for(int i = 0;i<vertex_length;i++){
        printf("%lf\n",res[i]);
    }
    free(res);

    //end of device memory initialization
    check_error(cudaFree(d_elength));
    check_error(cudaFree(d_vlength));
    check_error(cudaFree(d_vertices));
    check_error(cudaFree(d_transitions));
    check_error(cudaFree(d_map));
    check_error(cudaFree(d_ranks));
    check_error(cudaFree(d_res));
    check_error(cudaFree(d_tempranks));
    
    delete_edges();
    delete_vertices();
    delete_ranks();
    delete_transitions();
    delete_map();
    return 0;
}


