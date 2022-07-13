#include <time.h>
#include <stdlib.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjdata.h>
#include <engine/engine_macro.h>
#include <engine/engine_util_blas.h>

typedef enum kmeansResult
{
  KMEANS_SUCCESS,
  KMEANS_ERROR,
} kmeansResult;

typedef struct kmeansCluster
{
  // Cluster
  unsigned int k;  // number of cluster
  mjtNum *centers; // center of cluster
  // Points
  int pnt_dim;      // dimension of each point
  int pnt_N;        // number of points
  mjtNum *pnt_loc;  // size (dim*N)
  int *pnt_cluster; // the cluster correspond to point i in points
  // Results
  kmeansResult result;

} kmeansCluster;

typedef struct kmeansConfig
{
  int max_ite;
  // mjtNum *weights;
} kmeansConfig;

// distance function
// typedef void (*distance_fn)(mjtNum* vec1, mjtNum* vec2, int dim);

// TODO
static int
label_from_centers(kmeansCluster *cluster)
{
  int changed = 0;
  for (size_t i = 0; i < cluster->pnt_N; i++)
  {
    int closest = -1;
    mjtNum closest_dist = mjMAXVAL;
    for (size_t j = 0; j < cluster->k; j++)
    {
      // mjtNum dist=mju_sub
      mjtNum vec[cluster->pnt_dim];
      mju_sub(vec, cluster->centers + j * cluster->pnt_dim,
              cluster->pnt_loc + i * cluster->pnt_dim, cluster->pnt_dim);
      // // weight
      // for (size_t k = 0; k < cluster->pnt_dim; k++)
      // {
      //   vec[k]*=weights[k];
      // }
      
      mjtNum dist = mju_norm(vec, cluster->pnt_dim);
      if (dist < closest_dist)
      {
        closest_dist = dist;
        closest = j;
      }
    }
    if (closest != cluster->pnt_cluster[i])
      changed = 1;
    cluster->pnt_cluster[i] = closest;
  }
  return changed;
}

static void centers_from_label(kmeansCluster *cluster)
{
  // clear value
  mju_zero(cluster->centers, cluster->pnt_dim * cluster->k);

  mjtNum denom[cluster->k];
  mju_zero(denom, cluster->k);
  for (size_t i = 0; i < cluster->pnt_N; i++)
  {
    mju_addTo(cluster->centers + cluster->pnt_cluster[i] * cluster->pnt_dim,
              cluster->pnt_loc + i * cluster->pnt_dim, cluster->pnt_dim);
    denom[cluster->pnt_cluster[i]] += 1;
  }
  for (size_t i = 0; i < cluster->k; i++)
  {
    if (denom[i] == 0)
    {
      // set to a random point
      int random_idx = rand() % cluster->pnt_N;
      mju_copy(cluster->centers + i * cluster->pnt_dim,
               cluster->pnt_loc + random_idx * cluster->pnt_dim, cluster->pnt_dim);
    }
    else
    {
      mju_scl(cluster->centers + i * cluster->pnt_dim,
              cluster->centers + i * cluster->pnt_dim,
              1 / denom[i], cluster->pnt_dim);
    }
  }
}

// ------------------- utilities
void printf_3d(const mjtNum *vec)
{
  printf("[%.2f, %.2f, %.2f]\n", vec[0], vec[1], vec[2]);
}

void print_array_int(const int *arr, int size, const char *name)
{
  printf("array %s =[", name);
  for (size_t i = 0; i < size; i++)
  {
    printf("%d, ", arr[i]);
  }
  printf("]\n");
}

void print_array_mjtnum(const mjtNum *arr, int size, const char *name)
{
  printf("array %s =[", name);
  for (size_t i = 0; i < size; i++)
  {
    printf("%.2f, ", arr[i]);
  }
  printf("]\n");
}

// initialize cluster from a list of points
// void initialize_cluster(kmeansCluster cluster, int k, int num_points, const mjtNum *points)
// {
//   cluster.k=k;
//   cluster.ptn_N=num_points;

// }

// add points to existing cluster
// void add_points(kmeansCluster cluster, const mjtNum *point) {}

void mj_getDefaultKmeansConfig(kmeansConfig *config, const kmeansCluster *cluster)
{
  config->max_ite = 10;
  // for (size_t i = 0; i < cluster->pnt_dim; i++)
  // {
  //   config->weights[i] = 1;
  // }
}

// perform kmeans_clustering
// the init cluster should be initialized via `initialize_cluster`
// weight is a vector with same dimension as cluster->pnt_dim used to calculate distance
// between two points in cluster
// augment p' = [weight[0]*p[0], weight[1]*p[1], ... ]
// TODO initialize cluster?
void mj_kmeans(kmeansCluster *cluster, const kmeansConfig* config)
{
  // NOTE verification, print out all points in cluster
  // for (size_t i = 0; i < cluster->pnt_N; i++)
  // {
  //   print_array_mjtnum(cluster->pnt_loc + i*cluster->pnt_dim, cluster->pnt_dim, "cluster point");
  // }
  

  if (cluster->k >= cluster->pnt_N)
  {
    cluster->result = KMEANS_ERROR;
    return;
  }

  for (size_t i = 0; i < config->max_ite; i++)
  {
    if (!label_from_centers(cluster))
    {
      cluster->result = KMEANS_SUCCESS;
      // printf("Kmeans terminate successfully after %d iterations\n", i);
      return;
    }
    centers_from_label(cluster);
    // NOTE verification
    // printf("Iteration %d, Centers: \n", i);
    // for (size_t j = 0; j < cluster->k; j++)
    // {
    //   char name[50];
    //   snprintf(name, 50, "center%d", j);
    //   print_array_mjtnum(cluster->centers + j * cluster->pnt_dim, cluster->pnt_dim, name);
    // }
    // print_array_int(cluster->pnt_cluster, cluster->pnt_N, "pnt_cluster");
  }
  cluster->result = KMEANS_SUCCESS;
}

void mj_initializeKmeansRandom(mjData *d, kmeansCluster *cluster)
{
  srand(time(0));
  // create identity array [0,1,..., N]
  int N = cluster->pnt_N;
  int identity_arr[N];
  for (size_t i = 0; i < N; i++)
  {
    identity_arr[i] = i;
  }
  // permutation
  for (size_t i = 0; i < N; i++)
  {
    int k = i + rand() % (N - i);
    // swap
    int temp = identity_arr[i];
    identity_arr[i] = identity_arr[k];
    identity_arr[k] = temp;
  }

  // NOTE printout permutation result
  // print_array_int(identity_arr, N, "identity_arr");

  // TODO better resolution when N<cluster->k
  for (size_t i = 0; i < cluster->k; i++)
  {
    mju_copy(cluster->centers + i * cluster->pnt_dim,
             cluster->pnt_loc + identity_arr[i] * cluster->pnt_dim,
             cluster->pnt_dim);
  }
}

// TODO need imrpove
void mj_initializeKmeansPP(mjData *d, kmeansCluster *cluster)
{
  mjMARKSTACK;
  int N = cluster->pnt_N;
  int dim = cluster->pnt_dim;
  int k = cluster->k;
  // first sort contact normals based on dot product
  mjtNum *sorted_contact_normals = mj_stackAlloc(d, N * dim);
  mju_copy(sorted_contact_normals, cluster->pnt_loc, N * dim);
  for (size_t i = 1; i < N; i++)
  {
    // TODO generalize this to a dist function
    mjtNum max_dot = mju_abs(mju_dot(sorted_contact_normals + (i - 1) * dim,
                                     sorted_contact_normals + i * dim, dim));
    int id = i;
    for (size_t j = i + 1; j < N; j++)
    {
      mjtNum dot = mju_abs(mju_dot(sorted_contact_normals + (i - 1) * dim,
                                   sorted_contact_normals + j * dim, dim));
      if (dot > max_dot)
      {
        max_dot = dot;
        id = j;
      }
    }
    // swap
    mjtNum temp[dim];
    mju_copy(temp, sorted_contact_normals + id * dim, dim);
    mju_copy(sorted_contact_normals + id * dim, sorted_contact_normals + i * dim, dim);
    mju_copy(sorted_contact_normals + i * dim, temp, dim);
  }
  // NOTE verification
  // printf("sorted_contact_normal = [\n");
  // for (size_t i = 0; i < ncon; i++)
  // {
  //   printf_3d(sorted_contact_normals+i*3);
  // }
  // printf("]\n");
  // printf("cluter points %d, no clusters %d", N, k);
  for (size_t i = 0; i < k; i++)
  {
    int id = (i * N) / (k);
    mju_copy(cluster->centers + i * dim,
             sorted_contact_normals + id * dim, dim);
  }
  // NOTE verification
  // printf("initialized cluster = [\n");
  // for (size_t i = 0; i < k; i++)
  // {
  //   printf_3d(cluster->centers + i * dim);
  // }
  mjFREESTACK;
}
