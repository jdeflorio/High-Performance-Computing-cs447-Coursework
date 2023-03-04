#include <iostream>
#include <vector>
#include <fcntl.h>
#include <assert.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iomanip>
#include <thread>
#include <algorithm>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <math.h>
#include <fstream>

class Reader {
    public:
        Reader(const char *p) : ptr{p} {}
        template <typename T>
        Reader &operator>>(T &o) {
            // Assert alignment.
            assert(uintptr_t(ptr)%sizeof(T) == 0);
            o = *(T *) ptr;
            ptr += sizeof(T);
            return *this;
        }
    private:
        const char *ptr;
};

struct Node
{
    std::vector<float> point;
    Node *left, *right;
    int depth;
};

struct Node* newNode(std::vector<float> p, int dim, int depth, Node* n){
    for (int i=0; i < dim; i++){
        n->point.push_back(p[i]);
    }
    n->left = NULL;
    n->right = NULL;
    n->depth = depth;
    return n;
}

struct Comp {
    Comp(int dimension) {this->dimension = dimension;}
    bool operator ()(std::vector<float> a, std::vector<float> b) const {
        return a[dimension] < b[dimension];
    }
    int dimension;
};

struct Job {
  Node *root;
  std::vector<std::vector<float>> points;
  int depth;

  Job(Node *n, std::vector<std::vector<float>> p, int d) :
    root{n}, points{p}, depth{d} {};
};

struct Result {
    Node *n;
    float dist;

    Result(Node* node, float d):
        n{node}, dist{d} {};
};


std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
split(std::vector<std::vector<float>> v){
    if(v.size() <= 2){
        std::vector<std::vector<float>> left;
        left.push_back(v[0]);
        std::vector<std::vector<float>> right;
        return std::make_pair(left, right);
    }
    std::vector<std::vector<float>> left(v.begin(), v.begin() + v.size() / 2);
    std::vector<std::vector<float>> right((v.begin() + v.size() / 2) + 1, v.end());

    return std::make_pair(left, right);
}

float
e_distance(std::vector<float> p1, std::vector<float> p2, int dims){
    float total = 0;
    for(int i = 0; i < dims; i++){
        total += pow(p1[i] - p2[i], 2);
    }
    float dist = sqrt(total);
    return dist;
}

class KDTree{
    public:
        KDTree(std::vector<std::vector<float>> p, int d, int n) : 
            points(p), dims(d), queue_capacity(n) {
                std::cout << "Making tree with: " << points.size() << " points\n";
                std::sort(points.begin(), points.end(), Comp(0));       //sort on first dim
                root = newNode(points[points.size() / 2], dims, 0, root);         //make median root.
                n_nodes++;

                std::vector<std::vector<float>> left, right;
                std::tie(left, right) = split(points);          //split points

                Node *lroot = new Node;
                Node *rroot = new Node;

                root->left = lroot;
                root->right = rroot;

                work_queue.emplace(lroot, left, 1);     //Add left and right parts of tree to queue at depth == 1
                work_queue.emplace(rroot, right, 1);

                std::vector<std::thread> threads;
                for (int i = 0; i < queue_capacity; i++){
                    threads.push_back(std::thread(&KDTree::insert, this, i));
                }

                for (int i = 0; i < queue_capacity; ++i) {
                    threads[i].join();
                }
            }
        std::vector<std::vector<Result>>
        query_points(std::vector<std::vector<float>> p, int n_neighbors){
            std::vector<std::vector<Result>> final_results;
            std::cout << "Querying: " << p.size() << " points\n";
            for (int i = 0; i < p.size(); i++){
                n_visted = 0;
                std::vector<float> q_point = p[i];      //get current point to find knn

                float distance = e_distance(root->point, q_point, dims); 
                results.clear();
                results.push_back(Result(root, distance));           //add root node/distance as first nn
                n_visted++;

                work_queue.emplace(root->left, p, 1);
                work_queue.emplace(root->right, p, 1);

                std::vector<std::thread> threads;
                for (int i = 0; i < queue_capacity; i++){
                    threads.push_back(std::thread(&KDTree::query, this, i, q_point, n_neighbors));
                }

                for (int i = 0; i < queue_capacity; ++i) {
                    threads[i].join();
                }
                final_results.push_back(results);
            }
            return final_results;
        }
    private:
        Node* root = new Node;
        std::vector<std::vector<float>> points;
        std::vector<Result> results;
        int dims;
        int k;
        std::queue<Job> work_queue;
        std::mutex mutex;
        std::condition_variable cv;
        const size_t queue_capacity;
        int n_nodes = 0;
        int n_visted = 0;
        void insert(int id);
        void query(int id, std::vector<float> q_point, int k);
};

void
KDTree::insert(int id){
    while(true){
        std::unique_lock<std::mutex> work_q_lock(mutex);

        while(work_queue.empty()){
            if(n_nodes == points.size()){
                work_q_lock.unlock();
                cv.notify_all();
                return;
            }
            cv.wait(work_q_lock);
        }
        Job j = work_queue.front();
        work_queue.pop();

        work_q_lock.unlock();
        cv.notify_all();

        std::vector<std::vector<float>> p = j.points;
        Node *rt = j.root;
        int d = j.depth;

        int currentDim = d % dims;
        std::sort(p.begin(), p.end(), Comp(currentDim));       //sort on current dim
        rt = newNode(p[(p.size() / 2)], dims, d, rt);         //make median root

        work_q_lock.lock();
        n_nodes++;
        work_q_lock.unlock();

        if(p.size() == 1){
            continue;
        }
        std::vector<std::vector<float>> left, right;
        std::tie(left, right) = split(p);          //split points

        Node *lroot = new Node;
        Node *rroot = new Node;

        rt->left = lroot;
        rt->right = rroot;

        if(right.size() == 1){
            rt->right = newNode(right[0], dims, d + 1, rt->right);
            work_q_lock.lock();
            n_nodes++;
            work_q_lock.unlock();
        }

        if(left.size() == 1){
            rt->left = newNode(left[0], dims, d + 1, rt->left);
            work_q_lock.lock();
            n_nodes++;
            work_q_lock.unlock();
        }

        work_q_lock.lock(); //lock to add more jobs to queue

        if(left.size() > 1){
            work_queue.emplace(lroot, left, d + 1);     //Add left and right parts of tree to queue
        }
        if(right.size() > 1){
            work_queue.emplace(rroot, right, d + 1);
        }
        work_q_lock.unlock();
    }
}

void
KDTree::query(int id, std::vector<float> q_point, int k){
    while(true){
        std::unique_lock<std::mutex> work_q_lock(mutex);

        while(work_queue.empty()){
            if(n_visted == n_nodes){
                work_q_lock.unlock();
                cv.notify_all();
                return;
            }
            cv.wait(work_q_lock);
        }
        Job j = work_queue.front();
        work_queue.pop();

        work_q_lock.unlock();
        cv.notify_all();

        Node *n = j.root;

        if (n->point.empty()){  //leaf node
            continue;
        }

        float dist = e_distance(n->point, q_point, dims);

        work_q_lock.lock();
        n_visted++;

        if(results.size() == k){
            //replace lowest distance
            int lowest_index = 0;
            int lowest_dist = results[0].dist;
            for(int i = 0; i < results.size(); i++){
                if(results[i].dist < lowest_dist){
                    lowest_dist = results[i].dist;
                    lowest_index = i;
                }
            }
            results[lowest_index] = Result(n, dist);
        } else {
            results.push_back(Result(n, dist));
        }

        if(n->left != NULL){
            work_queue.emplace(n->left, j.points, 0);
        }
        if(n->right != NULL){
            work_queue.emplace(n->right, j.points, 0);
        }
        work_q_lock.unlock();
    }

}

std::vector<std::vector<float>> inputFile(std::string f, uint64_t *dim, uint64_t *k, uint64_t *n_q, uint64_t *tid, uint64_t *qid, std::string type){
  std::vector<std::vector<float>> points;
  int fd = open(f.c_str(), O_RDONLY);
  if (fd < 0) {
      int en = errno;
      std::cerr << "Couldn't open " << f << ": " << strerror(en) << "." << std::endl;
      exit(2);
  }
  struct stat sb;
  int rv = fstat(fd, &sb); assert(rv == 0);
  void *vp = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fd, 0);
  if (vp == MAP_FAILED) {
      int en = errno;
      fprintf(stderr, "mmap() failed: %s\n", strerror(en));
      exit(3);
  }
  char *file_mem = (char *) vp;
  rv = madvise(vp, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(rv == 0);
  rv = close(fd); assert(rv == 0);
  int n = strnlen(file_mem, 8);
  std::string file_type(file_mem, n);
  Reader reader{file_mem + 8};

  if (type == "TRAINING") {
    uint64_t id;
    uint64_t n_points;
    uint64_t n_dims;

    reader >> id >> n_points >> n_dims;

    *dim = n_dims;
    *tid = id;

    for (std::uint64_t i = 0; i < n_points; i++) {
            std::vector<float> point;
            points.push_back(point);
            for (std::uint64_t j = 0; j < n_dims; j++) {
                float f;
                reader >> f;
                points.back().push_back(f);
            }
    }

  } else if (type == "QUERY"){
      uint64_t id;
      uint64_t n_queries;
      uint64_t n_dims;
      uint64_t n_neighbors;

      reader >> id >> n_queries >> n_dims >> n_neighbors;

      *dim = n_dims;
      *k = n_neighbors;
      *n_q = n_queries;
      *qid = id;

      for (std::uint64_t i = 0; i < n_queries; i++) {
          std::vector<float> point;
          points.push_back(point);
          for (std::uint64_t j = 0; j < n_dims; j++) {
              float f;
              reader >> f;
              points.back().push_back(f);
          }
      }
  }
  return points;
}

void
write_result(std::string result_file, uint64_t tid, uint64_t qid, uint64_t n_queries, uint64_t dims, uint64_t k, std::vector<std::vector<Result>> results){
    std::ofstream file (result_file, std::ofstream::binary);

    char str[8] = "RESULT";
    file.write(str, 8);
    file.write((char*) &tid, 8);
    file.write((char*) &qid, 8);

    unsigned char buffer[8];
    int fd = open("/dev/urandom", O_RDONLY);
    read(fd, buffer, 8);

    file.write((char*) &buffer, 8);

    file.write((char*) &n_queries, 8);
    file.write((char*) &dims, 8);
    file.write((char*) &k, 8);

    for(int i = 0; i < results.size(); i++){
        for(int j = 0; j < k; j++){
            for(int l = 0; l < dims; l++){
                file.write((char*) &results[i][j].n->point[l], sizeof(float));
            }
        }
    }
}

int main(int argc, char** argv){
    if(argc != 5){
        std::cerr << "usage: ./k-nn n_cores training_file query_file result_file\n";
        return 1;
    }
    int n_cores = std::stoi(argv[1]);
    std::string training_file = argv[2];
    std::string query_file = argv[3];
    std::string result_file = argv[4];

    std::chrono::duration <double> diff, diff2;

    std::vector<std::vector<float>> training_points, query_points;
    uint64_t dim, k, n_queries, tid, qid;
    training_points = inputFile(training_file, &dim, &k, &n_queries, &tid, &qid, "TRAINING");

    auto start = std::chrono::system_clock::now();
    KDTree tree(training_points, dim, n_cores);
    auto end = std::chrono::system_clock::now();
    diff = end - start;

    std::cout << "Construction Time: " << diff.count() << std::endl;

    query_points = inputFile(query_file, &dim, &k, &n_queries, &tid, &qid, "QUERY");

    start = std::chrono::system_clock::now();
    std::vector<std::vector<Result>> results = tree.query_points(query_points, k);
    end = std::chrono::system_clock::now();
    diff = end - start;

    std::cout << "Query Time: " << diff.count() << std::endl;

    write_result(result_file, tid, qid, n_queries, dim, k, results);

    return 0;
}