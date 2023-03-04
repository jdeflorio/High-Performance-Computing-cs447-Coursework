#include <iostream>
#include <omp.h>
#include <random>
#include <vector>
#include <math.h>
#include <string>
#include <iomanip>

std::vector<int> make_histogram(std::vector<float> d){
  std::vector<int> hist(100, 0);

  for(int i = 0; i < d.size(); i++){
    int index = ceil(d[i] * 100) - 1;
    hist[index] += 1; 
  }

  return hist;
}

int main(int argc, char** argv){
  
  int n_points = 5000;
  int max_dims = 16;
  int min_dims = 2;

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<float> distr(-1.0, 1.0);

  std::vector<std::vector<float>> points(n_points, std::vector<float>(max_dims, 0));

  for (int dim = min_dims; dim <= max_dims; dim++){
    std::vector<float> distances(n_points, 0);

    std::vector<std::vector<float>> hists;

    #pragma omp parallel for private(distr, eng) shared(hists, distances, points)
    for (int j = 0; j < n_points; j++){
      float distance_from_origin = 0;
      bool rejected = true;
      while(rejected){       //keep looping until rand point is inside of sphere
        float total = 0;
        for (int k = 0; k < dim; k++){     //create dim amount of random floats for a point
          points[j][k] = distr(eng);
          total += points[j][k]*points[j][k];
        }
        distance_from_origin = sqrt(total);       //get distance from origin
        if (distance_from_origin <= 1){
          rejected = false;
        }
      }

      float distance = 1 - distance_from_origin;                //get distance from surface
      distances[j] = distance;
    }

    std::vector<int> hist = make_histogram(distances);

    std::cout << "-----For " << std::to_string(dim) << " Dims-----\n";
    float interval = 0.00;    
    for (int j = 0; j < hist.size(); j++){
      std::cout << std::setprecision(2) << interval << " to " << std::setprecision(2) << (interval + 0.01) << ": " << hist[j] << std::endl;
      interval += 0.01;
    }
  }

  return 0;
}