#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <random>
#include <thread>
#include <future>

double function(double x){
  return sin(x)/x;
}

double integrate(double a, double b, int n){
  double total = 0;

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<double> distr(a, b);

  for (int i = 0; i < n; i++){
    double rand = distr(eng);
    total += function(rand);
  }

  return total;
}

int main(int argc, char** argv){
  if(argc != 5){
    std::cerr << "usage: ./integrate a b n n_threads\n";
    return 1;
  }

  double a = std::stod(argv[1], NULL);
  double b = std::stod(argv[2], NULL);

  long long n = std::stoll(argv[3], NULL);
  int n_threads = std::stoi(argv[4], NULL);

  std::future<double> tArray[n_threads];
  for (int i = 0; i < n_threads; i++){
    tArray[i] = std::async(integrate, a, b, n/n_threads);
  }

  double total = 0;
  for(int i = 0; i < n_threads; i++){
    tArray[i].wait();
    total += tArray[i].get();
  }

  double monteCarloEst = (b - a)*total/n;
  std::cout << monteCarloEst << '\n';
  
  return 0;
}