#include "linear_regression.cpp"
#include "linear_regression.h"
#include <iostream>
#include <vector>
using namespace std;

int main() {
  cout << "Linear Regression" << endl;
  vector<vector<double>> X = {
        {1.0, 2.0},
        {2.0, 1.0},
        {3.0, 4.0},
        {4.0, 3.0},
        {5.0, 5.0}
    };
  vector<double> y = {
        13.1, 
        12.0, 
        23.2,
        22.1, 
        30.0  
    };

    LinearRegression mlr;
    mlr.fit(X, y);

  return 0;
}