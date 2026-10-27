#include "linear_regression.cpp"
#include "linear_regression.h"
#include <iostream>
#include <vector>
using namespace std;

int main() {
  cout << "Linear Regression" << endl;
  vector<vector<double>> X = {
    {1.0},
    {2.5},
    {3.0},
    {4.5},
    {5.0},
    {6.5},
    {7.0},
    {8.5},
    {9.0},
    {11.0}
};

vector<double> y = {
    15.2,
    21.8,
    24.5,
    31.0,
    33.8,
    41.2,
    43.0,
    50.5,
    53.1,
    62.0
};

    LinearRegression mlr;
    mlr.fit(X, y);
    cout << "Bias " << mlr.getBias() << endl;
    for(int i = 1; i < mlr.getWeights().size(); i++){
      cout << "W" << i << ": " << mlr.getWeights()[i] << endl;
    }

  return 0;
}