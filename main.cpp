#include "linear_regression.cpp"
#include "linear_regression.h"
#include <iostream>
#include <vector>
using namespace std;

int main() {
  cout << "Linear Regression" << endl;
  vector<vector<double>> X = {
    {2.0,  7.0,  1.0},
    {5.0,  3.0,  8.0},
    {1.0,  9.0,  4.0},
    {8.0,  2.0,  6.0},
    {3.0,  5.0,  9.0},
    {9.0,  8.0,  2.0},
    {4.0,  1.0,  7.0},
    {6.0,  6.0,  5.0},
    {7.0,  4.0,  3.0},
    {10.0, 10.0, 10.0},
};

  vector<double> y = {-7.5, 8.5, -11.5, 14.0, 2.0, 1.5, 10.5, 2.5, 7.0, 3.0};

    LinearRegression mlr;
    X = mlr.data_scaling(X);
    mlr.fit(X, y);
    cout << "Bias " << mlr.getBias() << endl;
    for(int i = 0; i < mlr.getWeights().size(); i++){
      cout << "W" << i << ": " << mlr.getWeights()[i] << endl;
    }

    vector<double> y_hat = mlr.predict(X);

    cout << "score: " << mlr.score(y_hat, y);

  return 0;
}