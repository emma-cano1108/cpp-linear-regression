#include "linear_regression.cpp"
#include "linear_regression.h"
#include <iostream>
#include <vector>
using namespace std;

int main() {
  cout << "Linear Regression" << endl;
  vector<vector<double>> X = {
    {0.5,   20000.0},
    {2.0,   100.0},
    {10.0,  5000.0},
    {40.0,  50.0},
    {150.0, 30000.0},
    {500.0, 800.0},
    {800.0, 12000.0},
    {100.0, 1000.0}
};

vector<double> y = {46.5, 11.2, 45.0, 125.1, 515.0, 1506.6, 2429.0, 307};

    LinearRegression mlr;
    X = mlr.data_scaling(X);
    mlr.fit(X, y);
    cout << "Bias " << mlr.getBias() << endl;
    for(int i = 0; i < mlr.getWeights().size(); i++){
      cout << "W" << i << ": " << mlr.getWeights()[i] << endl;
    }

    vector<double> y_hat = mlr.predict(X);
    for(const double value : y_hat){
      cout << value << endl;
    }

  return 0;
}