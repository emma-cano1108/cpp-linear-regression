#include "linear_regression.cpp"
#include "linear_regression.h"
#include <iostream>
#include <vector>
using namespace std;

int main() {
  cout << "Linear Regression" << endl;
  vector<vector<double>> X = {
    {1.0, 2.0, 3.0},
    {2.0, 5.0, 1.0},
    {3.0, 1.0, 4.0},
    {4.0, 4.0, 2.0},
    {5.0, 3.0, 5.0},
    {6.0, 6.0, 3.0},
    {7.0, 2.0, 7.0}
};

  vector<double> y = {5.5, 3.5, 11.0, 9.0, 13.5, 11.5, 19.0};

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