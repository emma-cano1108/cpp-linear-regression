#include "linear_regression.cpp"
#include "linear_regression.h"
#include <iostream>
#include <vector>
using namespace std;

int main() {
  cout << "Linear Regression" << endl;
  vector<vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};

vector<double> y = {2.2, 4.1, 6.2, 7.9, 10.1};

    LinearRegression mlr;
    mlr.fit(X, y);
    cout << "Bias " << mlr.getBias() << endl;
    for(int i = 0; i < mlr.getWeights().size(); i++){
      cout << "W" << i << ": " << mlr.getWeights()[i] << endl;
    }

    mlr.predict(X);

  return 0;
}