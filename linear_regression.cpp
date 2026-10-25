#include "linear_regression.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
using namespace std;

// Function for x and y same size validation
bool sizeValidation(vector<vector<double>> &x_inputs,
                    vector<double> &y_outputs) {
  return x_inputs.size() == y_outputs.size();
}

void LinearRegression::fit(vector<vector<double>> &x_inputs,
                           vector<double> &y_outputs) {
  // Size validation for x_inputs and y_outputs rows number. If not equal,
  // finishes the method execution.
  if (!sizeValidation(x_inputs, y_outputs)) {
    cout
        << "x inputs and y outputs size does not match. Can not train the model"
        << endl;
    return;
  }

  vector<vector<double>> x_train_inputs;
  vector<double> y_train_outputs;

  // Training values range: floors the value of the 70% of the data
  int train_range = floor(x_inputs.size() * 0.7);

  // Assigns the values for the training matrix and vector.
  for (int row = 0; row < train_range; row++) {
    x_train_inputs.push_back(x_inputs[row]);
    y_train_outputs.push_back(y_outputs[row]);
  }

  //"A" matrix. Formed by joining a "1" vector with the x_inputs matrix ([1 | X])
  vector<vector<double>> a_matrix = x_train_inputs;
  for (int row = 0; row < x_train_inputs.size(); row++) {
    //Inserts a 1 at the begin of each row
    a_matrix[row].insert(a_matrix[row].begin(), 1);
  }

}

//Displays the arguments data in cell format: | x0 | x1 | ... | xn | y |
void LinearRegression::displayData(vector<vector<double>> &x_inputs,
                                   vector<double> &y_outputs) {
  if (!sizeValidation(x_inputs, y_outputs)) {
    cout << "x inputs and y outputs size does not match. Can not display the "
            "data"
         << endl;
    return;
  }
  const int w = 10;
  cout << fixed << setprecision(4) << "Data:" << endl;

  for (size_t i = 0; i < x_inputs[0].size(); i++)
    cout << "|" << setw(w) << "x" + to_string(i);
  cout << "|" << setw(w) << "y" << "|" << endl
       << string((x_inputs[0].size() + 1) * (w + 1) + 1, '-') << endl;

  for (size_t r = 0; r < x_inputs.size(); r++) {
    for (double v : x_inputs[r])
      cout << "|" << setw(w) << v;
    cout << "|" << setw(w) << y_outputs[r] << "|" << endl;
  }
}
