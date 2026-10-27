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
// Function for matrix transposing
vector<vector<double>> transposeMatrix(vector<vector<double>> &matrix) {
  vector<vector<double>> transposed(matrix.size(),
                                    vector<double>(matrix[0].size(), 0.0));

  for (int row = 0; row < matrix.size(); row++) {
    for (int value = 0; value < matrix[row].size(); value++) {
      // Replaces xₙₘ value with xₘₙ value.
      transposed[row][value] = matrix[value][row];
    }
  }
  return transposed;
}

// Function for matrix product
vector<vector<double>> matrixProduct(vector<vector<double>>& left_matrix, vector<vector<double>>& right_matrix) {
  //Validation of Left matrix columns number = Right matrix rows number
  if (left_matrix[0].size() != right_matrix.size()) {
    cout << "Left matrix columns number does not match with right matrix rows "
            "number. Can not do the matrix product. Empty matrix will be "
            "returned"
         << endl;
    return {};
  }

  vector<vector<double>> matrix_product(
      left_matrix.size(), vector<double>(right_matrix[0].size(), 0.0));
  double current_value = 0;

  //Matrix product algorithm.
  for (int iteration = 0; iteration < left_matrix.size(); iteration++) {
    for (int row = 0; row < right_matrix[0].size(); row++) {
      double current_value = 0.0;
      for (int value = 0; value < left_matrix[0].size(); value++) {
        current_value += left_matrix[iteration][value] * right_matrix[value][row];
      }
      matrix_product[iteration][row] = current_value;
    }
  }
  return matrix_product;
}

//Function for inverse matrix
vector<vector<double>> inverseMatrix(vector<vector<double>>& matrix) {
  //Validation for square matrix
  if (matrix.size() != matrix[0].size()) {
    cout << "Matrix is not square. Can not invert the matrix. Empty matrix will "
            "be returned"
         << endl;
    return {};
  }

  int matrix_size = matrix.size();

  //Creation of [A | I] matrix for Gauss-Jordan reduction.
  vector<vector<double>> augmented_matrix(matrix_size, vector<double>(2 * matrix_size, 0.0));

  for (int row = 0; row < matrix_size; row++) {
    for (int col = 0; col < matrix_size; col++) {
      augmented_matrix[row][col] = matrix[row][col];
    }
    augmented_matrix[row][row + matrix_size] = 1.0;
  }

  //Gauss-Jordan reduction algorithm.
  for (int iteration = 0; iteration < matrix_size; iteration++) {
    int max_row = iteration;
    for (int row = iteration + 1; row < matrix_size; row++) {
      if (abs(augmented_matrix[row][iteration]) > abs(augmented_matrix[max_row][iteration])) {
        max_row = row;
      }
    }

    swap(augmented_matrix[iteration], augmented_matrix[max_row]);
    //Validation for det(A) = 0. Using 1e-12 instead of 0 to avoid possible rounding errors.
    if (abs(augmented_matrix[iteration][iteration]) < 1e-12) {
      cout << "Matrix is singular. Can not invert the matrix. Empty matrix will "
              "be returned"
           << endl;
      return {};
    }

    double pivot = augmented_matrix[iteration][iteration];
    for (int col = 0; col < 2 * matrix_size; col++) {
      augmented_matrix[iteration][col] /= pivot;
    }

    for (int row = 0; row < matrix_size; row++) {
      if (row != iteration) {
        double factor = augmented_matrix[row][iteration];
        for (int col = 0; col < 2 * matrix_size; col++) {
          augmented_matrix[row][col] -= factor * augmented_matrix[iteration][col];
        }
      }
    }
  }
  
  //Declaration of inverse_matrix with the same size as the original to save the result
  vector<vector<double>> inverse_matrix(matrix_size, vector<double>(matrix_size, 0.0));
  for (int row = 0; row < matrix_size; row++) {
    for (int col = 0; col < matrix_size; col++) {
      inverse_matrix[row][col] = augmented_matrix[row][col + matrix_size];
    }
  }

  return inverse_matrix;
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

  //"A" matrix. Formed by joining a "1" vector with the x_inputs matrix ([1 |
  // X])
  vector<vector<double>> a_matrix = x_train_inputs;
  for (int row = 0; row < x_train_inputs.size(); row++) {
    // Inserts a 1 at the begin of each row
    a_matrix[row].insert(a_matrix[row].begin(), 1);
  }

  //"Aᵀ" Matrix. Transposed A by using transposeMatrix function.
  vector<vector<double>> at_matrix = transposeMatrix(a_matrix);

  //Product matrix of AᵀA
  vector<vector<double>> at_a_product_matrix = matrixProduct(a_matrix, at_matrix);

  //Inverse matrix of AᵀA (AᵀA)⁻¹
  vector<vector<double>> at_a_product_inverse_matrix = inverseMatrix(at_a_product_matrix);
}

// Displays the arguments data in cell format: | x0 | x1 | ... | xn | y |
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
