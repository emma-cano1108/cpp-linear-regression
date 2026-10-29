#include "linear_regression.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

//Utility functions

// Function for x and y same size validation
bool sizeValidation(vector<vector<double>> &x_inputs,
                    vector<double> &y_outputs) {
  return x_inputs.size() == y_outputs.size();
}

//Function to parse vector<double> to vector<vector<double>> if needed
vector<vector<double>> parseToMatrix(const vector<double>& input_vector) {
  vector<vector<double>> matrix(input_vector.size(), vector<double>(1));

  for (size_t row = 0; row < input_vector.size(); row++) {
    matrix[row][0] = input_vector[row];
  }

  return matrix;
}

//Function to parse vector<vector<double>> (nx1) to vector<double> if needed
vector<double> parseToVector(const vector<vector<double>>& matrix) {
  vector<double> result(matrix.size());

  for (size_t row = 0; row < matrix.size(); row++) {
    if (!matrix[row].empty()) {
      result[row] = matrix[row][0];
    }
  }

  return result;
}

//Matrix Operation Functions

// Function for matrix transposing
vector<vector<double>> transposeMatrix(vector<vector<double>> &matrix) {
  vector<vector<double>> transposed(matrix[0].size(),
                                    vector<double>(matrix.size(), 0.0));

  for (int row = 0; row < matrix.size(); row++) {
    for (int value = 0; value < matrix[row].size(); value++) {
      transposed[value][row] = matrix[row][value];
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
  x_train_inputs = data_scaling(x_train_inputs);

  //"A" matrix. Formed by joining a "1" vector with the x_inputs matrix ([1 |
  // X])
  vector<vector<double>> a_matrix = x_train_inputs;
  for (int row = 0; row < x_train_inputs.size(); row++) {
    // Inserts a 1 at the begin of each row
    a_matrix[row].insert(a_matrix[row].begin(), 1);
  }

  //"Aᵀ" Matrix. Transposed A by using transposeMatrix function
  vector<vector<double>> at_matrix = transposeMatrix(a_matrix);

  //Product matrix of AᵀA
  vector<vector<double>> at_a_product_matrix = matrixProduct(at_matrix, a_matrix);

  //Inverse matrix of AᵀA (AᵀA)⁻¹
  vector<vector<double>> at_a_product_inverse_matrix = inverseMatrix(at_a_product_matrix);
  
  //Matrix-Vector product for Aᵀy (y= y_train_outputs). Returns a nx1 vector
  vector<vector<double>> parsed_y_train_outputs = parseToMatrix(y_train_outputs);
  vector<vector<double>> at_y_product_vector = matrixProduct(at_matrix, parsed_y_train_outputs);

  //Weights vector: Full (AᵀA)⁻¹Aᵀy product
  vector<vector<double>> not_parsed_weights = matrixProduct(at_a_product_inverse_matrix, at_y_product_vector);
  weights = parseToVector(not_parsed_weights);
  //Bias: First entry of weights vector
  bias = weights[0];
}

vector<vector<double>> LinearRegression::data_scaling(vector<vector<double>>& x_inputs){

  //Checks if it is really necessary to scale by dynamic range evaluation (max range / min range)

  //Gets the transpose for x_inputs to compute easily
  vector<vector<double>> x_t_inputs = transposeMatrix(x_inputs);

  vector<double> ranges = {};

  for(int row = 0; row < x_t_inputs.size(); row++){
    //Gets the max and min value for each variable (xₙ) and store its substraction (range) in ranges vector
    double max = *max_element(x_t_inputs[row].begin(), x_t_inputs[row].end());
    double min = *min_element(x_t_inputs[row].begin(), x_t_inputs[row].end());
    ranges.push_back(max - min);
  }
  //Gets the max and min range stored.
  double max_range = *max_element(ranges.begin(), ranges.end());
  double min_range = *min_element(ranges.begin(), ranges.end());

  //If (max range / min range) >= 500, normalization will be applied. Otherwise, same matrix will be returned.
  //To have in account: if x_inputs is an nx1 matrix (one variable), normalization will not be applied
  if(max_range/min_range <= 500){
    return x_inputs;
  }

  //Z-Score normaliztion algorithm

  //EACH FUNCTION NEEDS TO BE APPLIED TO THE TRANSPOSED MATRIX, NOT THE ORIGINAL


  //Arithmetic Average "μ"
  double accumulated_sum = 0;
  vector<double> sorted_averages = {};

  //Stores the average of each xₙ in a vector
  for(int row = 0; row < x_t_inputs.size(); row++){
    for(int value = 0; value < x_t_inputs[row].size(); value++){
      accumulated_sum += x_t_inputs[row][value];
    }
    //Average for current variable
    sorted_averages.push_back(accumulated_sum/x_t_inputs[row].size());
    accumulated_sum = 0;
  }

  for(int i = 0; i < sorted_averages.size(); i++){
    cout<< "Average " << i << ": " << sorted_averages[i] << endl;
  }

  //Standard deviation "σ". Calculated with population formula
  vector<double> sorted_standard_deviations = {};

  for(int row = 0; row < x_t_inputs.size(); row++){
    for(int value = 0; value < x_t_inputs[row].size(); value++){
      accumulated_sum += pow((x_t_inputs[row][value] - sorted_averages[row]), 2);
    }
    //Standard deviation for current variable
    sorted_standard_deviations.push_back(sqrt(accumulated_sum/x_t_inputs[row].size()));
    accumulated_sum = 0;
  }

  for(int i = 0; i < sorted_averages.size(); i++){
    cout<< "Standard Deviation " << i << ": " << sorted_standard_deviations[i] << endl;
  }

  //"Z" scaled value computing and scaled matrix return

  //Initializes an empty matrix with same size as x_inputs
  vector<vector<double>> scaled_matrix(x_t_inputs.size(), vector<double>(x_t_inputs[0].size(), 0.0));

  double z_value = 0;
  
  //Gets Z = (x - μ)/σ
  for(int row = 0; row < x_t_inputs.size(); row++){
    for(int value = 0; value < x_t_inputs[row].size(); value++){
      z_value = (x_t_inputs[row][value] - sorted_averages[row])/sorted_standard_deviations[row];
      //Stores scaled value in scaled_matrix
      scaled_matrix[row][value] = z_value;
    }
  }
  //Returns transposed scaled matrix to inverse the transpose made at the begin.
  return transposeMatrix(scaled_matrix);
}


//Getters
vector<double> LinearRegression::getWeights(){
  return weights;
}

double LinearRegression::getBias(){
  return bias;
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