#include "linear_regression.cpp"
#include "linear_regression.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

// Loads a CSV file into X and y
void loadData(const string& path, vector<vector<double>>& X, vector<double>& y, vector<string>& headers) {
  ifstream file(path);
  string line;
  getline(file, line);
  stringstream ss(line);
  string cell;
  while (getline(ss, cell, ',')) {
    headers.push_back(cell);
  }
  headers.pop_back();
  while (getline(file, line)) {
    stringstream ss2(line);
    string value;
    vector<double> row;
    while (getline(ss2, value, ',')) {
      row.push_back(stod(value));
    }
    vector<double> x_row(row.begin(), row.end() - 1);
    X.push_back(x_row);
    y.push_back(row.back());
  }
}

int main() {
  cout << "Linear Regression" << endl << endl;
  //Creates an instance of LinearRegression
  LinearRegression mlr;

  //Declaration of independent variable matrix, dependent variable vector and headers vector for ice_cream_selling_data.csv
  vector<vector<double>> X_ice_cream;
  vector<double> y_ice_cream;
  vector<string> headers_ice_cream;

  //By using loadData, stores the whole excel data in X_ice_cream and y_ice_cream, headers are only for print format
  loadData("data/Ice_cream_selling_data.csv", X_ice_cream, y_ice_cream, headers_ice_cream);

  //Shuffles, scales and trains the data. Bias and each weight value will slightly change in each execution because of data shuffle
  mlr.shuffleData(X_ice_cream, y_ice_cream);
  X_ice_cream = mlr.data_scaling(X_ice_cream);
  mlr.fit(X_ice_cream, y_ice_cream);

  //Prints the obtained linear regression model on its algebraic form (y= B0 + w1x1...)
  cout << "Ice Cream" << endl;
  cout << "y = " << mlr.getBias();
  for (int i = 0; i < (int)mlr.getWeights().size(); i++) {
    //Prints - instead of + if weight is negative
    if(mlr.getWeights()[i] < 0){
      cout << " - " << mlr.getWeights()[i]*-1 << " * " << headers_ice_cream[i];
      continue;
    }
    cout << " + " << mlr.getWeights()[i] << " * " << headers_ice_cream[i];

  }

  cout << endl << endl;

  //Runs the predict for X_ice_cream data and prints each y_hat vs real_y_value
  vector<double> y_hat = mlr.predict(X_ice_cream);

  int train_range = floor(X_ice_cream.size() * 0.7);
  for (int i = 0; i < (int)y_hat.size(); i++) {
    cout << "y_hat = " << y_hat[i] << " | y_real = " << y_ice_cream[train_range + i] << endl;
  }

  //Prints scoring value obtained by the y_hat prediction
  cout << "score: " << mlr.score(y_hat, y_ice_cream) << endl << endl;

  //Same process to the multivariable student_exam_scores.csv values
  vector<vector<double>> X_student_exam;
  vector<double> y_student_exam;
  vector<string> headers_student_exam;
  loadData("data/student_exam_scores.csv", X_student_exam, y_student_exam, headers_student_exam);

  mlr.shuffleData(X_student_exam, y_student_exam);
  X_student_exam = mlr.data_scaling(X_student_exam);
  mlr.fit(X_student_exam, y_student_exam);

  cout << "Student Exam" << endl;
  cout << "y = " << mlr.getBias();
  for (int i = 0; i < (int)mlr.getWeights().size(); i++) {
    if(mlr.getWeights()[i] < 0){
      cout << " - " << mlr.getWeights()[i]*-1 << " * " << headers_student_exam[i];
      continue;
    }
    cout << " + " << mlr.getWeights()[i] << " * " << headers_student_exam[i];
  }
  cout << endl << endl;

  y_hat = mlr.predict(X_student_exam);
  train_range = floor(X_student_exam.size() * 0.7);
  for (int i = 0; i < (int)y_hat.size(); i++) {
    cout << "y_hat = " << y_hat[i] << " | y_real = " << y_student_exam[train_range + i] << endl;
  }
  cout << "score: " << mlr.score(y_hat, y_student_exam) << endl;

  return 0;
}
