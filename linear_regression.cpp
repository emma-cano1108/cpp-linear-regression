#include <iostream>
#include <cmath>
#include "linear_regression.h"
using namespace std;


void LinearRegression::fit(vector<vector<double>>& x_inputs, vector<double>& y_outputs){
    //Size validation for x_inputs and y_outputs rows number. If not equal, finishes the method execution.
    if(x_inputs.size() != y_outputs.size()){
        cout << "x inputs and y outputs size not match. Can not train the model" << endl;
        return;
    }

    vector<vector<double>> x_train_inputs;
    vector<double> y_train_outputs;

    //Training values range: floors the value of the 70% of the data
    int train_range = floor(x_inputs.size()*0.7);

    //Assign the values for the training matrix and vector.
    for(int row = 0; row < train_range; row++){
        x_train_inputs.push_back(x_inputs[row]);
        y_train_outputs.push_back(y_outputs[row]);
    }
    

    
}

