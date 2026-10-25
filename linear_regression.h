#ifndef _LINEAR_REGRESSION_
#define _LINEAR_REGRESSION_
#include <vector>
using namespace std;

class LinearRegression{
    private:
        //Ideal weights vector
        vector<double> weights;
        //Ideal bias (intercept with y axis) value.
        double bias = 0;
    
    public:
        //Default empty Constructor.
        LinearRegression() = default;

        //m = number of samples (rows) and n = number of independent variables (columns)
        //Each method will separate its sample percentage for fit (70%) and scoring (30%)

        //Train method: Receives x_inputs (m x n matrix) and y_outputs (m x 1 vector)
        void fit(vector<vector<double>>& x_inputs, vector<double>& y_outputs);

        //Predict method: Receives x_inputs (m x n matrix) and returns prediction vector (ŷ)
        vector<double> predict(vector<vector<double>> x_inputs);

        //Scoring method: Receives prediction vector (ŷ) and returns the score value
        double score(vector<double> y_hat);

        //Data scaling method: Receives x_inputs (m x n matrix), and returns the scaled matrix
        vector<vector<double>> data_scaling(vector<vector<double>> x_inputs);

        //Displays data in cells format
        void displayData(vector<vector<double>>& x_inputs, vector<double>& y_outputs);
        //Getters
        vector<double> getWeights();
        double getBias();



};


#endif