# Linear Regression (C++) - Languages and Paradigms of Computation at EAFIT University

A from-scratch C++ implementation of multiple linear regression using the **normal equation** (closed-form solution) with Gauss–Jordan matrix inversion. It trains on numeric CSV data, applies optional Z-score normalization, evaluates with mean squared error, and prints the resulting model equation.

## Features

- Multiple linear regression via the normal equation: w = (AᵀA)⁻¹Aᵀy
- Bias (intercept) and per-feature weights learned from data
- Z-score normalization, gated by a dynamic-range threshold
- 70/30 train–test split with randomized row shuffling before training
- Prediction and mean squared error scoring on the final 30% of the loaded data
- Matrix toolkit: transpose, product, and inverse (Gauss–Jordan)
- Two included datasets: univariate (ice-cream sales) and multivariate (student exam scores)

## Project structure

```text
cpp-linear-regression/
├── main.cpp                    # Entry point: loads data, trains, predicts, scores
├── linear_regression.h         # LinearRegression class declaration
├── linear_regression.cpp       # Class implementation and matrix utilities
├── data/
│   ├── Ice_cream_selling_data.csv   # Univariate: Temperature → Ice Cream Sales
│   └── student_exam_scores.csv      # Multivariate: 4 features → exam score
```

## How it works

### Formulas

**Normal equation (training)**

The model is solved in closed form:

w = (AᵀA)⁻¹Aᵀy

where A is the training matrix augmented with a leading column of `1`s (for the bias `b`), and w = [b, w₁, …, wₙ]ᵀ. The inverse (AᵀA)⁻¹ is computed with Gauss–Jordan elimination.

**Prediction**

ŷ = b + w₁x₁ + w₂x₂ + … + wₙxₙ

**Z-score normalization**

Each feature is standardized as:

z = (x − μ) / σ

where:

μ = arithmetic average

σ = population standard deviation:

σ = √( Σ(x − μ)² / m )

Normalization only runs when the dynamic-range condition holds:

max(range) / min(range) >= 500

where range = max − min per feature. Otherwise the matrix is returned unchanged. For a single-variable matrix, normalization is skipped.

**Score (Mean Squared Error)**

MSE = (1/m) Σ(yᵢ − ŷᵢ)²

computed on the final 30% of the loaded data (the last rows of the array as loaded, independent of the training shuffle). Extremely small scores (< 1e-20) are reported as 0.

### Execution flow

`main.cpp` runs the same pipeline for each dataset:

1. Instantiate the class: `LinearRegression mlr;`
2. Load data (`loadData`): parse the CSV into feature matrix `X`, target vector `y`, and column headers.
3. Scale data (`data_scaling`): apply Z-score normalization if needed.
4. Train (`fit`): shuffle rows, reserve 70% for training, solve the normal equation, and store `bias` and `weights`.
5. Print the model: displays `y = b + w₁x₁ + …` using the header names.
6. Predict (`predict`): compute ŷ for the remaining 30% and print each prediction next to the real value.
7. Score (`score`): print the MSE.

This is the recommended execution flow to obtain better results — `data_scaling` should precede `fit`

**To have in account:** Bias, weights and score will vary slightly between executions


## Build and run

The project uses only the C++ standard library. Compile with any C++11 (or newer) compiler:

```bash
g++ -std=c++11 main.cpp -o linear_regression
```

*`main.cpp` already `#include`s `linear_regression.cpp`, so compiling `main.cpp` alone is sufficient.*

Run from the repository root so the relative `data/` paths resolve:

```bash
./linear_regression
```

On Windows:

```bash
linear_regression.exe
```

## Datasets

| Dataset | Variables | Target | Rows |
|---------|-----------|--------|------|
| `Ice_cream_selling_data.csv` | Temperature (°C) | Ice Cream Sales (units) | 49 |
| `student_exam_scores.csv` | hours_studied, sleep_hours, attendance_percent, previous_scores | exam_score | 200 |

Both CSVs expect a header row followed by numeric rows, with the target in the last column.

## Expected results


### Ice cream sales (univariate)

| Metric | Possible observed range |
|--------|----------------|
| Bias | ~13.8 to ~17.7 |
| Weight `Temperature (°C)` | ~-1.75 to ~-0.30 |
| Score (MSE) | ~126 to ~242 |


### Student exam scores (multivariate)

| Metric | Possible Observed range |
|--------|----------------|
| Bias | ~-3.0 to ~1.0 |
| Weight `hours_studied` | ~1.51 to ~1.61 |
| Weight `sleep_hours` | ~0.79 to ~1.03 |
| Weight `attendance_percent` | ~0.09 to ~0.12 |
| Weight `previous_scores` | ~0.16 to ~0.18 |
| Score (MSE) | ~8.15 to ~8.57 |

**The MSE is comparatively stable across runs because the dataset is larger and less sensitive to the shuffle.**

## Limitations and assumptions


- **Singular matrices.** If `AᵀA` is not invertible (e.g., collinear or duplicate features), a warning message is printed and training is aborted for that dataset.

### To have in account:
**Numeric CSVs only.** `loadData` parses every cell with `stod`; non-numeric values will fail at runtime.


