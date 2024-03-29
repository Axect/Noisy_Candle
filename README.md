# Noisy Regression with Candle

This project demonstrates how to perform regression on noisy data using various Rust libraries for machine learning. The main libraries used in this project are:

- `peroxide`: A library for data generation, preprocessing and visualization
- `candle`: A library for defining and training machine learning models
- `rayon`: A library for parallel processing
- `indicatif`: A library for progress tracking and visualization

## Key Features

1. Dataset Generation and Preprocessing
   - Generate noisy data using the `peroxide` library.
   - Split the data into train, validation, and test sets.
   - Scale the data using scalers such as `MinMaxScaler`, `StandardScaler`, and `RobustScaler`.

2. Model Definition and Training
   - Define an MLP (Multi-Layer Perceptron) model using the `candle` library.
   - Train the model using the Adam optimizer.
   - Track the training progress using the `indicatif` library.

3. Model Evaluation and Visualization
   - Evaluate the trained model on the test dataset and calculate the MSE.
   - Visualize the predictions using `plot` feature of `peroxide` library.

## Prerequisites

- Rust
- Python
- Required python libraries
  - `matplotlib`
  - `scienceplots` (optional)

## Usage

Just run the Rust code:

```sh
cargo run --release
```

## Results

- The training and validation losses are printed during the training process.
- The MSE on the test dataset is displayed after the model evaluation.
- The predicted results are visualized and saved to the `test_plot.png` file.
  ![test_plot.png](test_plot.png)

## LICENSE

This project is licensed under the MIT license.
