"""
Deep Kernel Bayesian Optimization with Synchronized Parallel Thompson Sampling

This script demonstrates the implementation of parallel Bayesian optimization
using deep kernel learning and synchronized Thompson sampling for batch selection.
The implementation uses a deep neural network to learn feature representations
combined with a Gaussian Process for uncertainty estimation.

Author: Xizhe Wang
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from dknet import NNRegressor
from dknet.layers import Dense, CovMat, Scale
from dknet.optimizers import Adam

def objective_function(x):
    """
    Test objective function: A challenging 1D function with multiple local optima
    Args:
        x: Input array of shape (n_samples, 1)
    Returns:
        Function values of shape (n_samples, 1)
    """
    return (1-np.abs(x)) * np.sin(20 * (1-np.abs(x))**6)

def create_deep_gp_model(batch_size):
    """
    Create a deep kernel Gaussian Process model
    The model architecture consists of:
    1. Feature extraction layers (neural network)
    2. Dimension reduction layer
    3. Length scale layer
    4. GP kernel layer
    
    Args:
        batch_size: Size of batch for training
    Returns:
        NNRegressor: Configured deep kernel GP model
    """
    layers = [
        Dense(10, activation='relu'),  # Feature extraction layer
        Dense(2),                      # Dimension reduction layer
        Scale(fixed=True, init_vals=20),  # Length scale layer
        CovMat(kernel='rbf', alpha_fixed=False)  # GP kernel layer
    ]
    
    optimizer = Adam(learning_rate=1e-3)
    model = NNRegressor(
        layers=layers,
        opt=optimizer,
        batch_size=batch_size, 
        maxiter=2000,
        gp=True,
        verbose=False
    )
    return model

def get_next_batch_thompson(model, x_test, batch_size=20):
    """
    Implement synchronized parallel Thompson Sampling for batch selection.
    Each batch point is selected based on an independent sample from the GP posterior.
    
    Args:
        model: Trained GP model
        x_test: Test points to evaluate
        batch_size: Number of points to select in parallel
    Returns:
        tuple: (selected point indices, samples used for selection)
    """
    selected_indices = []
    all_samples = []
    
    # Draw batch_size independent samples one at a time
    for i in range(batch_size):
        # Get a single sample
        sample = model.sample_y(x_test, n_samples=1, random_state=i)
        sample = sample.reshape(-1)  # Reshape to 1D array
        all_samples.append(sample)
        
        # Find maximum point for this sample
        max_idx = np.argmax(sample)
        selected_indices.append(max_idx)
    
    # Stack all samples into a 2D array (n_points × batch_size)
    samples = np.column_stack(all_samples)
    
    return selected_indices, samples

def plot_optimization_step(ax, x_train, y_train, x_test, y_pred, y_std, 
                         next_points, iteration):
    """
    Visualize the current state of the optimization process.
    
    Args:
        ax: Matplotlib axis
        x_train: Training points
        y_train: Training values
        x_test: Test points
        y_pred: GP predicted mean
        y_std: GP predicted std
        next_points: Next batch of points to evaluate
        iteration: Current iteration number
    """
    # Plot training data
    ax.plot(x_train, y_train, '.', label='Training samples')
    
    # Plot true function
    y_true = objective_function(x_test)
    ax.plot(x_test, y_true[:,0], label='True function')
    
    # Plot GP predictions with uncertainty
    ax.plot(x_test, y_pred, label='GP posterior mean')
    ax.fill_between(x_test[:,0], 
                   y_pred.flatten() - 2 * y_std,
                   y_pred.flatten() + 2 * y_std,
                   alpha=0.5,
                   label='95% confidence interval')
    
    # Plot next evaluation points
    for x in next_points:
        ax.axvline(x[0], -1, 1, color='red', linewidth=1)
    
    ax.set_title(f'Iteration {iteration+1}')
    ax.set_xticks([])
    ax.set_yticks([])

def main():
    # Set random seed for reproducibility
    np.random.seed(2)
    
    # Generate test points
    n_test = 1000
    x_test = np.linspace(-1, 1, n_test).reshape(-1, 1)
    
    # Initialize training data
    n_initial = 32
    initial_indices = np.random.randint(0, n_test, n_initial)
    x_train = x_test[initial_indices]
    # Add noise to observations
    noise_std = 0.05
    y_train = objective_function(x_train) + np.random.normal(0.0, noise_std, size=x_train.shape)

    # Create figure for visualization
    plt.figure(figsize=(12, 8))
    
    # Run optimization for 3 iterations
    n_iterations = 3
    batch_size = 32

    for iteration in range(n_iterations):
        # Create and fit model
        model = create_deep_gp_model(batch_size=x_train.shape[0])
        model.fit(x_train, y_train)
        
        # Make predictions
        y_pred, y_std, _ = model.predict(x_test)
        
        # Select next batch using Thompson sampling
        selected_indices, _ = get_next_batch_thompson(model, x_test, batch_size)  # 不使用返回的samples
        next_points = x_test[selected_indices]
        
        # Plot current state without Thompson samples
        ax = plt.subplot(n_iterations, 1, iteration+1)
        plot_optimization_step(ax, x_train, y_train, x_test, y_pred, y_std, 
                             next_points, iteration)
        
        # Update training data
        x_train = np.vstack((x_train, next_points))
        y_train = objective_function(x_train) + np.random.normal(0.0, noise_std, size=x_train.shape)
    
    # Add legend to the last subplot
    plt.subplot(n_iterations, 1, n_iterations).legend(
        bbox_to_anchor=(0, 0),
        loc=2,
        borderaxespad=0,
        ncol=3,
        prop={'size': 8}
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()