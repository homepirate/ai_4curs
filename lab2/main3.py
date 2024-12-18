import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting


def load_data(filename):
    """
    Load data from a CSV file.

    Parameters:
        filename (str): Path to the CSV file.

    Returns:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
    """
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Target variable
    return X, y


def feature_normalize(X):
    """
    Normalize the features in X.

    Parameters:
        X (ndarray): Feature matrix.

    Returns:
        X_norm (ndarray): Normalized feature matrix.
        mu (ndarray): Mean of each feature.
        sigma (ndarray): Standard deviation of each feature.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def compute_cost(X, y, theta):
    """
    Compute the cost for linear regression.

    Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
        theta (ndarray): Parameters.

    Returns:
        J (float): Computed cost.
    """
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    J = (1 / (2 * m)) * np.sum(errors ** 2)
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Perform gradient descent to learn theta.

    Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
        theta (ndarray): Initial parameters.
        alpha (float): Learning rate.
        num_iters (int): Number of iterations.

    Returns:
        theta (ndarray): Learned parameters.
        J_history (list): Cost history over iterations.
    """
    m = len(y)
    J_history = []

    for i in range(num_iters):
        gradients = (X.T @ (X @ theta - y)) / m
        theta -= alpha * gradients
        J_history.append(compute_cost(X, y, theta))

        # Optionally, print cost every 1000 iterations
        if (i + 1) % 1000 == 0:
            print(f"Iteration {i + 1}/{num_iters} | Cost: {J_history[-1]:.4f}")

    return theta, J_history


def normal_equation(X, y):
    """
    Compute the parameters using the Normal Equation.

    Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.

    Returns:
        theta (ndarray): Computed parameters.
    """
    # Using pseudo-inverse for numerical stability
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta


def plot_cost_vs_alpha(alpha_values, J_histories):
    """
    Plot cost vs iterations for different alpha values.

    Parameters:
        alpha_values (list): List of alpha (learning rate) values.
        J_histories (list): List of cost histories for each alpha.
    """
    plt.figure(figsize=(10, 6))
    for alpha, J_history in zip(alpha_values, J_histories):
        plt.plot(range(1, len(J_history) + 1), J_history, label=f'alpha = {alpha}')
    plt.xlabel('Iterations')
    plt.ylabel('Cost J')
    plt.title('Cost vs Iterations for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.savefig('cost_vs_alpha.png')
    plt.close()
#
#
# def plot_predictions(X, y, theta_gd, theta_normal, predicted_example):
#     """
#     Plot predictions from Gradient Descent and Normal Equation against actual data.
#
#     Parameters:
#         X (ndarray): Normalized feature matrix with intercept term.
#         y (ndarray): Target vector.
#         theta_gd (ndarray): Parameters from Gradient Descent.
#         theta_normal (ndarray): Parameters from Normal Equation.
#         predicted_example (ndarray): Example prediction [price, speed_norm, gears_norm].
#     """
#     plt.figure(figsize=(10, 6))
#
#     # Predictions
#     predictions_gd = X @ theta_gd
#     predictions_normal = X @ theta_normal
#
#     # Scatter plot of actual data
#     plt.scatter(X[:, 1] * sigma[0] + mu[0], y, color='blue', label='Actual Data')
#
#     # Plot predictions
#     plt.plot(X[:, 1] * sigma[0] + mu[0], predictions_gd, linestyle='--', color='red', label='Gradient Descent')
#     plt.plot(X[:, 1] * sigma[0] + mu[0], predictions_normal, linestyle=':', color='green', label='Normal Equation')
#
#     # Plot predicted example
#     plt.scatter(predicted_example[1] * sigma[0] + mu[0], predicted_example[0], marker='x', s=100, color='black',
#                 label='Predicted Example')
#
#     plt.xlabel('Engine Speed')
#     plt.ylabel('Price')
#     plt.title('Price vs Engine Speed')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("predict.png")
#     plt.close()


def plot_3d(X, y, theta_gd, theta_normal, predicted_example, mu, sigma):
    """
    Plot a 3D graph with actual data points, prediction planes, and a predicted point.

    Parameters:
        X (ndarray): Normalized feature matrix with intercept term.
        y (ndarray): Target vector.
        theta_gd (ndarray): Parameters from Gradient Descent.
        theta_normal (ndarray): Parameters from Normal Equation.
        predicted_example (ndarray): Example prediction [price, speed_norm, gears_norm].
        mu (ndarray): Mean of each feature.
        sigma (ndarray): Standard deviation of each feature.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Unnormalize features for plotting
    engine_speed = X[:, 1] * sigma[0] + mu[0]
    num_gears = X[:, 2] * sigma[1] + mu[1]

    # Scatter plot of actual data
    ax.scatter(engine_speed, num_gears, y, color='blue', label='Actual Data')

    # Create meshgrid for predictions
    speed_range = np.linspace(engine_speed.min(), engine_speed.max(), 50)
    gear_range = np.linspace(num_gears.min(), num_gears.max(), 50)
    speed_mesh, gear_mesh = np.meshgrid(speed_range, gear_range)

    # Normalize meshgrid
    speed_mesh_norm = (speed_mesh - mu[0]) / sigma[0]
    gear_mesh_norm = (gear_mesh - mu[1]) / sigma[1]

    # Prepare input for predictions (with intercept term)
    X_test = np.c_[np.ones(speed_mesh_norm.ravel().shape[0]), speed_mesh_norm.ravel(), gear_mesh_norm.ravel()]

    # Predictions
    price_pred_gd = X_test @ theta_gd
    price_pred_normal = X_test @ theta_normal

    # Reshape predictions to meshgrid shape
    price_pred_gd = price_pred_gd.reshape(speed_mesh.shape)
    price_pred_normal = price_pred_normal.reshape(speed_mesh.shape)

    # Plot prediction planes
    ax.plot_surface(speed_mesh, gear_mesh, price_pred_gd, color='red', alpha=0.5, edgecolor='none')
    # ax.plot_surface(speed_mesh, gear_mesh, price_pred_normal, color='green', alpha=0.3, edgecolor='none')

    # Plot predicted example point (unnormalized)
    predicted_price, predicted_speed_norm, predicted_gears_norm = predicted_example
    predicted_speed = predicted_speed_norm * sigma[0] + mu[0]
    predicted_gears = predicted_gears_norm * sigma[1] + mu[1]
    ax.scatter(predicted_speed, predicted_gears, predicted_price, color='black', s=100, marker='X',
               label='Predicted Point')

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Actual Data', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], color='red', lw=4, label='Gradient Descent Plane'),
        # Line2D([0], [0], color='green', lw=4, label='Normal Equation Plane'),
        Line2D([0], [0], marker='X', color='w', label='Predicted Point', markerfacecolor='black', markersize=10)
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel('Engine Speed')
    ax.set_ylabel('Number of Gears')
    ax.set_zlabel('Price')
    ax.set_title('3D Price Prediction')
    plt.savefig('3d_plot.png')
    plt.close()


# Main Execution
if __name__ == "__main__":
    # 1. Load Data
    X, y = load_data('ex1data2.txt')
    m = len(y)

    # 2. Normalize Features
    X_norm, mu, sigma = feature_normalize(X)

    # 3. Add Intercept Term
    X_norm = np.hstack((np.ones((m, 1)), X_norm))

    # 4. Gradient Descent for Different Alphas
    alpha_values = [0.001, 0.01, 0.1, 0.3]
    num_iters_alpha = 10000
    J_histories_alpha = []
    theta_initial = np.zeros(X_norm.shape[1])

    print("Running Gradient Descent for different learning rates...")
    for alpha in alpha_values:
        theta = theta_initial.copy()
        theta, J_history = gradient_descent(X_norm, y, theta, alpha, num_iters_alpha)
        J_histories_alpha.append(J_history)
        print(f"Completed Gradient Descent for alpha = {alpha}")

    # 5. Plot Cost vs Alpha
    plot_cost_vs_alpha(alpha_values, J_histories_alpha)
    print("Saved plot 'cost_vs_alpha.png'")

    # 6. Select Best Alpha (e.g., alpha = 0.01 based on convergence)
    best_alpha = 0.01
    num_iters_best = 1000
    theta_gd, J_history_gd = gradient_descent(X_norm, y, theta_initial.copy(), best_alpha, num_iters_best)

    # # 7. Plot Convergence for Best Alpha
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(J_history_gd) + 1), J_history_gd, '-b', linewidth=2)
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost J')
    # plt.title('Convergence of Gradient Descent')
    # plt.grid(True)
    # plt.savefig('convergence_gd.png')
    # plt.close()
    # print("Saved plot 'convergence_gd.png'")

    # 8. Predict Using Gradient Descent
    # Example: Engine Speed = 2104, Number of Gears = 3
    engine_speed = (2104 - mu[0]) / sigma[0]
    num_gears = (3 - mu[1]) / sigma[1]
    X_pred_gd = np.array([1, engine_speed, num_gears])
    predicted_price_gd = X_pred_gd @ theta_gd
    print(f'Predicted price (Gradient Descent): ${predicted_price_gd:.2f}')

    # 9. Normal Equation
    # For Normal Equation, use original X without normalization
    X_with_intercept = np.hstack((np.ones((m, 1)), X))
    theta_normal = normal_equation(X_with_intercept, y)

    # 10. Predict Using Normal Equation
    X_pred_normal = np.array([1, 2104, 3])
    predicted_price_normal = X_pred_normal @ theta_normal
    print(f'Predicted price (Normal Equation): ${predicted_price_normal:.2f}')

    # 11. Prepare Predicted Example for Plotting
    predicted_example = np.array([predicted_price_gd, engine_speed, num_gears])

    # 12. Plot Predictions
    # plot_predictions(X_norm, y, theta_gd, theta_normal, predicted_example)
    # print("Saved plot 'predict.png'")

    # 13. Plot 3D Graph
    plot_3d(X_norm, y, theta_gd, theta_normal, predicted_example, mu, sigma)
    print("Saved plot '3d_plot.png'")
