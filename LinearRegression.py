import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tabulate import tabulate

class LinearRegression:
    def __init__(self, learning_rate=0.001, precision=0.00001, max_iter=2500, test_size=0.1, random_state=80):
        """
        Initialize Linear Regression model From Skratch
        """
        self.learning_rate = learning_rate
        self.precision = precision
        self.max_iter = max_iter
        self.test_size = test_size
        self.random_state = random_state
        
        # Model parameters
        self.weights = None
        self.cost_history = None
        self.weights_history = None
        
        # Training metrics
        self.train_mse = None
        self.train_r2 = None
        self.test_mse = None
        self.test_r2 = None
        
        # Data
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, df):
        """
        Prepare the data for training and testing
        """
        cols = df.columns
        x = df[cols[:-1]]
        y = df[cols[-1]]
        
        # Split the data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Add bias term
        self.x_train = np.hstack([np.ones((self.x_train.shape[0], 1)), self.x_train])
        self.x_test = np.hstack([np.ones((self.x_test.shape[0], 1)), self.x_test])

    def cost_func(self, x, y, w):
        """
        Calculate Mean Squared Error
        """
        return np.mean((np.dot(x, w) - y) ** 2)

    def compute_gradient(self, x, y, w):
        """
        Compute gradient of the cost function
        """
        y_predict = np.dot(x, w)
        error = y_predict - y
        gradient = np.dot(x.T, error) / x.shape[0]
        return gradient

    def gradient_descent(self, x, y, w):
        """
        Perform gradient descent optimization
        """
        last = w + 100 * self.precision
        self.cost_history = [self.cost_func(x, y, w)]
        self.weights_history = [w]
        iter = 0
        
        while np.linalg.norm(last - w) > self.precision and iter < self.max_iter:
            last = w.copy()
            w -= self.compute_gradient(x, y, w) * self.learning_rate
            self.weights_history.append(w)
            self.cost_history.append(self.cost_func(x, y, w))
            iter += 1
            
        return w

    def evaluate_model(self, x, y):
        """
        Evaluate model performance using MSE and R²
        """
        y_pred = np.dot(x, self.weights)
        
        # Calculate MSE
        mse = np.mean((y - y_pred) ** 2)
        
        # Calculate R²
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        
        return mse, r2

    def visualize_training(self):
        """
        Create visualizations for the training process
        """
        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Plot cost curve
        ax1.plot(self.cost_history, marker='o', label='Training Cost')
        ax1.set_title('Cost Function Over Time')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.grid(True)
        ax1.legend()
        
        # 2. Plot actual vs predicted values for training data
        y_train_pred = np.dot(self.x_train, self.weights)
        ax2.scatter(self.y_train, y_train_pred, alpha=0.5, label='Training Data')
        ax2.plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        ax2.set_title('Training: Actual vs Predicted')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.grid(True)
        ax2.legend()
        
        # 3. Plot actual vs predicted values for testing data
        y_test_pred = np.dot(self.x_test, self.weights)
        ax3.scatter(self.y_test, y_test_pred, alpha=0.5, label='Test Data')
        ax3.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax3.set_title('Test: Actual vs Predicted')
        ax3.set_xlabel('Actual Values')
        ax3.set_ylabel('Predicted Values')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def fit(self, data_path):
        """
        Train the model on the provided data
        """
        # Load and prepare data
        df = pd.read_csv(data_path)
        self.prepare_data(df)
        
        # Initialize weights
        self.weights = np.zeros(self.x_train.shape[1])
        
        # Train model
        self.weights = self.gradient_descent(self.x_train, self.y_train, self.weights)
        
        # Evaluate model
        self.train_mse, self.train_r2 = self.evaluate_model(self.x_train, self.y_train)
        self.test_mse, self.test_r2 = self.evaluate_model(self.x_test, self.y_test)
        
        # Print metrics
        metrics = [
            ["MSE", f"{self.train_mse:.4f}", f"{self.test_mse:.4f}"],
            ["R² Score", f"{self.train_r2:.2f}%", f"{self.test_r2:.2f}%"]
        ]
        print("\nModel Evaluation Metrics:")
        print(tabulate(metrics, headers=["Metric", "Training Set", "Test Set"], tablefmt="fancy_grid"))
        
        # Create visualizations
        self.visualize_training()
        
        return self

    def predict(self, x):
        """
        Make predictions for new data
        """
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Add bias term if not present
        if x.shape[1] == self.weights.shape[0] - 1:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
            
        return np.dot(x, self.weights)

if __name__ == '__main__':

    # Create and train model
    model = LinearRegression()
    model.fit('Linear Regression/Linear Regression/Student_Marks.csv')
    
    # Make predictions
    # predictions = model.predict(new_data)