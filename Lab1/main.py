import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

class Module:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        raise NotImplementedError

class Linear(Module):
    def __init__(self, input_dim: int, output_dim: int, optimizer: str = 'sgd', momentum_factor: float = 0.9):
        # Scale weights to prevent vanishing/exploding gradients during early training
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(1. / input_dim)
        self.bias = np.zeros((1, output_dim))
        self.inputs = None
        
        self.optimizer = optimizer
        self.momentum_factor = momentum_factor
        
        if self.optimizer == 'momentum':
            self.vel_weights = np.zeros_like(self.weights)
            self.vel_bias = np.zeros_like(self.bias)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        grad_inputs = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.inputs.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        
        if self.optimizer == 'momentum':
            self.vel_weights = self.momentum_factor * self.vel_weights + lr * grad_weights
            self.vel_bias = self.momentum_factor * self.vel_bias + lr * grad_bias
            self.weights -= self.vel_weights
            self.bias -= self.vel_bias
        else:
            self.weights -= lr * grad_weights
            self.bias -= lr * grad_bias

        return grad_inputs

class Conv2D(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(1. / (in_channels * kernel_size ** 2))
        self.bias = np.zeros((out_channels, 1))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        batch_size, _, input_height, input_width = inputs.shape
        out_h = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        outputs = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        input_slice = inputs[b, :, h_start:h_end, w_start:w_end]
                        outputs[b, c_out, h, w] = np.sum(input_slice * self.weights[c_out]) + self.bias[c_out]
        return outputs

    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        batch_size, _, input_height, input_width = self.inputs.shape
        grad_inputs = np.zeros_like(self.inputs)
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.zeros_like(self.bias)
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(grad_output.shape[2]):
                    for w in range(grad_output.shape[3]):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        input_slice = self.inputs[b, :, h_start:h_end, w_start:w_end]
                        grad_weights[c_out] += input_slice * grad_output[b, c_out, h, w]
                        grad_bias[c_out] += grad_output[b, c_out, h, w]
                        grad_inputs[b, :, h_start:h_end, w_start:w_end] += self.weights[c_out] * grad_output[b, c_out, h, w]

        # Update weights and biases
        self.weights -= lr * grad_weights
        self.bias -= lr * grad_bias
        
        return grad_inputs

def test_conv2d():
    print("\n=== Testing Conv2D Layer ===")
    dummy_images = np.random.randn(2, 3, 8, 8) 
    
    conv_layer = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
    
    output = conv_layer.forward(dummy_images)
    
    dummy_grad = np.random.randn(*output.shape)
    conv_layer.backward(dummy_grad, lr=0.01)
    print("Conv2D forward and backward pass completed successfully.")
    
class Sigmoid(Module):
    def __init__(self):
        self.outputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow in np.exp
        inputs = np.clip(inputs, -500, 500)
        self.outputs = 1.0 / (1.0 + np.exp(-inputs))
        return self.outputs

    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        grad_activation = self.outputs * (1.0 - self.outputs)
        return grad_output * grad_activation

# Common activation
class ReLU(Module):
    def __init__(self):
        self.inputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        grad_activation = (self.inputs > 0).astype(float)
        return grad_output * grad_activation
    
# Common activation
class Tanh(Module):
    def __init__(self):
        self.outputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow in np.exp
        inputs = np.clip(inputs, -500, 500)
        self.outputs = np.tanh(inputs)
        return self.outputs

    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        grad_activation = 1.0 - self.outputs ** 2
        return grad_output * grad_activation
    
# Variant of ReLU
class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        self.inputs = None
        self.alpha = alpha

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.where(inputs > 0, inputs, self.alpha * inputs)

    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        grad_activation = np.where(self.inputs > 0, 1.0, self.alpha)
        return grad_output * grad_activation

class MSELoss:
    def __init__(self):
        self.preds = None
        self.targets = None

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        self.preds = preds
        self.targets = targets
        return np.mean((preds - targets) ** 2)

    def backward(self) -> np.ndarray:
        n_samples = self.targets.shape[0]
        return 2.0 * (self.preds - self.targets) / n_samples

class Sequential:
    def __init__(self, layers: List[Module]):
        self.layers = layers

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: np.ndarray, lr: float):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

def train(X: np.ndarray, y: np.ndarray, model: Sequential, criterion: MSELoss, 
          epochs: int, lr: float, print_interval: int = 5000) -> List[float]:
    loss_history = []
    
    for epoch in range(epochs):
        preds = model.forward(X)
        loss = criterion.forward(preds, y)
        loss_history.append(loss)
        
        grad_loss = criterion.backward()
        model.backward(grad_loss, lr)
        
        if epoch % print_interval == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:5d} loss : {loss:.10f}")
            
    return loss_history

def evaluate(X: np.ndarray, y: np.ndarray, model: Sequential) -> np.ndarray:
    preds = model.forward(X)
    loss = np.mean((preds - y) ** 2)
    
    rounded_preds = np.round(preds)
    accuracy = np.mean(rounded_preds == y) * 100
    
    for i in range(len(y)):
        print(f"Iter{i+1:02d} | Ground truth: {y[i][0]:.1f} | prediction: {preds[i][0]:.5f} |")
        
    print(f"loss={loss:.5f} accuracy={accuracy:.2f}%")
    return preds

def show_result(x: np.ndarray, y: np.ndarray, pred_y: np.ndarray, title: str):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        plt.plot(x[i][0], x[i][1], 'ro' if y[i] == 0 else 'bo')
        
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    rounded_preds = np.round(pred_y)
    for i in range(x.shape[0]):
        plt.plot(x[i][0], x[i][1], 'ro' if rounded_preds[i] == 0 else 'bo')
        
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{title}_prediction.png')
    plt.close()

def plot_learning_curve(loss_history: List[float], title: str):
    plt.figure()
    plt.plot(loss_history, color='b')
    plt.title(f'Learning Curve ({title})', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{title}_loss.png')
    plt.close()

if __name__ == "__main__":
    HIDDEN_DIM = 8
    
    print("Training on Linear Dataset")
    X_lin, y_lin = generate_linear(100)
    
    model_lin = Sequential([
        Linear(2, HIDDEN_DIM),
        LeakyReLU(),
        Linear(HIDDEN_DIM, HIDDEN_DIM),
        ReLU(),
        Linear(HIDDEN_DIM, 1),
        Sigmoid()
    ])
    
    criterion = MSELoss()
    loss_history_lin = train(X_lin, y_lin, model_lin, criterion, epochs=20000, lr=0.1)
    
    print("\n--- Linear Dataset Testing ---")
    preds_lin = evaluate(X_lin, y_lin, model_lin)
    show_result(X_lin, y_lin, preds_lin, 'linear')
    plot_learning_curve(loss_history_lin, 'linear')
    
    print("\nTraining on XOR Dataset")
    X_xor, y_xor = generate_XOR_easy()
    
    model_xor = Sequential([
        Linear(2, HIDDEN_DIM),
        LeakyReLU(),
        Linear(HIDDEN_DIM, HIDDEN_DIM),
        ReLU(),
        Linear(HIDDEN_DIM, 1),
        Sigmoid()
    ])
    
    loss_history_xor = train(X_xor, y_xor, model_xor, criterion, epochs=80000, lr=0.5)
    
    print("\n--- XOR Dataset Testing ---")
    preds_xor = evaluate(X_xor, y_xor, model_xor)
    show_result(X_xor, y_xor, preds_xor, 'xor')
    plot_learning_curve(loss_history_xor, 'xor')
    
    # test_conv2d()