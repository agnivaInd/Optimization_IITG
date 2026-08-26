import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import math
import random
from decimal import Decimal, getcontext

# Set decimal precision
getcontext().prec = 4

e = 2.7183

class FunctionCounter:
    def __init__(self):
        self.count = 0
        self.evaluated = set()

    def reset(self):
        self.count = 0
        self.evaluated.clear()

    def __call__(self, f):
        def wrapper(x):
            if x not in self.evaluated:
                self.count += 1
                self.evaluated.add(x)
            return f(x)
        return wrapper

# Instantiate the counter
counter = FunctionCounter()

@counter
def f(x: float) -> float:
    # return - pow(pow(x,2)-1,3) + pow(2*x-5,4)
    # return 8 + pow(x,3) - 2*x - 2*pow(e,x)
    # return 4*x*math.sin(x)
    # return 2 * pow(x - 3, 2) + pow(e, 0.5 * pow(x, 2))
    return x**2 - 10*e**(0.1*x)
    # return 20*math.sin(x) - 15*x**2

# Central Difference Function
def central_difference(f: Callable[[float], float], x: float) -> float:
    h = 1e-6
    return (f(x + h) - f(x - h)) / (2 * h)

# Bounding Phase method
def bounding_phase(f: Callable[[float], float], x_0: float, delta: float, flag: int, max_iterations: int = 100) -> tuple:
    k = 0
    a, b, c = x_0, x_0, x_0
    
    # Determine the direction of search
    if flag == 0:  # Minimization
        if f(x_0 - delta) >= f(x_0) >= f(x_0 + delta):
            delta = delta
        else:
            delta = -delta
    else:  # Maximization
        if f(x_0 - delta) <= f(x_0) <= f(x_0 + delta):
            delta = delta
        else:
            delta = -delta
    
    while k < max_iterations:
        c = b + pow(2, k) * delta
        if (flag == 0 and f(c) < f(b)) or (flag == 1 and f(c) > f(b)):
            k += 1
            a, b = b, c
        else:
            return min(a, c), max(a, c)
    
    return min(a, c), max(a, c)

# Bisection method with convergence tracking
def bisection(f: Callable[[float], float], a: float, b: float, flag: int, tolerance: float = 1e-6, max_iterations: int = 100) -> tuple:
    x1, x2 = a, b
    iteration_values = []
    
    for i in range(max_iterations):
        z = (x1 + x2) / 2
        value = central_difference(f, z)
        iteration_values.append(f(z))
        
        if abs(value) <= tolerance:
            break
        if (flag == 0 and value < 0) or (flag == 1 and value > 0):
            x1 = z
        else:
            x2 = z
    
    return z, iteration_values

# Function to plot the results
def plot_results(f: Callable[[float], float], a: float, b: float, x0: float, bounds: tuple, optimum: float, iteration_values: list, flag: int):
    x = np.linspace(a, b, 1000)
    y = [f(xi) for xi in x]
    
    # Plot 1: Optimization Results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, y, label=r'$f(x) = 20sin(x) - 15x^{2} $')
    
    # Plot initial guess
    plt.plot(x0, f(x0), 'ro', label='Initial Guess')
    plt.annotate(f'Initial Guess\n({x0:.2f}, {f(x0):.2f})', (x0, f(x0)), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Plot bounding phase limits
    plt.axvline(x=bounds[0], color='r', linestyle='--', label='Bounding Phase Limits')
    plt.axvline(x=bounds[1], color='r', linestyle='--')
    plt.plot(bounds[0], f(bounds[0]), 'rx')
    plt.plot(bounds[1], f(bounds[1]), 'rx')
    plt.annotate(f'({bounds[0]:.2f}, {f(bounds[0]):.2f})', (bounds[0], f(bounds[0])), textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f'({bounds[1]:.2f}, {f(bounds[1]):.2f})', (bounds[1], f(bounds[1])), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Plot optimum
    plt.axvline(x=optimum, color='g', linestyle='-', label='Optimum')
    plt.plot(optimum, f(optimum), 'go')
    plt.annotate(f'Optimum\n({optimum:.2f}, {f(optimum):.2f})', (optimum, f(optimum)), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Optimization Results: Minimise $f(x) = 20sin(x) - 15x^{2}$\n')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Convergence of f(x) vs. Iterations
    plt.subplot(1, 2, 2)
    plt.plot(range(len(iteration_values)), iteration_values, marker='o', color='b')
    plt.xlabel('Iterations')
    plt.ylabel('f(x)')
    plt.title('Convergence: f(x) vs Iterations')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Optimization Problem:")
    a, b = -6, 6 # Search interval
    
    # Set the flag for minimization (0) or maximization (1)
    flag = 0  # Change this to 1 for maximization
    
    for t in range(10, 0, -1):
        counter.reset()
        x0 = random.uniform(a, b)
        print(f"Initial guess (randomly generated): {x0:.4f}")
        
        bounds = bounding_phase(f, x0, 0.1, flag)
        print(f"Bounds obtained from Bounding Phase: {bounds}")
        
        optimum, iteration_values = bisection(f, bounds[0], bounds[1], flag)
        print(f"Optimum obtained: {optimum:.4f}")
        print(f"Optimum function value: {f(optimum):.4f}")
        print(f"Number of unique function evaluations: {counter.count}")
        print(f"Iteration complete: {11-t}")
        print("\n")
        
        plot_results(f, a, b, x0, bounds, optimum, iteration_values, flag)

if __name__ == "__main__":
    main()