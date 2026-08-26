import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from typing import Callable, List, Tuple

# Function call counter
class FunctionCounter:
    def __init__(self):
        self.count = 0
        self.evaluated = set()

    def reset(self):
        self.count = 0
        self.evaluated.clear()

    def __call__(self, f):
        def wrapper(x):
            x_tuple = tuple(np.round(x, 4)) if isinstance(x, np.ndarray) else x
            if x_tuple not in self.evaluated:
                self.count += 1
                self.evaluated.add(x_tuple)
            return f(x)
        return wrapper

counter = FunctionCounter()

# Central difference gradient evaluation
def central_difference_grad(func: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-8) -> np.ndarray:
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
    return grad
def central_difference(f: Callable[[float], float],x:float,h:float=1e-6)-> float:
    return (f(x+h)-f(x-h))/(2*h);

# Bounding phase method
def bounding_phase(f: Callable[[float], float], initial_point: float, h: float = 1e-6, max_iterations: int = 1000) -> Tuple[float, float]:
    a, b = initial_point, initial_point + h
    f_a, f_b = f(a), f(b)
    
    if f_a < f_b:
        a, b = b, a
        f_a, f_b = f_b, f_a
        h = -h
    
    for i in range(max_iterations):
        c = b + h
        f_c = f(c)
        if f_b <= f_c:
            return (a, c) if a < c else (c, a)
        a, b = b, c
        f_a, f_b = f_b, f_c
        h *= 2
    
    return (a, b) if a < b else (b, a)

# Bisection method
def bisection_method(f: Callable[[float], float], a: float, b: float, tolerance: float = 1e-6, max_iterations: int = 1000) -> float:
    grad_a = central_difference(f,a);
    grad_b = central_difference(f,b);
    if grad_a * grad_b > 0:
        a,b = bounding_phase(f,(a+b)/2);
   
    for i in range(max_iterations):
        c = (a + b) / 2
        grad_c = central_difference(f,c)
        if grad_c > 0:
            b=c;
        else:
            a=c;
        if abs(grad_c) < tolerance:
            return (a + b) / 2
    
    return (a + b) / 2

# Conjugate Gradient Method
class ConjugateGradient:
    def __init__(self, func: Callable[[np.ndarray], float], n_vars: int, epsilon: float = 1e-6, max_iter: int = 15000, restart_interval: int = 100, independence_threshold: float = 0.2):
        self.func = counter(func)
        self.n_vars = n_vars
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.restart_interval = restart_interval
        self.independence_threshold = independence_threshold

    def optimize(self, x0: np.ndarray) -> Tuple[np.ndarray, List[float], List[float]]:
        x = x0
        f_values = [self.func(x)]
        grad = central_difference_grad(self.func, x)
        d = -grad
        grad_norms = [np.linalg.norm(grad)]

        for k in range(self.max_iter):
            alpha = self.unidirectional_search(x, d)
            x_new = x + alpha * d
            grad_new = central_difference_grad(self.func, x_new)
            
            if np.linalg.norm(grad_new) < self.epsilon:
                break
            
            if k % self.restart_interval == 0 and k > 0:
                # Restart Checking
                d = -grad_new
            else:
                beta = self.compute_beta(grad, grad_new)
                d_new = -grad_new + beta * d
                
                # Linear independence Checking
                if self.check_linear_independence(d, d_new):
                    d = d_new
                else:
                    d = -grad_new
            
            if np.linalg.norm(x_new - x) / np.linalg.norm(x) < self.epsilon:
                break

            x = x_new
            grad = grad_new
            f_values.append(np.round(self.func(x), 4))
            grad_norms.append(np.linalg.norm(grad))

        return x, f_values, grad_norms
    
    def compute_beta(self, grad: np.ndarray, grad_new: np.ndarray) -> float:
        return np.dot(grad_new, grad_new) / np.dot(grad, grad) if np.dot(grad, grad) != 0 else 0
    
    def check_linear_independence(self, d: np.ndarray, d_new: np.ndarray) -> bool:
        cos_theta = np.abs(np.dot(d, d_new)) / (np.linalg.norm(d) * np.linalg.norm(d_new))
        return cos_theta < self.independence_threshold
    
    def unidirectional_search(self, x: np.ndarray, d: np.ndarray) -> float:
        def f(alpha):
            return self.func(x + alpha * d)
        
        a, b = bounding_phase(f, 0)
        return bisection_method(f, a, b)

# Function to run the optimization and save results
def run_optimization(func: Callable[[np.ndarray], float], n_vars: int, x0: np.ndarray, problem_no: int, iteration_no: int) -> None:
    counter.reset()
    optimizer = ConjugateGradient(func, n_vars)
    x_opt, f_values, grad_norms = optimizer.optimize(x0)
    
    print(f"Optimal solution: {np.round(x_opt, 4)}")
    print(f"Optimal value: {np.round(func(x_opt),4)}")
    print(f"Number of iterations: {len(f_values) - 1}")
    print(f"Number of function evaluations: {counter.count}")

    os.makedirs("output", exist_ok=True)

    csv_filename = f"D:/7th Sem/ME609_Optimisation/Phase-2/test_results/problem_{problem_no}_iteration_{iteration_no}_data.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Function Value'])
        for i, f_val in enumerate(f_values):
            writer.writerow([i, f_val])
    
    print(f"Results saved to {csv_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(f_values)), f_values, marker='o')
    plt.title(f"Problem {problem_no}, Iteration {iteration_no}: Objective Function Value vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = f"D:/7th Sem/ME609_Optimisation/Phase-2/test_results/problem_{problem_no}_iteration_{iteration_no}_ConvergenceVsIteration.png"
    plt.savefig(plot_filename)
    plt.close()

    print(f"Convergence plot saved as '{plot_filename}'")
    print("\n")

# Reading problem definitions from .txt file
def read_problem_definitions(filename: str):
    functions = []
    n_vars_list = []
    
    with open(filename, 'r') as file:
        for line in file:
            func_name, n_vars = line.strip().split(',')
            functions.append(eval(func_name))
            n_vars_list.append(int(n_vars))
    
    return functions, n_vars_list

# Problem definitions
def sum_square(x: np.ndarray) -> float:
    return np.sum([(i + 1) * (x[i] ** 2) for i in range(len(x))])

def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def dixon_price(x: np.ndarray) -> float:
    return (x[0] - 1)**2 + np.sum([(i + 1) * (2 * x[i]**2 - x[i-1])**2 for i in range(1, len(x))])

def trid(x: np.ndarray) -> float:
    return np.sum((x - 1)**2) - np.sum(x[1:] * x[:-1])

def zakharov(x: np.ndarray) -> float:
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, n+1) * x)
    return sum1 + sum2**2 + sum2**4

if __name__ == "__main__":
    functions, n_vars_list = read_problem_definitions('D:/7th Sem/ME609_Optimisation/Phase-2/problems.txt')

    while True:
        try:
            problem = int(input("Enter a problem number (1-5): "))
            if problem not in range(1, len(functions) + 1):
                raise ValueError("Please enter a valid number between 1 and 5")
            break
        except ValueError as e:
            print(e)

    for i in range(10):
        n_vars = n_vars_list[problem - 1]
        if problem==1:
            x0 = np.random.uniform(-5.12, 5.12, n_vars)
        elif problem==2:
            x0 = np.random.uniform(-2.048, 2.048, n_vars)
        elif problem==3:
            x0 = np.random.uniform(-10, 10, n_vars)
        elif problem==4:
            x0 = np.random.uniform(-n_vars**2, n_vars**2, n_vars)
        elif problem==5:
            x0 = np.random.uniform(-5, 10, n_vars)
        print(f"Problem {problem}, Iteration {i + 1}, Initial point: {np.round(x0,3)}")
        run_optimization(functions[problem - 1], n_vars, x0, problem, i + 1)
