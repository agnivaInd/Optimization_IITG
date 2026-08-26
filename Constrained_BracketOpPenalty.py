import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import warnings
from typing import Callable, List, Tuple

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')

# Function call counter remains the same
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

# Improved gradient evaluation
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

def central_difference(f: Callable[[float], float], x: float, h: float = 1e-8) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)

# Improved bounding phase method
def bounding_phase(f: Callable[[float], float], initial_point: float, h: float = 1e-4, 
                  max_iterations: int = 1000, max_step: float = 1.0) -> Tuple[float, float]:
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

# Improved bisection method
def bisection_method(f: Callable[[float], float], a: float, b: float, 
                    tolerance: float = 1e-8, max_iterations: int = 1000) -> float:
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

# Improved Bracket-Operator Penalty Method
class BracketOperatorPenalty:
    def __init__(self, func: Callable[[np.ndarray], float], constraints: List[Callable[[np.ndarray], float]], 
                 n_vars: int, epsilon: float = 1e-8, max_iter: int = 2000, 
                 restart_interval: int = 50, independence_threshold: float = 0.1):
        self.func = counter(func)
        self.constraints = constraints
        self.n_vars = n_vars
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.restart_interval = restart_interval
        self.independence_threshold = independence_threshold
        self.r_initial = 0.1
        self.r_multiplier = 10.0
        self.r = self.r_initial
        self.tolerance = 1e-8
        self.r_max = 1e6

    def penalty_func(self, x: np.ndarray) -> float:
        f = self.func(x)
        if not np.isfinite(f):
            return float('inf')
        
        penalty = 0
        for g in self.constraints:
            violation = g(x)
            if violation > 0:
                penalty += self.r * (max(violation, 0) ** 2)
        return f + penalty
    
    def check_constraints(self, x: np.ndarray) -> bool:
        return all(g(x) <= 0 for g in self.constraints)

    def compute_beta(self, grad: np.ndarray, grad_new: np.ndarray) -> float:
        return np.dot(grad_new, grad_new - grad) / np.dot(grad, grad)

    def check_linear_independence(self, d: np.ndarray, d_new: np.ndarray) -> bool:
        angle = np.dot(d, d_new) / (np.linalg.norm(d) * np.linalg.norm(d_new))
        return abs(angle) < 1 - self.independence_threshold

    def unidirectional_search(self, x: np.ndarray, d: np.ndarray, bounds: List[Tuple[float, float]]) -> float:
        def f(alpha):
            x_new = np.clip(x + alpha * d, [b[0] for b in bounds], [b[1] for b in bounds])
            return self.penalty_func(x_new)

        alpha_max_list = []
        alpha_min_list = []
        
        for i, (low, high) in enumerate(bounds):
            alpha_1 = (high - x[i]) / d[i] if d[i] != 0 else float('inf')
            alpha_2 = (low - x[i]) / d[i] if d[i] != 0 else -float('inf')
            alpha_max_list.append(max(alpha_1, alpha_2))
            alpha_min_list.append(min(alpha_1, alpha_2))
        
        alpha_max = min(alpha_max_list)
        alpha_min = max(alpha_min_list)
        alpha_min = max(alpha_min, -1e10)
        alpha_max = min(alpha_max, 1e10)

        initial_alpha = np.random.uniform(alpha_min, alpha_max) if alpha_min < alpha_max else 0.0
        a, b = bounding_phase(f, initial_alpha, 1e-3, max_step=alpha_max)
        alpha = bisection_method(f, a, b)
        return min(alpha, alpha_max)

    def optimize(self, x0: np.ndarray, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, List[float], List[float]]:
        x = x0.copy()
        print(f"Initial Guess: {x}")
        best_x = x0.copy()
        best_f = float('inf')
        total_f_values = []
        total_grad_norms = []
        penalty_param = self.r
        iteration = 0
        
        for i in range(100):
            grad = central_difference_grad(self.penalty_func, x)
            grad_norm = np.linalg.norm(grad)
            grad_norms = [grad_norm]
            f_values = [self.penalty_func(x)]
            d = -grad
            
            for k in range(self.max_iter):
                alpha = self.unidirectional_search(x, d, bounds)
                x_new = np.clip(x + alpha * d, [b[0] for b in bounds], [b[1] for b in bounds])
                if abs(self.func(x) - self.func(x_new)) < self.epsilon:
                    break
                f_new = self.penalty_func(x_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                
                grad_new = central_difference_grad(self.penalty_func, x_new)
                grad_new_norm = np.linalg.norm(grad_new)
                
                grad_norms.append(grad_new_norm)
                f_values.append(f_new)
                
                if k % self.restart_interval == 0:
                    d = -grad_new
                else:
                    beta = max(0, self.compute_beta(grad, grad_new))
                    d_new = -grad_new + beta * d
                    
                    if self.check_linear_independence(d, d_new):
                        d = d_new
                    else:
                        d = -grad_new
                
                x = x_new
                grad = grad_new
                current_penalty_value = self.penalty_func(x_new)
                penalty_change = abs(current_penalty_value - self.penalty_func(x))
                
                if self.check_constraints(x) < self.epsilon:
                    break
                
                if penalty_param > self.r_max:
                    break
                
                if penalty_change < self.tolerance:
                    break
                
                penalty_param *= self.r_multiplier
                iteration += 1

            total_f_values.extend(f_values)
            total_grad_norms.extend(grad_norms)
            
            if grad_norm < self.epsilon:
                break

        return best_x, total_f_values, total_grad_norms

# Problem definitions
def problem1(x: np.ndarray) -> float:
    return (x[0] - 10)**3 + (x[1] - 20)**3

def problem1_constraint1(x: np.ndarray) -> float:
    return -(x[0] - 5)**2 - (x[1] - 5)**2 + 100

def problem1_constraint2(x: np.ndarray) -> float:
    return (x[0] - 6)**2 + (x[1] - 5)**2 - 82.81

def problem2(x: np.ndarray) -> float:
    denominator = x[0]**3 * (x[0] + x[1])
    if abs(denominator) < 1e-10: 
        return float('inf')
    return -((np.sin(2*np.pi*x[0])**3) * np.sin(2*np.pi*x[1]) / denominator)

def problem2_constraint1(x: np.ndarray) -> float:
    return x[0]**2 - x[1] + 1

def problem2_constraint2(x: np.ndarray) -> float:
    return 1 - x[0] + (x[1] - 4)**2

def problem3(x: np.ndarray) -> float:
    return x[0] + x[1] + x[2]

def problem3_constraint1(x: np.ndarray) -> float:
    return -1 + 0.0025*(x[3] + x[5])

def problem3_constraint2(x: np.ndarray) -> float:
    return -1 + 0.0025*(-x[3] + x[4] + x[6])

def problem3_constraint3(x: np.ndarray) -> float:
    return -1 + 0.01*(-x[5] + x[7])

def problem3_constraint4(x: np.ndarray) -> float:
    return 100*x[0] - x[0]*x[5] + 833.33252*x[3] - 83333.333

def problem3_constraint5(x: np.ndarray) -> float:
    return x[1]*x[3] - x[1]*x[6] - 1250*x[3] + 1250*x[4]

def problem3_constraint6(x: np.ndarray) -> float:
    return x[2]*x[4] - x[2]*x[7] - 2500*x[4] + 1250000

# Function to run the optimization and save results
def run_optimization(func: Callable[[np.ndarray], float], constraints: List[Callable[[np.ndarray], float]], 
                     n_vars: int, bounds: List[Tuple[float, float]], problem_no: int, iteration_no: int) -> float:
    counter.reset()
    x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
    optimizer = BracketOperatorPenalty(func, constraints, n_vars)
    x_opt, f_values, grad_norms = optimizer.optimize(x0, bounds)
    
    print(f"Optimal solution: {x_opt}")
    print(f"Optimal value: {func(x_opt)}")
    print(f"Number of iterations: {len(f_values) - 1}")
    print(f"Number of function evaluations: {counter.count}")

    os.makedirs("D:/7th Sem/ME609_Optimisation/Phase-3/output", exist_ok=True)

    csv_filename = f"D:/7th Sem/ME609_Optimisation/Phase-3/output/problem_{problem_no}_iteration_{iteration_no}_data.csv"
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
    
    plot_filename = f"D:/7th Sem/ME609_Optimisation/Phase-3/output/problem_{problem_no}_iteration_{iteration_no}_ConvergenceVsIteration.png"
    plt.savefig(plot_filename)
    plt.close()

    print(f"Convergence plot saved as '{plot_filename}'")
    print("\n")

    return func(x_opt)

if __name__ == "__main__":
    problems = [
        (problem1, [problem1_constraint1, problem1_constraint2], 2, [(13, 15), (0, 1)]),
        (problem2, [problem2_constraint1, problem2_constraint2], 2, [(1, 2), (0, 5)]),
        (problem3, [problem3_constraint1, problem3_constraint2, problem3_constraint3, 
                    problem3_constraint4, problem3_constraint5, problem3_constraint6], 
         8, [    (463.4534, 695.1800),
    (1087.9544, 1631.9316),
    (4088.0568, 6132.0852),
    (145.6139, 218.4209),
    (236.4788, 354.7182),
    (174.3839, 261.5759),
    (229.1329, 343.6994),
    (316.4783, 474.7175)])
    ]

    results = []
    problem_no = int(input("Enter the problem number (1-3): "))
    
    # Validate input
    if problem_no < 1 or problem_no > len(problems):
        print("Invalid problem number. Please enter a number between 1 and 3.")
    else:
        func, constraints, n_vars, bounds = problems[problem_no - 1]
        problem_results = []
        for i in range(10):
            print(f"Problem {problem_no}, Iteration {i + 1}")
            result = run_optimization(func, constraints, n_vars, bounds, problem_no, i + 1)
            problem_results.append(result)

        best = min(problem_results)
        worst = max(problem_results)
        mean = np.mean(problem_results)
        median = np.median(problem_results)
        std_dev = np.std(problem_results)

        results.append({
            'Problem': problem_no,
            'Best': best,
            'Worst': worst,
            'Mean': mean,
            'Median': median,
            'Std Dev': std_dev
        })

        # Save summary results
        summary_filename = "D:/7th Sem/ME609_Optimisation/Phase-3/output/summary_results.csv"
        with open(summary_filename, 'w', newline='') as csvfile:
            fieldnames = ['Problem', 'Best', 'Worst', 'Mean', 'Median', 'Std Dev']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"Summary results saved to {summary_filename}")
