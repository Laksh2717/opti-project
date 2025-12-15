import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Dict
import time

"""
Three Selected Problems (all with n >= 10):
1. Extended Rosenbrock Function (n=10)
2. Extended Powell Singular Function (n=12)  
3. Penalty Function I (n=10)

Three Required Algorithms:
1. Gradient Descent with Backtracking Line Search
2. BFGS (Quasi-Newton Method)
3. Trust Region Method (Cauchy Point)
"""

# PROBLEM DEFINITIONS FROM HILLSTROM (1981)

class OptimizationProblem:
    """Base class for optimization problems"""
    def __init__(self, n: int, name: str):
        self.n = n
        self.name = name
    
    def objective(self, x: np.ndarray) -> float:
        """Compute objective function value"""
        raise NotImplementedError
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient analytically"""
        raise NotImplementedError
    
    def get_starting_points(self) -> List[np.ndarray]:
        """Return two different starting points"""
        raise NotImplementedError

class ExtendedRosenbrock(OptimizationProblem):
    """
    Extended Rosenbrock Function
    f(x) = sum_{i=1}^{n/2} [100(x_{2i} - x_{2i-1}^2)^2 + (1 - x_{2i-1})^2]
    """
    def __init__(self, n: int = 10):
        assert n >= 10 and n % 2 == 0, "n must be even and >= 10"
        super().__init__(n, f"Extended Rosenbrock (n={n})")
        
    def objective(self, x: np.ndarray) -> float:
        f = 0.0
        for i in range(0, self.n, 2):
            f += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return f
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        g = np.zeros(self.n)
        for i in range(0, self.n, 2):
            g[i] = -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
            g[i+1] = 200 * (x[i+1] - x[i]**2)
        return g
    
    def get_starting_points(self) -> List[np.ndarray]:
        # Standard starting point from Hillstrom paper
        x1 = np.array([-1.2 if i % 2 == 0 else 1.0 for i in range(self.n)])
        # Alternative starting point
        x2 = np.array([0.5 if i % 2 == 0 else -0.5 for i in range(self.n)])
        return [x1, x2]


class ExtendedPowellSingular(OptimizationProblem):
    """
    Extended Powell Singular Function
    f(x) = sum_{i=1}^{n/4} [(x_{4i-3} + 10*x_{4i-2})^2 + 5*(x_{4i-1} - x_{4i})^2 
            + (x_{4i-2} - 2*x_{4i-1})^4 + 10*(x_{4i-3} - x_{4i})^4]
    """
    def __init__(self, n: int = 12):
        assert n >= 10 and n % 4 == 0, "n must be divisible by 4 and >= 10"
        super().__init__(n, f"Extended Powell Singular (n={n})")
    
    def objective(self, x: np.ndarray) -> float:
        f = 0.0
        for i in range(0, self.n, 4):
            f += (x[i] + 10*x[i+1])**2
            f += 5 * (x[i+2] - x[i+3])**2
            f += (x[i+1] - 2*x[i+2])**4
            f += 10 * (x[i] - x[i+3])**4
        return f
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        g = np.zeros(self.n)
        for i in range(0, self.n, 4):
            # Partial derivatives
            g[i] += 2*(x[i] + 10*x[i+1]) + 40*(x[i] - x[i+3])**3
            g[i+1] += 20*(x[i] + 10*x[i+1]) + 4*(x[i+1] - 2*x[i+2])**3
            g[i+2] += 10*(x[i+2] - x[i+3]) - 8*(x[i+1] - 2*x[i+2])**3
            g[i+3] += -10*(x[i+2] - x[i+3]) - 40*(x[i] - x[i+3])**3
        return g
    
    def get_starting_points(self) -> List[np.ndarray]:
        # Standard starting point from Hillstrom paper
        x1 = np.array([3.0 if i % 4 == 0 else -1.0 if i % 4 == 1 
                       else 0.0 if i % 4 == 2 else 1.0 for i in range(self.n)])
        # Alternative starting point
        x2 = np.ones(self.n) * 0.5
        return [x1, x2]


class PenaltyFunctionI(OptimizationProblem):
    """
    Penalty Function I
    f(x) = 1e-5 * sum_{i=1}^{n} (x_i - 1)^2 + (sum_{i=1}^{n} x_i^2 - 0.25)^2
    """
    def __init__(self, n: int = 10):
        assert n >= 10, "n must be >= 10"
        super().__init__(n, f"Penalty Function I (n={n})")
        self.alpha = 1e-5
    
    def objective(self, x: np.ndarray) -> float:
        term1 = self.alpha * np.sum((x - 1)**2)
        term2 = (np.sum(x**2) - 0.25)**2
        return term1 + term2
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        g = 2 * self.alpha * (x - 1)
        g += 4 * (np.sum(x**2) - 0.25) * x
        return g
    
    def get_starting_points(self) -> List[np.ndarray]:
        # Standard starting point from Hillstrom paper
        x1 = np.arange(1, self.n + 1, dtype=float)
        # Alternative starting point
        x2 = np.ones(self.n) * 0.5
        return [x1, x2]

# ALGORITHM 1: GRADIENT DESCENT WITH BACKTRACKING LINE SEARCH

def backtracking_line_search(problem: OptimizationProblem,  x: np.ndarray, 
                             d: np.ndarray, alpha_init: float = 1.0,
                             rho: float = 0.5, c: float = 1e-4,
                             max_iter: int = 50) -> float:
    """
    Backtracking line search with Armijo condition
    
    Parameters:
    - x: current point
    - d: search direction
    - alpha_init: initial step size
    - rho: reduction factor (0 < rho < 1)
    - c: Armijo constant (0 < c < 1)
    
    Returns: step size alpha
    """
    alpha = alpha_init
    f_x = problem.objective(x)
    grad_x = problem.gradient(x)
    grad_dot_d = np.dot(grad_x, d)
    
    # If not a descent direction, return small step
    if grad_dot_d >= 0: return 1e-8
    
    for _ in range(max_iter):
        x_new = x + alpha * d
        f_new = problem.objective(x_new)
        
        # Armijo condition
        if f_new <= f_x + c * alpha * grad_dot_d: return alpha
        
        alpha *= rho
    
    return alpha


def gradient_descent(problem: OptimizationProblem, x0: np.ndarray, max_iter: int = 10000, tol: float = 1e-6) -> Dict:
    """
    Gradient Descent with Backtracking Line Search
    
    Returns dictionary with:
    - x: final solution
    - f_vals: function values at each iteration
    - grad_norms: gradient norms at each iteration
    - n_iter: number of iterations
    - n_feval: number of function evaluations
    - success: whether converged
    """
    x = x0.copy()
    f_vals = [problem.objective(x)]
    grad_norms = [np.linalg.norm(problem.gradient(x))]
    n_feval = 1
    
    for k in range(max_iter):
        grad = problem.gradient(x)
        grad_norm = np.linalg.norm(grad)
        
        # Check convergence
        if grad_norm < tol:
            return {
                'x': x, 'f_vals': np.array(f_vals), 'grad_norms': np.array(grad_norms),
                'n_iter': k, 'n_feval': n_feval, 'success': True
            }
        
        # Search direction (steepest descent)
        d = -grad
        
        # Line search
        alpha = backtracking_line_search(problem, x, d)
        n_feval += 10  # Approximate number of evaluations in line search
        
        # Update
        x = x + alpha * d
        
        # Record
        f_vals.append(problem.objective(x))
        grad_norms.append(np.linalg.norm(problem.gradient(x)))
        n_feval += 1
    
    return {
        'x': x, 'f_vals': np.array(f_vals), 'grad_norms': np.array(grad_norms),
        'n_iter': max_iter, 'n_feval': n_feval, 'success': False
    }

# ALGORITHM 2: BFGS (QUASI-NEWTON METHOD)

def bfgs(problem: OptimizationProblem, x0: np.ndarray, max_iter: int = 10000, tol: float = 1e-6) -> Dict:
    """
    BFGS Quasi-Newton Method
    
    Updates an approximation of the inverse Hessian using the BFGS formula:
    H_{k+1} = (I - rho*s*y^T) H_k (I - rho*y*s^T) + rho*s*s^T
    
    where s = x_{k+1} - x_k, y = grad_{k+1} - grad_k, rho = 1/(y^T s)
    """
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)  # Initial inverse Hessian approximation
    
    f_vals = [problem.objective(x)]
    grad_norms = []
    n_feval = 1
    
    grad = problem.gradient(x)
    grad_norm = np.linalg.norm(grad)
    grad_norms.append(grad_norm)
    
    for k in range(max_iter):
        # Check convergence
        if grad_norm < tol:
            return {
                'x': x, 'f_vals': np.array(f_vals), 'grad_norms': np.array(grad_norms),
                'n_iter': k, 'n_feval': n_feval, 'success': True
            }
        
        # Search direction
        d = -H @ grad
        
        # Line search
        alpha = backtracking_line_search(problem, x, d)
        n_feval += 10
        
        # Update position
        x_new = x + alpha * d
        grad_new = problem.gradient(x_new)
        
        # BFGS update
        s = x_new - x
        y = grad_new - grad
        
        rho = 1.0 / (np.dot(y, s) + 1e-10)  # Add small value to avoid division by zero
        
        if rho > 0:  # Only update if curvature condition is satisfied
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        
        # Move to next iteration
        x = x_new
        grad = grad_new
        grad_norm = np.linalg.norm(grad)
        
        # Record
        f_vals.append(problem.objective(x))
        grad_norms.append(grad_norm)
        n_feval += 1
    
    return {
        'x': x, 'f_vals': np.array(f_vals), 'grad_norms': np.array(grad_norms),
        'n_iter': max_iter, 'n_feval': n_feval, 'success': False
    }

# ALGORITHM 3: TRUST REGION METHOD (CAUCHY POINT)

def trust_region_cauchy(problem: OptimizationProblem, x0: np.ndarray, max_iter: int = 10000, tol: float = 1e-6, delta_init: float = 2.0, delta_max: float = 100.0,eta: float = 0.1) -> Dict:
    """
    Trust Region Method with Cauchy Point (Simplified - Uses Identity Hessian)
    
    This is a more stable implementation that uses B = I (identity matrix)
    for the quadratic model, making it essentially a gradient descent with
    adaptive trust region radius.
    
    Parameters:
    - delta_init: initial trust region radius (increased to 2.0 for better coverage)
    - delta_max: maximum trust region radius (increased to 100.0)
    - eta: threshold for accepting step (0 < eta < 0.25)
    """
    x = x0.copy()
    delta = delta_init
    
    f_vals = [problem.objective(x)]
    grad_norms = []
    n_feval = 1
    
    # Counter for consecutive rejections (to detect if stuck)
    consecutive_rejections = 0
    
    for k in range(max_iter):
        grad = problem.gradient(x)
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)
        
        # Check convergence
        if grad_norm < tol:
            return {
                'x': x, 'f_vals': np.array(f_vals), 'grad_norms': np.array(grad_norms),
                'n_iter': k, 'n_feval': n_feval, 'success': True
            }
        
        # If trust region becomes too small, reset it
        if delta < 1e-10:
            delta = delta_init
            consecutive_rejections = 0
        
        # Cauchy point with B = I (identity)
        if grad_norm > 0:
            # Step size: take full gradient step or trust region boundary
            step_length = min(delta, grad_norm)
            p = -step_length * grad / grad_norm
        else: p = np.zeros_like(grad)
        
        # Evaluate at new point
        f_x = problem.objective(x)
        x_new = x + p
        f_new = problem.objective(x_new)
        n_feval += 1
        
        # Compute actual reduction
        actual_reduction = f_x - f_new
        
        # Predicted reduction for quadratic model with B = I
        predicted_reduction = -(grad @ p + 0.5 * np.dot(p, p))
        
        # Compute ratio
        if abs(predicted_reduction) < 1e-14: rho = 0.0
        else: rho = actual_reduction / predicted_reduction
        
        # Update trust region radius based on rho
        if rho < 0.25:
            # Bad step - shrink radius
            delta = 0.25 * delta
            consecutive_rejections += 1
        else:
            consecutive_rejections = 0
            if rho > 0.75 and np.linalg.norm(p) >= 0.8 * delta:
                # Good step at boundary - expand radius more aggressively
                delta = min(3.0 * delta, delta_max)
        
        # If stuck (too many rejections), force expansion
        if consecutive_rejections > 20:
            delta = min(2.0 * delta, delta_max)
            consecutive_rejections = 0
        
        # Accept step if rho > eta
        if rho > eta:
            x = x_new
        
        # Record function value at current point
        f_vals.append(problem.objective(x))
        n_feval += 1
    
    return {
        'x': x, 'f_vals': np.array(f_vals), 'grad_norms': np.array(grad_norms),
        'n_iter': max_iter, 'n_feval': n_feval, 'success': False
    }

# EXPERIMENTS AND VISUALIZATION

def run_algorithm(algorithm_name: str, algorithm_func: Callable, problem: OptimizationProblem,
                  x0: np.ndarray, starting_point_name: str) -> Dict:
    """Run a single algorithm on a problem with timing"""
    print(f"  Running {algorithm_name} from {starting_point_name}...", end=" ")
    start_time = time.time()
    result = algorithm_func(problem, x0)
    elapsed = time.time() - start_time
    
    print(f"✓ ({elapsed:.2f}s, {result['n_iter']} iter, "
          f"f={result['f_vals'][-1]:.2e}, ||∇f||={result['grad_norms'][-1]:.2e})")
    
    result['algorithm'] = algorithm_name
    result['starting_point'] = starting_point_name
    result['time'] = elapsed
    
    return result


def run_all_experiments() -> Dict:
    """Run all algorithms on all problems with all starting points"""
    
    # Define problems
    problems = [
        ExtendedRosenbrock(n=10),
        ExtendedPowellSingular(n=12),
        PenaltyFunctionI(n=10)
    ]
    
    # Define algorithms
    algorithms = [
        ("Gradient Descent", gradient_descent),
        ("BFGS", bfgs),
        ("Trust Region", trust_region_cauchy)
    ]
    
    results = {}
    
    for problem in problems:
        print(f"\nPROBLEM: {problem.name}")
        
        results[problem.name] = {}
        starting_points = problem.get_starting_points()
        
        for sp_idx, x0 in enumerate(starting_points):
            sp_name = f"Starting Point {sp_idx + 1}"
            print(f"\n{sp_name}: x0 = [{', '.join([f'{v:.2f}' for v in x0[:5]])}...]")
            
            results[problem.name][sp_name] = {}
            
            for algo_name, algo_func in algorithms:
                result = run_algorithm(algo_name, algo_func, problem, x0, sp_name)
                results[problem.name][sp_name][algo_name] = result
    
    return results


def plot_convergence(results: Dict, save_dir: str = "."):
    """Create convergence plots for all experiments"""
    
    problems = list(results.keys())
    
    for problem_name in problems:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{problem_name}", fontsize=14, fontweight='bold')
        
        for sp_idx, sp_name in enumerate(results[problem_name].keys()):
            ax = axes[sp_idx]
            
            for algo_name, result in results[problem_name][sp_name].items():
                if len(result['grad_norms']) > 0:
                    # Plot log of gradient norm (clip to avoid -inf)
                    grad_norms = np.array(result['grad_norms'])
                    grad_norms = np.maximum(grad_norms, 1e-16)  # Avoid log(0)
                    log_grad_norms = np.log10(grad_norms)
                    
                    # Plot only valid values (not NaN or Inf)
                    valid_mask = np.isfinite(log_grad_norms)
                    iterations = np.arange(len(log_grad_norms))[valid_mask]
                    valid_log_norms = log_grad_norms[valid_mask]
                    
                    if len(valid_log_norms) > 0:
                        ax.plot(iterations, valid_log_norms, label=algo_name, linewidth=2)
            
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('log₁₀(||∇f||)', fontsize=11)
            ax.set_title(sp_name, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-8, 5])  # Set reasonable y-axis limits
        
        plt.tight_layout()
        filename = f"{save_dir}/{problem_name.replace(' ', '_')}_convergence.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

def create_summary_table(results: Dict):
    """Create a summary table of all results"""
    
    print("\nSUMMARY TABLE - ALL EXPERIMENTS")
    print(f"{'Problem':<30} {'Start':<8} {'Algorithm':<18} {'Iter':<6} {'F-Eval':<8} "
          f"{'Final f':<12} {'||∇f||':<12} {'Time(s)':<8} {'Conv':<6}")
    
    for problem_name in results.keys():
        for sp_name in results[problem_name].keys():
            sp_short = sp_name.split()[-1]  # "1" or "2"
            for algo_name, result in results[problem_name][sp_name].items():
                conv_status = "✓" if result['success'] else "✗"
                print(f"{problem_name:<30} {sp_short:<8} {algo_name:<18} "
                      f"{result['n_iter']:<6} {result['n_feval']:<8} "
                      f"{result['f_vals'][-1]:<12.2e} {result['grad_norms'][-1]:<12.2e} "
                      f"{result['time']:<8.2f} {conv_status:<6}")

# MAIN EXECUTION

def main():
    """Main function to run all experiments"""
    
    # Run all experiments
    results = run_all_experiments()
    
    # Create summary table
    create_summary_table(results)
    
    # Generate convergence plots
    plot_convergence(results)
    
    return results

if __name__ == "__main__":
    results = main()