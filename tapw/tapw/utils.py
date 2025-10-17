"""
Utility functions for TAPW calculations
"""
import numpy as np
import scipy.linalg
import time
import sys
from functools import wraps
from datetime import datetime
import psutil

# Constants
HARTREE = 27.211386245988

def timing_decorator_factory(process_id):
    """Factory function to create timing decorators"""
    def timing_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if process_id == 0:
                start_time = time.time()
                process = psutil.Process()
                mem_before = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB

                result = func(*args, **kwargs)

                mem_after = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
                end_time = time.time()
                duration = end_time - start_time
                mem_peak = mem_after - mem_before

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Function '{func.__name__}' executed in {duration:.6f} seconds, Memory peak: {mem_peak:.6f} GB")
                sys.stdout.flush() 
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return timing_decorator

def rotate_vector(vec, angle):
    """Rotate a vector by a given angle in degrees"""
    theta = np.radians(angle)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    if len(vec) == 2:
        return np.dot(rot_matrix, vec)
    elif len(vec) == 3:
        result = np.zeros(3)
        result[:2] = np.dot(rot_matrix, vec[:2])
        result[2] = vec[2]
        return result

def unique_sorted(arr, tolerance=0.02):
    """Return a sorted array with unique elements within a specified tolerance"""
    sorted_arr = np.sort(arr)
    unique_list = []

    if len(sorted_arr) > 0:
        unique_list.append(sorted_arr[0])

    for element in sorted_arr:
        if abs(element - unique_list[-1]) > tolerance:
            unique_list.append(element)

    return unique_list

def check_hermitian(M, tol=1e-4):
    """Check if a matrix is Hermitian"""
    if not scipy.sparse.issparse(M):
        if not np.allclose(M, M.conj().T, atol=tol):
            raise ValueError("Matrix is not Hermitian")
    else:
        if not np.allclose(M.toarray(), M.toarray().conj().T, atol=tol):
            raise ValueError("Matrix is not Hermitian")

def is_positive_definite(matrix, method='cholesky', tol=1e-10):
    """
    Check if a matrix is positive definite.
    
    Args:
        matrix (numpy.ndarray): Matrix to check
        method (str): 'cholesky' or 'eigen'
        tol (float): Tolerance for numerical stability
        
    Returns:
        bool: True if positive definite
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Input matrix must be square.")
    
    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T.conj(), atol=tol):
        print("Matrix is not symmetric.")
        return False
    
    if method == 'cholesky':
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            print("Cholesky decomposition failed, matrix is not positive definite.")
            return False
    elif method == 'eigen':
        eigenvalues = scipy.linalg.eigvalsh(matrix)
        if np.all(eigenvalues > tol):
            print(f"True - minimum 10 eigenvalues: {np.sort(eigenvalues)[:10]}")
            return True
        else:
            print("Non-positive eigenvalues found, matrix is not positive definite.")
            print(f"Minimum 10 eigenvalues: {np.sort(eigenvalues)[:200]}")
            return False
    else:
        raise ValueError("Unknown method. Choose 'cholesky' or 'eigen'.")

def print_sparse_matrix_info(matrix):
    """Print information about a sparse matrix"""
    if scipy.sparse.isspmatrix(matrix):
        nnz = matrix.nnz
        shape = matrix.shape
        total_elements = shape[0] * shape[1]
        sparsity_degree = nnz / total_elements
        sparsity_percentage = sparsity_degree * 100
        memory_usage = sum(getattr(matrix, attr).nbytes for attr in ['data', 'indices', 'indptr'])
    elif isinstance(matrix, np.ndarray):
        nnz = np.count_nonzero(matrix)
        shape = matrix.shape
        total_elements = matrix.size
        sparsity_degree = nnz / total_elements
        sparsity_percentage = sparsity_degree * 100
        memory_usage = matrix.nbytes
    else:
        raise ValueError("Input must be a SciPy sparse matrix or NumPy ndarray.")
    
    print(f"\nMatrix info:")
    print(f"Shape: {shape}")
    print(f"Non-zero elements: {nnz}")
    print(f"Sparsity: {sparsity_degree:.6f} ({sparsity_percentage:.2f}%)")
    print(f"Estimated memory usage: {memory_usage / 1e9:.6f} GB")

def check_sparsity(matrix):
    """Check the sparsity of a matrix"""
    if scipy.sparse.issparse(matrix):
        non_zero_elements = matrix.count_nonzero()
        total_elements = matrix.shape[0] * matrix.shape[1]
    else:
        non_zero_elements = np.count_nonzero(matrix)
        total_elements = matrix.size

    sparsity_ratio = non_zero_elements / total_elements
    sparsity_percentage = (1 - sparsity_ratio) * 100

    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero elements: {non_zero_elements}")
    print(f"Total elements: {total_elements}")
    print(f"Sparsity ratio: {sparsity_ratio:.6f}")

    return sparsity_ratio 