import itertools
import math
import numpy as np

def generate_column_states(Nx):
    """
    生成所有 Nx 個自旋的單列可能組態。
    每個組態是一個 tuple (例如: (-1, -1), (-1, 1), (1, -1), (1, 1))。
    """
    return list(itertools.product([-1, 1], repeat=Nx))

def calculate_vertical_energy(column_state_np, J, Nx):
    """
    計算單一列組態內部的垂直交互作用能量 (E_V)。
    column_state_np: NumPy array 形式的單列組態。
    J: 交換耦合常數。
    Nx: 列的自旋數量。
    """
    energy_v = 0
    for i in range(Nx):
        # 垂直週期性邊界條件: s_i 與 s_(i+1) 相鄰 (包括 s_(Nx-1) 與 s_0)
        energy_v += column_state_np[i] * column_state_np[(i + 1) % Nx]
    return -J * energy_v

def calculate_horizontal_energy(col1_state_np, col2_state_np, J, Nx):
    """
    計算兩個相鄰列組態之間的水平交互作用能量 (E_H)。
    col1_state_np: NumPy array 形式的第一列組態。
    col2_state_np: NumPy array 形式的第二列組態。
    J: 交換耦合常數。
    Nx: 列的自旋數量。
    """
    # 這裡可以直接使用 NumPy 的點積，因為是逐元素相乘後求和
    energy_h = np.dot(col1_state_np, col2_state_np)
    return -J * energy_h

def calculate_field_energy(column_state_np, h, Nx):
    """
    計算外部磁場對單一列組態的作用能量 (E_F)。
    column_state_np: NumPy array 形式的單列組態。
    h: 外部磁場強度。
    Nx: 列的自旋數量。
    """
    # 直接對列組態所有自旋求和
    energy_f = np.sum(column_state_np)
    return -h * energy_f

def build_transfer_matrix(J, h, beta, Nx):
    """
    建構 Nx 維 Ising 模型的一維列傳遞矩陣。
    T_sigma_sigma_prime = exp(-beta * (E_V(sigma_prime) + E_H(sigma, sigma_prime) + E_F(sigma_prime)))
    """
    column_states_tuples = generate_column_states(Nx)
    num_column_states = len(column_states_tuples) # 也就是 2^Nx
    
    # 初始化傳遞矩陣
    T = np.zeros((num_column_states, num_column_states))
    
    # 遍歷所有可能的 (sigma, sigma_prime) 對
    for idx_sigma, sigma_tuple in enumerate(column_states_tuples):
        sigma_np = np.array(sigma_tuple)
        for idx_sigma_prime, sigma_prime_tuple in enumerate(column_states_tuples):
            sigma_prime_np = np.array(sigma_prime_tuple)
            
            # 計算能量貢獻
            E_V_prime = calculate_vertical_energy(sigma_prime_np, J, Nx)
            E_V = calculate_vertical_energy(sigma_np, J, Nx)
            E_H_sigma_sigma_prime = calculate_horizontal_energy(sigma_np, sigma_prime_np, J, Nx)
            E_F_prime = calculate_field_energy(sigma_prime_np, h, Nx)         
            E_F = calculate_field_energy(sigma_np, h, Nx)
            
            # 計算 Boltzmann 因子並填入矩陣元素
            T[idx_sigma, idx_sigma_prime] = math.exp(-beta * (E_V_prime + E_H_sigma_sigma_prime + E_F_prime))
            # T[idx_sigma, idx_sigma_prime] = \
                # math.exp(-beta * (E_V_prime/2 + E_H_sigma_sigma_prime + E_F_prime/2 + E_F/2 + E_V/2))
            
    return T

def partition_function_transfer_matrix(J, h, beta, Nx, Ny):
    """
    使用傳遞矩陣方法計算 Nx x Ny Ising 模型的週期性邊界條件下的分佈函數。
    Nx: 列數 (垂直方向的自旋數量)
    Ny: 行數 (水平方向的列數量)
    """
    if Ny <= 0:
        return 0.0 # 點陣長度為0沒有意義
        
    # 1. 建構傳遞矩陣 T
    T_matrix = build_transfer_matrix(J, h, beta, Nx)
    print(T_matrix)
    # 2. 計算 T 的 Ny 次方
    T_power_Ny = np.linalg.matrix_power(T_matrix, Ny)
    
    # 3. 計算矩陣跡 (Trace)
    Z = np.trace(T_power_Ny)
    
    return Z

# ---
# 定義參數
J_value = 1.0 # 交換耦合常數
h_value = 0.0 # 外部磁場強度
temperature = 1.0 # 溫度
beta_value = 1.0 / temperature # 逆溫度 (假設 k_B = 1)

print("--- 使用傳遞矩陣方法計算 ---")

# 2x2 點陣
Nx_val_2x2 = 2
Ny_val_2x2 = 2
Z_tm_2x2 = partition_function_transfer_matrix(J_value, h_value, beta_value, Nx_val_2x2, Ny_val_2x2)
print(f"當 Nx={Nx_val_2x2}, Ny={Ny_val_2x2}, J={J_value}, h={h_value}, T={temperature} 時，分佈函數 Z (傳遞矩陣) = {Z_tm_2x2:.4f}")

# 2x3 點陣
# Nx_val_2x3 = 2
# Ny_val_2x3 = 3
# Z_tm_2x3 = partition_function_transfer_matrix(J_value, h_value, beta_value, Nx_val_2x3, Ny_val_2x3)
# print(f"當 Nx={Nx_val_2x3}, Ny={Ny_val_2x3}, J={J_value}, h={h_value}, T={temperature} 時，分佈函數 Z (傳遞矩陣) = {Z_tm_2x3:.4f}")

# # 3x2 點陣 (注意：這裡 Nx 和 Ny 對調了，傳遞矩陣的維度會變大)
# # 如果想計算 3x2，那麼 Nx=3, Ny=2。傳遞矩陣大小會是 2^3 x 2^3 = 8x8。
# Nx_val_3x2 = 3
# Ny_val_3x2 = 2
# Z_tm_3x2 = partition_function_transfer_matrix(J_value, h_value, beta_value, Nx_val_3x2, Ny_val_3x2)
# print(f"當 Nx={Nx_val_3x2}, Ny={Ny_val_3x2}, J={J_value}, h={h_value}, T={temperature} 時，分佈函數 Z (傳遞矩陣) = {Z_tm_3x2:.4f}")

# # 3x3 點陣
# Nx_val_3x3 = 3
# Ny_val_3x3 = 3
# Z_tm_3x3 = partition_function_transfer_matrix(J_value, h_value, beta_value, Nx_val_3x3, Ny_val_3x3)
# print(f"當 Nx={Nx_val_3x3}, Ny={Ny_val_3x3}, J={J_value}, h={h_value}, T={temperature} 時，分佈函數 Z (傳遞矩陣) = {Z_tm_3x3:.4f}")

# print("\n--- 嘗試更大的 Nx (例如 Nx=4, Ny=4) ---")

