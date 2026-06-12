import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cmath
import matplotlib
from tqdm import tqdm
try:
    matplotlib.rc('font', family='Times New Roman')
except:
    matplotlib.rc('font', family='serif')
#公式也是Times New Roman
plt.rc('mathtext',fontset='stix')

def compute_berry_curvature(band, band_index, num_kx):
    import cmath
    U = np.zeros((num_kx-1, num_kx-1), dtype=complex)
    for i in range(num_kx-1):
        for j in range(num_kx-1):
            vector = band[i, j, :, band_index]
            vector_delta_kx = band[i+1, j, :, band_index]
            vector_delta_ky = band[i, j+1, :, band_index]
            vector_delta_kx_ky = band[i+1, j+1, :, band_index]
            Ux = np.dot(np.conj(vector), vector_delta_kx) / abs(np.dot(np.conj(vector), vector_delta_kx))
            Uy = np.dot(np.conj(vector), vector_delta_ky) / abs(np.dot(np.conj(vector), vector_delta_ky))
            Ux_y = np.dot(np.conj(vector_delta_ky), vector_delta_kx_ky) / abs(np.dot(np.conj(vector_delta_ky), vector_delta_kx_ky))
            Uy_x = np.dot(np.conj(vector_delta_kx), vector_delta_kx_ky) / abs(np.dot(np.conj(vector_delta_kx), vector_delta_kx_ky))
            U[i, j] = cmath.log(Ux * Uy_x * (1 / Ux_y) * (1 / Uy))
    return U

def compute_berry_curvature_multiband(band, band_indices, num_kx, eps=1e-14):
    """
    Berry 曲率 (FHS 方法，多带，向量化 + 稳定)
    band: (Nkx, Nky, dim_H, Nband)
    band_indices: list[int]
    num_kx: 网格数
    """
    Nkx, Nky, dim_H, Nband = band.shape
    n_band = len(band_indices)

    # 截取子空间
    band_sub = band[:, :, :, band_indices]  # (Nkx, Nky, dim_H, n_band)

    # 定义四个点
    v_k     = band_sub[:-1, :-1]   # (Nk-1, Nk-1, dim, n_band)
    v_kx    = band_sub[1:, :-1]
    v_ky    = band_sub[:-1, 1:]
    v_kxky  = band_sub[1:, 1:]

    # 批量计算重叠矩阵：M = v1^† v2
    def overlap(v1, v2):
        # v1,v2: (Nk-1, Nk-1, dim, n_band)
        return np.einsum("...ia,...ib->...ab", np.conj(v1), v2)

    Mx   = overlap(v_k, v_kx)    # (..., n_band, n_band)
    My   = overlap(v_k, v_ky)
    Mx_y = overlap(v_ky, v_kxky)
    My_x = overlap(v_kx, v_kxky)

    # 批量行列式
    det_Mx   = np.linalg.det(Mx)
    det_My   = np.linalg.det(My)
    det_Mx_y = np.linalg.det(Mx_y)
    det_My_x = np.linalg.det(My_x)

    # 数值稳定相位
    def det_phase(det_val):
        mask = np.abs(det_val) < eps
        phase = np.exp(1j * np.angle(det_val))
        phase[mask] = 1.0 + 0j
        return phase

    Ux   = det_phase(det_Mx)
    Uy   = det_phase(det_My)
    Ux_y = det_phase(det_Mx_y)
    Uy_x = det_phase(det_My_x)

    # 最终 Berry 曲率
    val = Ux * Uy_x / (Ux_y * Uy)
    val[np.abs(val) < eps] = 1.0 + 0j
    U = np.log(val)   # (Nk-1, Nk-1)

    return U

def compute_quantum_metric(band, band_index, num_kx):
    U = np.zeros((num_kx-2, num_kx-2), dtype=np.complex128)
    for i in range(1, num_kx-1):
        for j in range(1, num_kx-1):
            vector = band[i, j, :, band_index]
            vector_delta_kx = band[(i+1)%num_kx, j, :, band_index]
            vector_delta_minus_kx = band[i-1, j, :, band_index]
            vector_delta_ky = band[i, (j+1)%num_kx, :, band_index]
            vector_delta_minus_ky = band[i, j-1, :, band_index]
            U[i-1, j-1] = 4 - np.abs(np.conj(vector) @ vector_delta_kx)**2 - np.abs(np.conj(vector) @ vector_delta_ky)**2 - np.abs(np.conj(vector) @ vector_delta_minus_kx)**2 - np.abs(np.conj(vector) @ vector_delta_minus_ky)**2
    return U

def generate_k_mesh(num_k, reciprocal_Tmat):
    kx = np.linspace(0, 1, num_k)
    ky = np.linspace(0, 1, num_k)
    K_mesh = np.meshgrid(kx, ky)
    K_mesh = np.array(K_mesh).reshape(2, -1).T
    K_mesh = np.hstack((K_mesh, np.zeros((K_mesh.shape[0], 1))))
    kpoints = K_mesh @ reciprocal_Tmat
    return kpoints

def plot_berry_curvature(kpoints, berry_curv, band_index, output_path, title=None):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    cbar = ax.tricontourf(kpoints[:, 0], kpoints[:, 1], berry_curv.imag.flatten(), levels=100)#, cmap='RdBu')
    fig.colorbar(cbar, ax=ax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_{x}/b_{M}$',fontsize=16)
    ax.set_ylabel(r'$k_{y}/b_{M}$',fontsize=16)
    ax.set_title(title,fontsize=16)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    # 再保存数据：将 kx, ky, curvature 三列合并
    data = np.column_stack([
        kpoints[:, 0], 
        kpoints[:, 1], 
        berry_curv.imag.flatten()
    ])
    txt_path = output_path.replace('.pdf', f'.txt')
    header = 'kx    ky    berry_curvature_imag'
    np.savetxt(txt_path, data, header=header, fmt='%12.6f')
    print(f'Saved data to {txt_path}')

def plot_quantum_metric(kpoints, qmetric, band_index, output_path):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    cbar = ax.tricontourf(kpoints[:, 0], kpoints[:, 1], qmetric.real.flatten(), levels=100, cmap='viridis')
    fig.colorbar(cbar, ax=ax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_{x}/b_{M}$',fontsize=16)
    ax.set_ylabel(r'$k_{y}/b_{M}$',fontsize=16)
    ax.set_title(f'Quantum Metric for Band {band_index}',fontsize=16)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    data = np.column_stack([
        kpoints[:, 0], 
        kpoints[:, 1], 
        qmetric.real.flatten()
    ])
    txt_path = output_path.replace('.pdf', f'.txt')
    header = 'kx    ky    quantum_metric_real'
    np.savetxt(txt_path, data, header=header, fmt='%12.6f')
    print(f'Saved data to {txt_path}')

def plot_wcc(k_grid, wcc_branches, output_path, direction, title=None):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    num_occ = wcc_branches.shape[1]
    k_plot = k_grid / np.max(k_grid) if np.max(k_grid) > 0 else k_grid
    for i in range(num_occ):
        ax.scatter(k_plot-0.5, wcc_branches[:, i], c='black', s=2)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks(np.arange(-0.5, 0.51, 0.25))
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=16)
    if direction == 'ky':
        xlabel = r'$k_x/2\pi$'
    else:
        xlabel = r'$k_y/2\pi$'
    ax.set_xlabel(xlabel, fontsize=16)
    if title:
        ax.set_title(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    header = f'k_fraction'
    for i in range(num_occ):
        header += f'    wcc_band_{i}'
    data_to_save = np.hstack([k_plot.reshape(-1, 1), wcc_branches])
    txt_path = output_path.replace('.pdf', f'.txt')
    np.savetxt(txt_path, data_to_save, header=header, fmt='%12.6f')
    print(f'Saved WCC data to {txt_path}')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_lattice_vectors_from_openmx(openmx_path):
    if not os.path.exists(openmx_path):
        raise FileNotFoundError(f"openmx.dat 文件未找到: {openmx_path}")
    with open(openmx_path, 'r') as f:
        lines = f.readlines()
    start, end = None, None
    for idx, line in enumerate(lines):
        if '<Atoms.UnitVectors' in line:
            start = idx + 1
        if 'Atoms.UnitVectors>' in line:
            end = idx
            break
    if start is None or end is None or end - start < 3:
        raise ValueError(f"openmx.dat 文件中未找到合法的 Atoms.UnitVectors 区块: {openmx_path}")
    lattice = []
    for i in range(3):
        lattice.append([float(x) for x in lines[start + i].split()[:3]])
    lattice = np.array(lattice)
    reciprocal = 2 * np.pi * np.linalg.inv(lattice.T)
    return reciprocal

def parallel_transport_path(vecs_path):
    for i in range(len(vecs_path) - 1):
        S_i = np.conj(vecs_path[i].T) @ vecs_path[i+1]
        U, _, V_dag = np.linalg.svd(S_i)
        M_i = U @ V_dag
        vecs_path[i+1] = vecs_path[i+1] @ M_i
    return vecs_path

def wilson_loop(vecs_occ_path):
    # vecs_occ_path = parallel_transport_path(vecs_occ_path)
    Npath, _, Nocc = vecs_occ_path.shape
    W = np.eye(Nocc, dtype=complex)
    for i in range(Npath):
        U_i = np.conj(vecs_occ_path[(i + 1) % Npath].T) @ vecs_occ_path[i]
        W = U_i @ W
    phases = np.angle(np.linalg.eigvals(W)) / (2 * np.pi)
    return np.sort(phases) % 1

def sweep_wcc(eig_vec_BZ, occ_bands, direction='ky'):
    if direction == 'ky':
        Nk1, Nk2 = eig_vec_BZ.shape[0], eig_vec_BZ.shape[1]
    else:
        Nk1, Nk2 = eig_vec_BZ.shape[1], eig_vec_BZ.shape[0]
    k1 = np.arange(Nk1)
    all_wcc = []
    for k_fixed in k1:
        vecs = []
        for k_loop in range(Nk2):
            if direction == 'ky':
                k_fixed_, k_loop_ = k_fixed, k_loop
            else:
                k_fixed_, k_loop_ = k_loop, k_fixed
            eigvec = eig_vec_BZ[k_fixed_, k_loop_]
            vecs.append(eigvec[:, occ_bands])
        vecs = np.stack(vecs, axis=0)
        all_wcc.append(wilson_loop(vecs))
    return k1, np.stack(all_wcc, axis=0)

def main():
    parser = argparse.ArgumentParser(description='TAPW Chern number and Berry curvature post-processing')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('-b', '--band', type=int, nargs='+', help='Band index for Berry curvature (e.g. -1 1 2)')
    parser.add_argument('-wb', '--wcc-bands', type=int, nargs='+', help='Band indices for WCC (e.g. 10 11 12)')
    parser.add_argument('-wd', '--wcc-direction', type=str, default='ky', choices=['kx', 'ky'], help='Wilson loop direction (default: ky)')
    parser.add_argument('-o', '--output-dir', type=str, default='./', help='Output directory (default: config.yaml output_dir)')
    parser.add_argument('-v', '--valley', type=int, default=1, help='Valley number (1:K1, 2:K2, 5:Gamma, 11:K1_120, 12:K1_240, 31:M1, 32:M2, 33:M3)')
    args = parser.parse_args()

    if not args.band and not args.wcc_bands:
        parser.error('No action requested, add --band or --wcc-bands')

    # 数字到valley字符串的映射
    valley_map = {1: 'K1', 2: 'K2', 5: 'Gamma', 11: 'K1_120', 12: 'K1_240', 31: 'M1', 32: 'M2', 33: 'M3'}
    valley_str = valley_map.get(args.valley, f'valley{args.valley}')

    config = load_config(args.config)
    output_dir = args.output_dir# or config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    num_chern = config['compute']['num_chern']
    # 获取band_type信息
    band_type = config['compute'].get('band_type', 'CBM')
    # 自动定位 openmx.dat 路径
    input_file = config['paths'].get('input_file', 'openmx.dat')
    openmx_path = input_file if os.path.isabs(input_file) else os.path.join(config['paths']['output_dir'], input_file)
    if not os.path.exists(openmx_path):
        print(f"[ERROR] openmx.dat 文件未找到: {openmx_path}, {os.path.join(config['paths']['output_dir'], input_file)}")
        sys.exit(1)
    try:
        reciprocal_Tmat = parse_lattice_vectors_from_openmx(openmx_path)
        reciprocal_Tmat = reciprocal_Tmat/np.linalg.norm(reciprocal_Tmat[0])
    except Exception as e:
        print(f"[ERROR] 解析 openmx.dat 失败: {e}")
        sys.exit(1)
    # 自动寻找vec文件
    vec_file = None
    # 首先尝试寻找带有band_type和num_chern标识的文件（新格式）
    if 'num_chern' in config['compute']:
        num_chern_suffix = f"_2d_{num_chern}"
        candidate = os.path.join(output_dir, f'vec_{band_type}_{valley_str}_valley{num_chern_suffix}.npy')
        if os.path.exists(candidate):
            vec_file = candidate
    
    # 如果没有找到，则尝试其他后缀（新格式）
    if vec_file is None:
        for suffix in ['2d', '2D', 'chern', 'Chern', 'CHERN', '']:  # 兼容不同后缀
            candidate = os.path.join(output_dir, f'vec_{band_type}_{valley_str}_valley{f"_{suffix}" if suffix else ""}.npy')
            if os.path.exists(candidate):
                vec_file = candidate
                break
    
    # 为了兼容性，如果仍然没有找到，则尝试旧格式
    if vec_file is None:
        if 'num_chern' in config['compute']:
            num_chern_suffix = f"_2d_{num_chern}"
            candidate = os.path.join(output_dir, f'vec_{valley_str}_valley{num_chern_suffix}.npy')
            if os.path.exists(candidate):
                vec_file = candidate
        
        if vec_file is None:
            for suffix in ['2d', '2D', 'chern', 'Chern', 'CHERN', '']:  # 兼容不同后缀
                candidate = os.path.join(output_dir, f'vec_{valley_str}_valley{f"_{suffix}" if suffix else ""}.npy')
                if os.path.exists(candidate):
                    vec_file = candidate
                    break
    if vec_file is None:
        print(f"[ERROR] 波函数文件未找到，请检查 {output_dir} 下的 vec_{band_type}_{valley_str}_valley_2d.npy 或 vec_{valley_str}_valley_2d.npy 等文件是否存在。")
        sys.exit(1)
    band_vec = np.load(vec_file)
    try:
        band_vec_reshape = band_vec.reshape(num_chern, num_chern, band_vec.shape[-2], band_vec.shape[-1])
    except Exception as e:
        print(f"[ERROR] 波函数文件形状不正确: {e}")
        sys.exit(1)
    kpoints = generate_k_mesh(num_chern-1, reciprocal_Tmat)
    kpoints_qmetric = generate_k_mesh(num_chern-2, reciprocal_Tmat)
    fig_suffix = f'{band_type}_{valley_str}_{num_chern}'
    
    if args.band:
        # print(args.band)
        for band_index in args.band:
            idx = band_index if band_index >= 0 else band_vec_reshape.shape[-1] + band_index
            if idx < 0 or idx >= band_vec_reshape.shape[-1]:
                print(f"[ERROR] 带号 {band_index} 超出范围，有效范围: 0 ~ {band_vec_reshape.shape[-1]-1} 或负数索引。")
                continue
            # 贝利曲率
            berry_curv = compute_berry_curvature(band_vec_reshape, idx, num_chern)
            chern_number = np.sum(berry_curv.imag) / (2 * np.pi)
            berry_img = os.path.join(output_dir, f'berry_band_{band_index}_{fig_suffix}.pdf')
            plot_berry_curvature(kpoints, berry_curv, band_index, berry_img, title=f'BC for Band {band_index} in {valley_str} C={chern_number:.2f}')
            with open(os.path.join(output_dir, f'chern_{band_type}_{valley_str}_band{band_index}.txt'), 'w') as f:
                f.write(f'Chern number for band {band_index} ({band_type}): {chern_number:.6f}\n')
            print(f'[INFO] Band {band_index}: Chern number = {chern_number:.6f}, 图片已保存 {berry_img}')
            # 量子几何
            qmetric = compute_quantum_metric(band_vec_reshape, idx, num_chern)
            qmetric_img = os.path.join(output_dir, f'qmetric_band_{band_index}_{fig_suffix}.pdf')
            plot_quantum_metric(kpoints_qmetric, qmetric, band_index, qmetric_img)
            print(f'[INFO] Band {band_index}: 量子几何图片已保存 {qmetric_img}')


    # ===== 多带计算 =====
    if len(args.band) > 1:
        band_str = "_".join(map(str, args.band))
        print(band_str)
        berry_curv_multiband = compute_berry_curvature_multiband(band_vec_reshape, args.band, num_chern)
        chern_number = np.sum(berry_curv_multiband.imag) / (2 * np.pi)
        berry_img_multi = os.path.join(output_dir, f'berry_bands_{band_str}_{fig_suffix}.pdf')
        plot_berry_curvature(kpoints, berry_curv_multiband, band_str, berry_img_multi,
                             title=f'BC for Bands {band_str} in {valley_str} C={chern_number:.2f}')
        with open(os.path.join(output_dir, f'chern_{band_type}_bands_{band_str}.txt'), 'w') as f:
            f.write(f'Chern number for bands {band_str} ({band_type}): {chern_number:.6f}\n')
        print(f'[INFO] Bands {band_str}: Chern number = {chern_number:.6f}, 图片已保存 {berry_img_multi}')       

    if args.wcc_bands:
        occ_bands = [i if i >= 0 else band_vec_reshape.shape[-1] + i for i in args.wcc_bands]
        for i in occ_bands:
            if i < 0 or i >= band_vec_reshape.shape[-1]:
                print(f"[ERROR] WCC 带号 {i} 超出范围，有效范围: 0 ~ {band_vec_reshape.shape[-1]-1} 或负数索引。")
                sys.exit(1)
        
        k_grid, wcc_branches = sweep_wcc(band_vec_reshape, occ_bands, direction=args.wcc_direction)
        wcc_bands_str = '_'.join(map(str, args.wcc_bands))
        wcc_img = os.path.join(output_dir, f'wcc_{args.wcc_direction}_{wcc_bands_str}_{fig_suffix}.pdf')
        title = f'WCC bands {args.wcc_bands}'
        plot_wcc(k_grid, wcc_branches, wcc_img, args.wcc_direction, title=title)
        print(f'[INFO] WCC for bands {args.wcc_bands} in direction {args.wcc_direction} saved to {wcc_img}')

if __name__ == '__main__':
    main() 
