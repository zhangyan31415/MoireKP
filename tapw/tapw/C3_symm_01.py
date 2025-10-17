import numpy as np
import scipy

from .rot_matrix import get_any_rot_orb_twostep


sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

def get_g_vec_perlayer(m_g_vec,twisted_index_m):
    # K1 = twisted_index_m*m_g_vec[0] + 1/3*m_g_vec[0] + 1/3*m_g_vec[1]
    # K2 = twisted_index_m*m_g_vec[0] + 2/3*m_g_vec[0] - 1/3*m_g_vec[1]
    
    K1 = twisted_index_m*m_g_vec[0] + 2/3*m_g_vec[0] - 1/3*m_g_vec[1]
    K2 = twisted_index_m*m_g_vec[0] + 1/3*m_g_vec[0] + 1/3*m_g_vec[1]
    K1 = K1[:2]
    K2 = K2[:2]
    K1_G1 = np.sqrt(3)*rot(K1,-30)
    K1_G2 = np.sqrt(3)*rot(K1,30)
    K2_G1 = np.sqrt(3)*rot(K2,-30)
    K2_G2 = np.sqrt(3)*rot(K2,30)
    g_vec_1l = np.array([K1_G1,K1_G2])
    g_vec_2l = np.array([K2_G1,K2_G2])
    return g_vec_1l,g_vec_2l


def direct_sum(*matrices):
    """
    计算多个矩阵的直和
    
    参数：
    matrices: 一个矩阵列表
    
    返回值：
    直和矩阵
    """
    # 计算直和矩阵的形状
    shape_sum = np.sum([matrix.shape for matrix in matrices], axis=0)
    
    # 构造直和矩阵
    direct_sum_matrix = np.zeros(shape_sum,dtype=np.complex128)
    row_start = 0
    col_start = 0
    for matrix in matrices:
        rows, cols = matrix.shape
        direct_sum_matrix[row_start:row_start+rows, col_start:col_start+cols] = matrix
        row_start += rows
        col_start += cols

    return direct_sum_matrix

def rot_matrix(theta):
    """
    Args:
        theta: rotation angle in degree
    """
    theta = np.deg2rad(theta)
    rot = np.zeros((3,3),dtype=np.float64)
    rot[0,0] = np.cos(theta)
    rot[0,1] = -np.sin(theta)
    rot[1,0] = np.sin(theta)
    rot[1,1] = np.cos(theta)
    rot[2,2] = 1.0
    return rot

def C3_G_matrix(g_vec_list_K1_1layer,g_vec_list_K1_2layer,m_g_vec,twisted_index_m,valley):
    num_gn = len(g_vec_list_K1_1layer)
    C3 = np.zeros((num_gn+num_gn,num_gn+num_gn),dtype=np.complex128)
    C3_1layer = np.zeros((num_gn,num_gn),dtype=np.complex128)
    C3_2layer = np.zeros((num_gn,num_gn),dtype=np.complex128)

    g_vec_1l,g_vec_2l = get_g_vec_perlayer(m_g_vec,twisted_index_m)
    # m_K1 = 1/3*m_g_vec[0] + 1/3*m_g_vec[1]
    # m_K2 = 2/3*m_g_vec[0] - 1/3*m_g_vec[1]
    # m_K3 = 1/3*m_g_vec[0] - 2/3*m_g_vec[1]
    # m_K4 = -m_K1
    # m_K5 = -m_K2
    # m_K6 = -m_K3
    K1_1layer = rot(1/3*g_vec_1l[0]+1/3*g_vec_1l[1],120) #+ m_K1[:2]
    K1_2layer = rot(1/3*g_vec_2l[0]+1/3*g_vec_2l[1],120) #+ m_K1[:2]
    K2_1layer = -rot(1/3*g_vec_1l[0]+1/3*g_vec_1l[1],120) #+ m_K1[:2]
    K2_2layer = -rot(1/3*g_vec_2l[0]+1/3*g_vec_2l[1],120) #+ m_K1[:2]
    
    m_g1 = m_g_vec[0][:2]
    m_g2 = m_g_vec[1][:2]
    if twisted_index_m % 2 == 1:
        offset_1 = (twisted_index_m + 1) * (m_g1 + m_g2) / 2
        offset_2 = rot(offset_1, 120)
        offset_3 = rot(offset_1, 240)
        m_M1 = -1/2 * m_g1
        m_M2 = -1/2 * m_g2
        m_M3 = rot(m_M2, 120)
    else:
        offset_1 = twisted_index_m * (m_g1 + m_g2) / 2
        offset_2 = rot(offset_1, 120)
        offset_3 = rot(offset_1, 240)
        m_M1 = 1/2 * m_g1
        m_M2 = 1/2 * m_g2
        m_M3 = rot(m_M2, 120)

    if valley == 1:
        K_1layer = K1_1layer
        K_2layer = K1_2layer

    elif valley == 2:
        K_1layer = K2_1layer
        K_2layer = K2_2layer
    elif valley == 11:
        K_1layer = rot(K1_1layer,120)
        K_2layer = rot(K1_2layer,120)
    elif valley == 12:
        K_1layer = rot(K1_1layer,240)
        K_2layer = rot(K1_2layer,240)
    elif valley == 5:
        K_1layer = np.zeros(2)
        K_2layer = np.zeros(2)
    else:
        raise ValueError("valley should be 1 or 2")
    print("valley = ",valley)
    print("K_1layer = ",K_1layer)
    print("K_2layer = ",K_2layer)
    

    value = 1
    for i in range(num_gn):
        for j in range(num_gn):
            delta1 = np.linalg.norm(g_vec_list_K1_1layer[i]-(rot(g_vec_list_K1_1layer[j]-K_1layer,120)+K_1layer))
            delta2 = np.linalg.norm(g_vec_list_K1_2layer[i]-(rot(g_vec_list_K1_2layer[j]-K_2layer,120)+K_2layer))
            if delta1 < 1e-2:
                C3[i,j] = value
                C3_1layer[i,j] = value
                # print("i = ",i,"j = ",j)
            if delta2 < 1e-2:
                C3[i+num_gn,j+num_gn] = value
                C3_2layer[i,j] = value
                # print("i = ",i+num_gn,"j = ",j+num_gn)
    print(" abs(C3 G matrix) = ",np.abs(C3).sum())
    return C3_1layer,C3_2layer

def generate_direct_sum_params(orbitals, orbital_mapping):
    params = []
    for orbital, count in orbitals.items():
        if orbital in orbital_mapping:
            params.extend([orbital_mapping[orbital]] * count)
        else:
            raise ValueError(f"Unknown orbital type: {orbital}")
    return params

def C3_MoTe2_all(C3_Gn_1layer,C3_Gn_2layer,atoms_species_1layer=None,atoms_species_2layer=None,spin=False):
    """
    Args:
        C3_Gn: 2D array
        valley: int
        phase: bool
        atoms_species: list. eg. {'Mo': {'orb_num': 19, 'index': 0, 'orbitals': {'s': 3, 'p': 2, 'd': 2}, 'atom_num': 434}
    """

    C3_rot_matrix = rot_matrix(120)


    C3spin = scipy.linalg.expm(-1j*sigma_z/2*2*np.pi/3)

    C3_s = get_any_rot_orb_twostep('s',C3_rot_matrix)
    C3_p = get_any_rot_orb_twostep('p',C3_rot_matrix)
    C3_d = get_any_rot_orb_twostep('d',C3_rot_matrix) 
    C3_f = get_any_rot_orb_twostep('f',C3_rot_matrix)

    orbital_mapping = {
    's': C3_s,
    'p': C3_p,
    'd': C3_d,
    'f': C3_f
}

    # for key in atoms_species.keys():
    #     if key in ['Mo','W']:
    #         params = generate_direct_sum_params(atoms_species[key]['orbitals'], orbital_mapping)
    #         C3_M_rep = direct_sum(*params)
    #     elif key in ['Te','Se']:
    #         params = generate_direct_sum_params(atoms_species[key]['orbitals'], orbital_mapping)
    #         C3_X_rep = direct_sum(*params)

    # C3_MX2_spin_rep = direct_sum(np.kron(C3spin,C3_X_rep),np.kron(C3spin,C3_X_rep),np.kron(C3spin,C3_M_rep))
    # C3_all_rep = np.kron(C3_Gn,C3_MX2_spin_rep)
    
    C3_rep_matrices_1layer = []
    C3_rep_matrices_2layer = []
    for atom_index in atoms_species_1layer.keys():
        params = generate_direct_sum_params(atoms_species_1layer[atom_index]['orbitals'], orbital_mapping)
        # if spin:
        #     C3_rep_matrices.append(np.kron(C3spin,direct_sum(*params)))
        # else:
        C3_rep_matrices_1layer.append(direct_sum(*params))
    for atom_index in atoms_species_2layer.keys():
        params = generate_direct_sum_params(atoms_species_2layer[atom_index]['orbitals'], orbital_mapping)
        # if spin:
        #     C3_rep_matrices.append(np.kron(C3spin,direct_sum(*params)))
        # else:
        C3_rep_matrices_2layer.append(direct_sum(*params))
    C3_rep_1layer = direct_sum(*C3_rep_matrices_1layer)
    C3_rep_2layer = direct_sum(*C3_rep_matrices_2layer)
    C3_all_rep = direct_sum(np.kron(C3_Gn_1layer,C3_rep_1layer),np.kron(C3_Gn_2layer,C3_rep_2layer))
    # C3_all_rep = np.kron(C3_Gn_1layer,C3_rep)
    if spin:
        C3_all_rep = np.kron(C3spin,C3_all_rep)

    return C3_all_rep

def rot(vec,theta):
    theta = np.pi/180*theta
    rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.dot(rot_mat,vec)

def spin_reps(prep):
    """
    Copied from G Winker's symmetry_soc.py.
    Calculates the spin rotation matrices. The formulas to determine the rotation axes and angles
    are taken from `here <http://scipp.ucsc.edu/~haber/ph116A/rotation_11.pdf>`_.

    :param prep:   List that contains 3d rotation matrices.
    :type prep:    list(array)
    """
    # general representation of the D1/2 rotation about the axis (l,m,n) around the
    # angle phi
    D12 = lambda l,m,n,phi: np.array([[np.cos(phi/2.) - 1j*n*np.sin(phi/2.), (-1j*l -m)*np.sin(phi/2.)],
                               [(-1j*l + m)*np.sin(phi/2.), np.cos(phi/2.) + 1j*n*np.sin(phi/2.)]])

    #print "prep\n", prep
    n = np.zeros(3)
    tr = np.trace(prep)
    det = np.round(np.linalg.det(prep),5)
    if  det == 1.: #rotations
        theta = np.arccos(0.5*(tr-1.))
        if theta != 0:
            n[0] = prep[2,1]-prep[1,2]
            n[1] = prep[0,2]-prep[2,0]
            n[2] = prep[1,0]-prep[0,1]               
            if np.round(np.linalg.norm(n),5) == 0.: # theta = pi, that is C2 rotations
                e,v = np.linalg.eig(prep)
                n = v[:,list(np.round(e,10)).index(1.)]
                spin=np.round(D12(n[0],n[1],n[2],np.pi),15)
            else:
                n /= np.linalg.norm(n)
                spin=np.round(D12(n[0],n[1],n[2],theta),15)
        else: # case of unitiy
            spin=D12(0,0,0,0)
    elif det == -1.: #improper rotations and reflections
        theta = np.arccos(0.5*(tr+1.)) 
        if np.round(theta,5) != np.round(np.pi,5):                 
            n[0] = prep[2,1]-prep[1,2]
            n[1] = prep[0,2]-prep[2,0]
            n[2] = prep[1,0]-prep[0,1]                
            if np.round(np.linalg.norm(n),5)== 0.: # theta = 0 (reflection)
                e,v = np.linalg.eig(prep)
                n = v[:,list(np.round(e,10)).index(-1.)] # normal vector is eigenvector to eigenvalue -1
                spin=np.round(D12(n[0],n[1],n[2],np.pi),15) #spin is a pseudovector!
            else:
                n /= np.linalg.norm(n)
                # rotation followed by reflection:
                spin=np.round(np.dot(D12(n[0],n[1],n[2],np.pi),D12(n[0],n[1],n[2],theta)),15)
        else: # case of inversion (does not do anything to spin)
            spin=D12(0,0,0,0)
    else:
        print("rotation matrix is ",prep)
        raise ValueError(f"Determinant of the rotation matrix is not 1 or -1. det = {det}")
    return np.array(spin)


def rotate_mat(axis, radian):
    rot_matrix = scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * radian))
    return rot_matrix

def C2_G_matrix(g_vec_list_K1_1layer,g_vec_list_K1_2layer,axis):
    num_gn = len(g_vec_list_K1_1layer)
    C2 = np.zeros((num_gn+num_gn,num_gn+num_gn),dtype=np.complex128)

    for i in range(num_gn):
        for j in range(num_gn):
            # shift = g_vec_list_K1_1layer[i]-g_vec_list_K1_2layer[j]
            # center = (g_vec_list_K1_1layer[i]+g_vec_list_K1_2layer[j])/2
            # delta1 = np.abs(shift@axis)
            # delta2 = np.abs(shift@center)
            delta = np.linalg.norm(g_vec_list_K1_1layer[i]-rotate_mat(axis,np.pi)[:2][:,:2]@g_vec_list_K1_2layer[j])
            if delta < 1e-4:
                C2[i,j+num_gn] = 1
                C2[j+num_gn,i] = 1
                # print("i = ",i,"j = ",j,g_vec_list_K1_1layer[i],g_vec_list_K1_2layer[j])
    

                # print("i = ",i+num_gn,"j = ",j+num_gn)
    print(" abs(C2 G matrix) = ",np.abs(C2).sum())

    return C2

def C2_MoTe2_all(C2_Gn,axis):

    C2_rot_matrix = rotate_mat(np.concatenate([axis,[0]]),np.pi)
    C2_spin = spin_reps(C2_rot_matrix)
    # print(C2_spin)
    # print(C2_spin*1j*sigma_y)
    C2T_spin = C2_spin*1j*sigma_y
    C2_s = get_any_rot_orb_twostep('s',C2_rot_matrix)
    C2_p = get_any_rot_orb_twostep('p',C2_rot_matrix)
    C2_d = get_any_rot_orb_twostep('d',C2_rot_matrix) 

    C2_Te_rep = direct_sum(C2_s,C2_s,C2_s,C2_p,C2_p,C2_d,C2_d)
    C2_Mo_rep = direct_sum(C2_s,C2_s,C2_s,C2_p,C2_p,C2_d)

    C2T_all_rep = direct_sum(np.kron(C2T_spin,C2_Te_rep),np.kron(C2T_spin,C2_Te_rep),np.kron(C2T_spin,C2_Mo_rep))

    C2T_all_rep = np.kron(C2_Gn,C2T_all_rep)
    return C2T_all_rep
