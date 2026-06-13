import numpy as np
import scipy
from tqdm import tqdm
import sys
import os

import numpy as np
import scipy
from tqdm import tqdm
import time
import psutil
from datetime import datetime
from functools import wraps
Hartree = 27.21138602435532
_INTERNAL_OPENMX_TAPW_BAND_TAG = "A.tapw_band_from_" + "li" + "jh"


def _timing_enabled() -> bool:
    value = os.environ.get("TAPW_ENABLE_TIMING", "")
    return value.lower() in {"1", "true", "yes", "on"}

def timing_decorator_factory(process_id):
    def timing_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if process_id == 0 and _timing_enabled():
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
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return timing_decorator

class HrSparseHandler:
    def __init__(self, file_name=None, npz_file_name=None, A=None,read_from_npz=False):
        self.file_name = file_name
        self.npz_file_name = npz_file_name
        self.A = A
        self.hr_sparse = None
        self.read_from_npz = read_from_npz
    @timing_decorator_factory(0)
    def read_txt_file(self):
        hr_sparse_chunk = {}
        rvec_list_chunk = []

        with open(self.file_name, 'r') as f:
            f.readline()
            num_nonzero = int(f.readline().strip().split()[0])
            nwann = int(f.readline().strip().split()[0])
            self.nwann = nwann
            nrpt = int(f.readline().strip().split()[0])
            print("nrpt nwann num_nonzero", nrpt, nwann, num_nonzero)
            print("loading hamr...")

            for line in tqdm(f, total=num_nonzero):
                if len(line.strip().split()) == 7:
                    rx, ry, rz, hr_m, hr_n, hr_real, hr_imag = line.strip().split()
                    rvec = (rx, ry, rz)
                    if rvec not in rvec_list_chunk:
                        rvec_list_chunk.append(rvec)
                        hr_sparse_chunk[rvec] = {"col": [], "row": [], "val": [], "hr": [], "real": [], "imag": []}

                    hr_sparse_chunk[rvec]["row"].append(hr_m)
                    hr_sparse_chunk[rvec]["col"].append(hr_n)
                    hr_sparse_chunk[rvec]["real"].append(hr_real)
                    hr_sparse_chunk[rvec]["imag"].append(hr_imag)
                else:
                    print("line = ", line)
                    # break

            for key in tqdm(hr_sparse_chunk.keys()):
                hr_sparse_chunk[key]["row"] = np.int32(hr_sparse_chunk[key]["row"]) - 1
                hr_sparse_chunk[key]["col"] = np.int32(hr_sparse_chunk[key]["col"]) - 1
                hr_sparse_chunk[key]["val"] = np.float64(hr_sparse_chunk[key]["real"]) + 1j * np.float64(hr_sparse_chunk[key]["imag"])
                # print("A shape",self.A.shape)
                # print("nwann",nwann)
                if self.A is not None:
                    hr_sparse_chunk[key]["hr"] = self.A.dot(
                        scipy.sparse.csr_matrix((hr_sparse_chunk[key]["val"], (hr_sparse_chunk[key]["row"], hr_sparse_chunk[key]["col"])), shape=(nwann, nwann)).dot(self.A.T)
                    )
                    hr_sparse_chunk[key]["row"], hr_sparse_chunk[key]["col"] = hr_sparse_chunk[key]["hr"].nonzero()
                    hr_sparse_chunk[key]["val"] = hr_sparse_chunk[key]["hr"].data
                # else:
                #     hr_sparse_chunk[key]["hr"] = scipy.sparse.csr_matrix((hr_sparse_chunk[key]["val"], (hr_sparse_chunk[key]["row"], hr_sparse_chunk[key]["col"])), shape=(nwann, nwann))
                


            new_hr_sparse = {}
            for key in tqdm(hr_sparse_chunk.keys()):
                rvec = tuple([int(key[0]), int(key[1]), int(key[2])])
                new_hr_sparse[rvec] = hr_sparse_chunk[key]
                del new_hr_sparse[rvec]["hr"]
                del new_hr_sparse[rvec]["real"]
                del new_hr_sparse[rvec]["imag"]

        self.hr_sparse = new_hr_sparse

        npz_file_name = self.file_name.replace('.dat', '.npz')
        self.save_to_npz(npz_file_name)

    def save_to_npz(self, file_path):
        storable_data = {}
        for key, value in self.hr_sparse.items():
            storable_data[f"{key}_row"] = value['row']
            storable_data[f"{key}_col"] = value['col']
            storable_data[f"{key}_val"] = value['val']
        np.savez(file_path, **storable_data)

    @timing_decorator_factory(0)
    def load_from_npz(self, file_path):
        loaded_data = np.load(file_path, allow_pickle=True)
        data = {}
        for key in loaded_data.files:
            key_tuple_str, attribute = key.rsplit('_', 1)
            key_tuple = tuple(map(int, key_tuple_str.strip('()').split(', ')))
            if key_tuple not in data:
                data[key_tuple] = {'row': None, 'col': None, 'val': None}
            if attribute == 'row':
                data[key_tuple]['row'] = loaded_data[key]
            elif attribute == 'col':
                data[key_tuple]['col'] = loaded_data[key]
            elif attribute == 'val':
                if ('deeph-pack' in file_path or 'DeepH-pack' in file_path) and 'H.npz' in file_path:
                    data[key_tuple]['val'] = loaded_data[key] * Hartree
                else:
                    data[key_tuple]['val'] = loaded_data[key]
        self.hr_sparse = data
    
    @timing_decorator_factory(0)
    def load_from_npz_new(self, file_path):
        loaded_data = np.load(file_path, allow_pickle=True)
        data = {}
        for key in loaded_data.files:
            key_tuple_str, attribute = key.rsplit('_', 1)
            key_tuple = tuple(map(int, key_tuple_str.strip('()').split(', ')))
            if key_tuple not in data:
                data[key_tuple] = {'row': None, 'col': None, 'val': None}
            if attribute == 'row':
                data[key_tuple]['row'] = loaded_data[key]
            elif attribute == 'col':
                data[key_tuple]['col'] = loaded_data[key]
            elif attribute == 'val':
                if 'deeph-pack' in file_path and 'H.npz' in file_path:
                    print("loaded data from deeph-pack")
                    data[key_tuple]['val'] = loaded_data[key]# * Hartree
                elif 'DeepH-pack' in file_path and 'H.npz' in file_path:
                    print("loaded data from DeepH-pack")
                    data[key_tuple]['val'] = loaded_data[key] * Hartree
                elif _INTERNAL_OPENMX_TAPW_BAND_TAG in file_path and 'H.dat' not in file_path:
                    print("loaded data from internal OpenMX TAPW band archive")
                    data[key_tuple]['val'] = loaded_data[key] * Hartree
                elif "Z.hr_sr_mat_openmx_recalc_from_relaxed_str" in file_path and 'H.npz' in file_path:
                    print("loaded data from Z.hr_sr_mat_openmx_recalc_from_relaxed_str")
                    data[key_tuple]['val'] = loaded_data[key] * Hartree
                else:
                    print("loaded data from openmx file")
                    data[key_tuple]['val'] = loaded_data[key]
        self.hr_sparse = data
        
        hr_sparse_chunk = data 
        for key in tqdm(hr_sparse_chunk.keys()):
                nwann = np.unique(hr_sparse_chunk[(0,0,0)]['row']).shape[0]
                # print("nwann = ",nwann)
                if self.A is not None:
                    hr_sparse_chunk[key]["hr"] = self.A.dot(
                        scipy.sparse.csr_matrix((hr_sparse_chunk[key]["val"], (hr_sparse_chunk[key]["row"], hr_sparse_chunk[key]["col"])), shape=(nwann, nwann)).dot(self.A.T)
                    )
                    hr_sparse_chunk[key]["row"], hr_sparse_chunk[key]["col"] = hr_sparse_chunk[key]["hr"].nonzero()
                    hr_sparse_chunk[key]["val"] = hr_sparse_chunk[key]["hr"].data
                # else:
                #     hr_sparse_chunk[key]["hr"] = scipy.sparse.csr_matrix((hr_sparse_chunk[key]["val"], (hr_sparse_chunk[key]["row"], hr_sparse_chunk[key]["col"])), shape=(nwann, nwann))
                


        new_hr_sparse = {}
        for key in tqdm(hr_sparse_chunk.keys()):
            rvec = tuple([int(key[0]), int(key[1]), int(key[2])])
            new_hr_sparse[rvec] = hr_sparse_chunk[key]
            del new_hr_sparse[rvec]["hr"]
            # del new_hr_sparse[rvec]["real"]
            # del new_hr_sparse[rvec]["imag"]
        self.hr_sparse = new_hr_sparse
        # self.save_to_npz(self.npz_file_name)

    def get_hr_sparse(self):
        if self.file_name.endswith('.dat'):
            npz_file_name = self.file_name.replace('.dat', '.npz')
        else:
            npz_file_name = self.npz_file_name
        if self.read_from_npz or os.path.exists(self.npz_file_name) or os.path.exists(npz_file_name):
            # self.load_from_npz(npz_file_name)
            if 'deeph-pack' in self.npz_file_name or 'DeepH-pack' in self.npz_file_name or _INTERNAL_OPENMX_TAPW_BAND_TAG in self.npz_file_name or "Z.hr_sr_mat_openmx_recalc_from_relaxed_str" in self.npz_file_name:
                self.load_from_npz_new(npz_file_name)
            else:
                self.load_from_npz(npz_file_name)
            # self.load_from_npz_new(self.npz_file_name)
        else:
            self.read_txt_file()
        return self.hr_sparse
