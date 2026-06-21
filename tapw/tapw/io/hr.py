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

    @staticmethod
    def _parse_npz_rvec_key(key_tuple_str):
        inner = key_tuple_str.strip().strip("()")
        parts = [part.strip() for part in inner.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid NPZ real-space key: {key_tuple_str}")
        return tuple(int(part) for part in parts)

    def _scale_npz_values(self, file_path, values):
        file_path = str(file_path)
        if 'deeph-pack' in file_path and 'H.npz' in file_path:
            return values
        if 'DeepH-pack' in file_path and 'H.npz' in file_path:
            return values * Hartree
        if _INTERNAL_OPENMX_TAPW_BAND_TAG in file_path and 'H.dat' not in file_path:
            return values * Hartree
        if "Z.hr_sr_mat_openmx_recalc_from_relaxed_str" in file_path and 'H.npz' in file_path:
            return values * Hartree
        return values

    @staticmethod
    def _is_symm_npz(file_path):
        name = os.path.basename(str(file_path))
        return name in {"H_symm.npz", "S_symm.npz"}

    def _apply_basis_transform(self, hr_sparse_chunk):
        if self.A is None:
            return hr_sparse_chunk

        nwann = 0
        for item in hr_sparse_chunk.values():
            row = np.asarray(item["row"], dtype=np.int64)
            col = np.asarray(item["col"], dtype=np.int64)
            if row.size:
                nwann = max(nwann, int(row.max()) + 1)
            if col.size:
                nwann = max(nwann, int(col.max()) + 1)

        transformed = {}
        for key in sorted(hr_sparse_chunk.keys()):
            row = np.asarray(hr_sparse_chunk[key]["row"], dtype=np.int64)
            col = np.asarray(hr_sparse_chunk[key]["col"], dtype=np.int64)
            val = np.asarray(hr_sparse_chunk[key]["val"], dtype=np.complex128)
            matrix = scipy.sparse.csr_matrix((val, (row, col)), shape=(nwann, nwann))
            rotated = self.A.dot(matrix.dot(self.A.T))
            new_row, new_col = rotated.nonzero()
            transformed[key] = {
                "row": np.asarray(new_row, dtype=np.int64),
                "col": np.asarray(new_col, dtype=np.int64),
                "val": np.asarray(rotated.data, dtype=np.complex128),
            }
        return transformed

    def _load_npz_common(self, file_path):
        loaded_data = np.load(file_path, allow_pickle=True)
        data = {}
        for key in loaded_data.files:
            key_tuple_str, attribute = key.rsplit('_', 1)
            key_tuple = self._parse_npz_rvec_key(key_tuple_str)
            if key_tuple not in data:
                data[key_tuple] = {'row': None, 'col': None, 'val': None}
            if attribute == 'row':
                data[key_tuple]['row'] = np.asarray(loaded_data[key], dtype=np.int64)
            elif attribute == 'col':
                data[key_tuple]['col'] = np.asarray(loaded_data[key], dtype=np.int64)
            elif attribute == 'val':
                data[key_tuple]['val'] = np.asarray(
                    self._scale_npz_values(file_path, loaded_data[key]),
                    dtype=np.complex128,
                )

        for key in data:
            if data[key]['row'] is None or data[key]['col'] is None or data[key]['val'] is None:
                raise ValueError(f"Incomplete NPZ sparse block for rvec={key} in {file_path}")

        self.hr_sparse = self._apply_basis_transform(data)

    @timing_decorator_factory(0)
    def load_from_npz(self, file_path):
        loaded_data = np.load(file_path, allow_pickle=True)
        data = {}
        for key in loaded_data.files:
            key_tuple_str, attribute = key.rsplit('_', 1)
            key_tuple = self._parse_npz_rvec_key(key_tuple_str)
            if key_tuple not in data:
                data[key_tuple] = {'row': None, 'col': None, 'val': None}
            if attribute == 'row':
                data[key_tuple]['row'] = loaded_data[key]
            elif attribute == 'col':
                data[key_tuple]['col'] = loaded_data[key]
            elif attribute == 'val':
                if ('deeph-pack' in str(file_path) or 'DeepH-pack' in str(file_path)) and 'H.npz' in str(file_path):
                    data[key_tuple]['val'] = loaded_data[key] * Hartree
                else:
                    data[key_tuple]['val'] = loaded_data[key]
        self.hr_sparse = data
    
    @timing_decorator_factory(0)
    def load_from_npz_new(self, file_path):
        self._load_npz_common(file_path)

    def get_hr_sparse(self):
        if self.file_name.endswith('.dat'):
            npz_file_name = self.file_name.replace('.dat', '.npz')
        else:
            npz_file_name = self.npz_file_name
        if self.read_from_npz or os.path.exists(self.npz_file_name) or os.path.exists(npz_file_name):
            if self._is_symm_npz(npz_file_name):
                self._load_npz_common(npz_file_name)
            elif 'deeph-pack' in self.npz_file_name or 'DeepH-pack' in self.npz_file_name or _INTERNAL_OPENMX_TAPW_BAND_TAG in self.npz_file_name or "Z.hr_sr_mat_openmx_recalc_from_relaxed_str" in self.npz_file_name:
                self.load_from_npz_new(npz_file_name)
            else:
                self.load_from_npz(npz_file_name)
        else:
            self.read_txt_file()
        return self.hr_sparse
