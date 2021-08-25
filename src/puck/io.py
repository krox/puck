""" utilities for json/pickle/hdf5 files  """

import glob
import numpy as np
import pickle
import json
import sys
import os
from copy import copy


def write_pickle(filename, obj):
    with open(filename, "wb") as file:
        # protocol 4 supported since python 3.4 (but only default since 3.8)
        pickle.dump(obj, file, protocol=4)


def read_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def write_json(filename, obj):
    with open(filename, "w") as file:
        json.dump(obj, file)


def read_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def joinFiles(path, filename, chunksize=10, dset_filter=lambda s: True, file_limit=1000000000, force=False):
    """join multiple hdf5 files with the same structure by adding a new first axis to every dataset"""

    from natsort import natsorted
    import progressbar as pb
    import h5py

    filenames = natsorted(glob.glob(path))

    print(f"found {len(filenames)} files")
    if len(filenames) == 0:
        return
    if len(filenames) > file_limit:
        filenames = filenames[:file_limit]
    print(f"template: {filenames[0]}")

    filesize = os.path.getsize(filenames[0])
    for fn in copy(filenames):
        if abs(os.path.getsize(fn) - filesize) > filesize * 0.05:
            print(f"SUSPICIOUS FILE SIZE: {fn} ")
            filenames.remove(fn)
            continue

    # read template and create some buffers
    buffer_list = []
    tot_size = 0
    with h5py.File(filenames[0], "r") as template:
        dataset_list = [s for s in list(template) if dset_filter(s)]
        for dataset in dataset_list:
            dtype = template[dataset].dtype
            shape = template[dataset].shape
            tot_size += dtype.itemsize * np.prod(shape)
            buffer_list.append(np.zeros((chunksize, *shape), dtype=dtype))
    print(f"found {len(dataset_list)} datasets")
    print(f"{tot_size/1024.**2:.2f} MiB per file, {len(dataset_list)*tot_size/1024.**3:.2f} GiB total")

    with h5py.File(filename, "w" if force else "w-") as file_out:

        # create output file
        for i in range(len(dataset_list)):
            chunks = buffer_list[i].shape
            shape = (len(filenames), *buffer_list[i].shape[1:])
            dtype = buffer_list[i].dtype
            file_out.create_dataset(
                dataset_list[i], shape=shape, dtype=dtype, fletcher32=True, chunks=chunks)

        sys.stdout.flush()
        sys.stderr.flush()
        pbar = pb.ProgressBar(maxval=len(filenames)).start()
        for k1 in range(0, len(filenames), chunksize):
            k2 = min(k1 + chunksize, len(filenames))
            for k in range(k1, k2):
                with h5py.File(filenames[k], "r") as file:
                    for j in range(len(dataset_list)):
                        buffer_list[j][k - k1] = file[dataset_list[j]][:]
                pbar.update(k + 1)

            for j in range(len(dataset_list)):
                file_out[dataset_list[j]][k1:k2] = buffer_list[j][0:k2 - k1]

        pbar.finish()
        sys.stdout.flush()
        sys.stderr.flush()
