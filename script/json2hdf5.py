#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" JSON to HDF5 converter
"""

import os
import concurrent
import concurrent.futures
import tqdm
import numpy as np
import json
import msgpack
import h5py

_DEBUG = True
FMT_CHUNKID = "%.8d"
GROUP_CHUNKMAP = "chunkmap"
GROUP_CHUNKED = "chunked"


def doit_parallel(files, profile, verbose):
    "Parallel execution of json2hdf5 for given list of files"
    # IMPORTANT: use process pool rather than thread pool for performance
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_list = []
        for f in files:
            if os.path.exists(f) and os.path.splitext(f)[1] == ".json":
                future = executor.submit(json2hdf5, f, profile, verbose=verbose)
                future_list.append(future)
        # show progress bar
        progress_bar = tqdm.tqdm(total=len(future_list), desc="Generating HDF5")
        progress_bar.update(0)
        for f in concurrent.futures.as_completed(future_list):
            progress_bar.update(1)


def json2hdf5(jsonfile, profile, hdffile=None, verbose=True):
    "Generate HDF5 file from given JSON file"
    jsondir = os.path.dirname(jsonfile)
    if jsondir == "":
        jsondir = "."

    if hdffile is None:
        hdffile = os.path.splitext(jsonfile)[0] + ".h5"

    if verbose:
        print("Processing {} to produce {}".format(jsonfile, hdffile))

    # read json and store in memory
    with open(jsonfile, "r") as fp:
        obj = json.load(fp)

    # get meta data
    try:
        meta = obj.get("meta")

        # check endian
        endian = meta.get("endian")
        if endian == 1:  # little endian
            byteorder = "<"
        elif endian == 16777216:  # big endian
            byteorder = ">"
        else:
            errmsg = "unrecognized endian flag: {}".format(endian)
            report_error(errmsg)

        # check raw data file
        rawfile = meta.get("rawfile")
        datafile = os.sep.join([jsondir, rawfile])
        if not os.path.exists(datafile):
            errmsg = "rawfile {} does not exist".format(datafile)
            report_error(errmsg)

        # check array order
        order = meta.get("order", 0)
    except Exception:
        print("ignoring {}".format(jsonfile))
        return

    #
    # create hdf5 and add attribute
    #
    if os.path.exists(hdffile):
        os.remove(hdffile)
    with h5py.File(hdffile, "w", libver="latest") as h5fp:
        for key, item in meta.items():
            h5fp.attrs.create(key, item)

    # create dataset
    group = ""
    extpath = rawfile
    add_dataset(obj, hdffile, datafile, extpath, byteorder, order, group, verbose)

    # create virtual dataset for chunked dataset
    with open(profile, "rb") as f:
        data = msgpack.unpack(f)
        chunkmap = data.get("chunkmap", None)
    chunk = add_chunkmap(chunkmap, hdffile, GROUP_CHUNKMAP, verbose)

    if chunk:
        dataset = create_chunked_dataset(hdffile, GROUP_CHUNKMAP, GROUP_CHUNKMAP, verbose)
        create_vds(hdffile, GROUP_CHUNKMAP, GROUP_CHUNKED, dataset, verbose)

    if verbose:
        print("done !")


def add_dataset(obj, hdffile, datafile, extpath, byteorder, order, group="", verbose=True):
    "Add dataset to HDF5 file"
    with h5py.File(hdffile, "r+", libver="latest") as h5fp, open(datafile, "r") as datafp:
        # attribute
        attribute = obj.get("attribute", [])
        for attr_name in attribute:
            attr = attribute.get(attr_name)
            data = read_data(datafp, attr, byteorder)
            name = "/".join([group, attr_name])
            h5fp.attrs.create(name, data)
            if verbose:
                print('  - attribute "{}" has been created '.format(name), end="")
                print("with data : {}".format(data))

        # dataset
        dataset = obj.get("dataset", [])
        for ds_name in dataset:
            data = dataset.get(ds_name)
            offset, dsize, dtype, shape = read_info(data, byteorder)
            if order == 0:
                shape = shape[::-1]
            ext = ((extpath, offset, dsize),)
            name = "/".join([group, ds_name])
            h5fp.create_dataset(
                name, shape=shape, dtype=dtype, external=ext, efile_prefix="${ORIGIN}"
            )
            if verbose:
                print(
                    '  - dataset "{}" has been created '
                    'with dtype = "{}" and shape = "{}"'.format(name, dtype, shape)
                )


def add_chunkmap(chunkmap, hdffile, group="", verbose=True):
    "Add chunkmap to HDF5 file"
    if chunkmap is None:
        return False

    with h5py.File(hdffile, "r+", libver="latest") as h5fp:
        for ds_name in ("coord", "chunkid"):
            if not ds_name in chunkmap:
                continue
            data = np.array(chunkmap.get(ds_name))
            name = "/".join([group, ds_name])
            h5fp.create_dataset(name, data=data)
            if verbose:
                print('  - dataset "{}" has been created '.format(name), end="")
                print('with dtype = "{}" and shape = "{}"'.format(data.dtype, data.shape))

    return True


def is_chunked_dataset(dataset, cshape):
    "Return true if the given dataset is chunked one"
    dshape = dataset.shape
    status = dshape[0] == np.prod(cshape)
    return status


def is_external_contiguous(external):
    "Return true if the external tuple describes contiguous block in disk"
    status = True
    filename, offset, size = external[0]
    for ext in external[1:]:
        status = status and (filename == ext[0])
        status = status and (offset + size == ext[1])
        offset = ext[1]
        size = ext[2]
    return status


def create_chunked_dataset(hdffile, chunkmap, chunked, verbose=True):
    "Create chunked dataset"
    chunked_dataset = []
    with h5py.File(hdffile, "r+", libver="latest") as h5fp:
        cmap = h5fp.get(chunkmap, None)
        csh = cmap.get("chunkid")[()].shape
        for name, data in h5fp.items():
            #
            # check dataset
            #
            if name == chunkmap:
                # ignore chunkmap
                continue
            if not is_chunked_dataset(data, csh):
                # check if the dataset may be divided into chunks
                continue
            if not is_external_contiguous(data.external):
                # check if the external storage is contiguous in disk
                continue
            #
            # now create chunked dataset
            #
            dshape = data.shape
            datafile = data.external[0][0]
            chunked_offset = data.external[0][1]
            for i in range(dshape[0]):
                chunked_name = "/".join([chunked, ("%s" + FMT_CHUNKID) % (name, i)])
                chunked_shape = dshape[1:]
                chunked_dtype = data.dtype
                chunked_dsize = np.prod(chunked_shape) * chunked_dtype.itemsize
                chunked_ext = ((datafile, chunked_offset, chunked_dsize),)
                kwargs = {
                    "name": chunked_name,
                    "shape": chunked_shape,
                    "dtype": chunked_dtype,
                    "external": chunked_ext,
                }
                h5fp.create_dataset(**kwargs)
                # move offset
                chunked_offset = chunked_offset + chunked_dsize
                if verbose:
                    print(
                        '  - chunked dataset "{}" has been created '
                        'with dtype = "{}" and shape = "{}"'.format(
                            chunked_name, chunked_dtype, chunked_shape
                        )
                    )
            chunked_dataset.append(name)
    return chunked_dataset


def create_vds(hdffile, chunkmap, chunked, dataset, verbose=True):
    "Create VDS from given chunked datasets"
    group = "vds"
    with h5py.File(hdffile, "r+", libver="latest") as h5fp:
        # get chunk ID and coordinate
        chunkid = h5fp.get("/".join([chunkmap, "chunkid"]))[()]
        coord = h5fp.get("/".join([chunkmap, "coord"]))[()]
        # try to create VDS for each dataset
        for ds in dataset:
            prefix = "/".join([chunked, ds]) + FMT_CHUNKID
            srcfile = [prefix % i for i in range(chunkid.size)]
            srcdata = [h5fp.get(src) for src in srcfile]
            #
            # determine data shape assuming 3D chunk
            #
            csh = list(srcdata[0].shape)
            if len(csh) < 3:  # ignore dataset with dimensions < 3
                continue
            gsh = csh.copy()
            gsh[0] = csh[0] * chunkid.shape[0]
            gsh[1] = csh[1] * chunkid.shape[1]
            gsh[2] = csh[2] * chunkid.shape[2]
            csh = tuple(csh)  # shape of each chunk
            gsh = tuple(gsh)  # shape of global array
            dtype = srcdata[0].dtype  # datatype
            # create VirtualLayout object and assign chunks
            layout = h5py.VirtualLayout(shape=gsh, dtype=dtype)
            for iz in range(chunkid.shape[0]):
                for iy in range(chunkid.shape[1]):
                    for ix in range(chunkid.shape[2]):
                        ii = chunkid[iz, iy, ix]
                        cx = coord[ii, 0]
                        cy = coord[ii, 1]
                        cz = coord[ii, 2]
                        xslice = slice(cx * csh[2], (cx + 1) * csh[2])
                        yslice = slice(cy * csh[1], (cy + 1) * csh[1])
                        zslice = slice(cz * csh[0], (cz + 1) * csh[0])
                        src = h5py.VirtualSource(srcdata[ii])
                        layout[zslice, yslice, xslice, :] = src
            # create VirtualDataset using the layout
            vds_name = "/".join([group, ds])
            h5fp.create_virtual_dataset(vds_name, layout)
            if verbose:
                print(
                    '  - virtual dataset "{}" has been created '
                    'with dtype = "{}" and shape = "{}"'.format(vds_name, dtype, gsh)
                )


def read_data(fp, obj, byteorder):
    "Read data content from disk"
    offset = obj["offset"]
    datatype = byteorder + obj["datatype"]
    shape = obj["shape"]
    size = np.product(shape)
    fp.seek(offset)
    x = np.fromfile(fp, datatype, size).reshape(shape)
    if len(shape) == 1 and shape[0] == 1:
        x = x[0]
    return x


def read_info(obj, byteorder):
    "Read offset, datasize, datatype, and shape of given data"
    offset = obj["offset"]
    datatype = byteorder + obj["datatype"]
    shape = obj["shape"]
    datasize = np.product(shape) * np.dtype(datatype).itemsize
    return offset, datasize, datatype, shape


def report_error(msg):
    "Print error message"
    print("Error: {}".format(msg))
    if _DEBUG:
        raise ValueError(msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--profile",
        dest="profile",
        type=str,
        required=True,
        help="Filename for profile in msgpack format containing chunkmap information",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print verbose messages",
    )
    parser.add_argument("jsonfile", nargs="*", help="Input json format files")
    args = parser.parse_args()
    doit_parallel(args.jsonfile, args.profile, args.verbose)
