#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import json
import glob

def read_config(cfg):
	obj = json.loads(open(cfg, 'r').read())
	Nx = obj['parameter']['Nx']
	Ny = obj['parameter']['Ny']
	Nz = obj['parameter']['Nz']
	delt = obj['parameter']['delt']
	delh = obj['parameter']['delh']
	for d in obj['diagnostic']:
		if d['name'] == 'field':
			delt *= d['interval']
			filename = os.sep.join([d['path'], d['prefix']])
	files = glob.glob(filename + '*.h5')
	files.sort()
	return files, Nz, Ny, Nx, delt, delh

def read_emf(files, Nz, Ny, Nx):
	Nt = len(files)
	uf = np.zeros((Nt, Nz, Ny, Nx, 6), dtype=np.float64)
	for i in range(Nt):
		with h5py.File(files[i], 'r') as h5fp:
			uf[i,...] = h5fp.get('/vds/uf')[()]
	return uf

def get_wk_spectrum(f, delt=1.0, delh=1.0):
	if f.ndim != 2:
		raise ValueError('Input must be a 2D array')
	if f.dtype == np.float32 or f.dtype == np.float64:
		# real
		Nt = f.shape[0]
		Nx = f.shape[1]
		P = np.abs(np.fft.fftshift(np.fft.rfft2(f, norm='ortho'), axes=(0,)))**2
		w = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(Nt, delt))
		k = 2*np.pi * np.fft.rfftfreq(Nx, delh)
		W, K = np.broadcast_arrays(w[:,None], k[None,:])
		return P, W, K
	else:
		# complex
		Nt = f.shape[0]
		Nx = f.shape[1]
		P = np.abs(np.fft.fftshift(np.fft.fft2(f, norm='ortho'), axes=(0, 1)))**2
		w = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(Nt, delt))
		k = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, delh))
		W, K = np.broadcast_arrays(w[:,None], k[None,:])
		return P, W, K

def plot_wk_spectrum(P, W, K, filename, title, **kwargs):
	import matplotlib
	from matplotlib import pyplot as plt

	figure = plt.figure()
	im = plt.pcolormesh(K, W, np.log10(P), shading='nearest')
	cl = plt.colorbar(im)
	# xlim
	kmax = kwargs.get('kmax', 0.25 * K.max())
	kmin = kwargs.get('kmin', 0.25 * K.min())
	plt.xlim(kmin, kmax)
	plt.xlabel(r'$k$')
	# ylim
	wmax = kwargs.get('wmax', W.max())
	wmin = kwargs.get('wmin', W.min())
	plt.ylim(wmin, wmax)
	plt.ylabel(r'$\omega$')
	# clim
	cmax = kwargs.get('cmin', np.log10(P.max()))
	cmin = kwargs.get('cmin', cmax - 4)
	plt.clim(cmin, cmax)
	# save
	plt.title(title)
	plt.savefig(filename)

if __name__ == '__main__':
	pass