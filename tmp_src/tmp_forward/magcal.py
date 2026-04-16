# coding: utf-8

import numpy as np
import pandas as pd

def di2xyz(dec, inc):
	'''
	偏角(dec: degree)・伏角(inc: degree)から単位ベクトル作成
	x = cos(inc) * sin(dec)
	y = cos(inc) * cos(dec)
	z = -sin(inc)
	'''
	dec_rad = dec * np.pi / 180.
	inc_rad = inc * np.pi / 180.

	x =  np.cos(inc_rad) * np.sin(dec_rad)
	y =  np.cos(inc_rad) * np.cos(dec_rad)
	z = -np.sin(inc_rad)

	return np.array([x, y, z])

def i2yz(inc):
	'''
	伏角(inc: degree)から単位ベクトル作成
	y = cos(inc)
	z = -sin(inc)
	'''
	inc_rad = inc * np.pi / 180.

	y =  np.cos(inc_rad)
	z = -np.sin(inc_rad)

	return np.array([y, z])

def _dipole_yz_kernel_func_ (mgz, y, z):
	'''
	2次元ダイポールによる磁気異常の磁場3成分を計算(hx=0)
	mgz: 磁化ベクトル(0, jy, jz)
	y: y座標 (ysrc - yobs)
	z: z座標 (zsrc - zobs)
	'''
	r2 = y**2 + z**2
	r4 = np.power (r2, 2.)

	jy = mgz[0]
	jz = mgz[1]

	hy = jy * (1. / r2 - 2. * y**2 / r4) - jz * 2. * y * z / r4
	hz = - jy * 2. * y * z / r4 + jz * (1. / r2 - 2. * z**2 / r4)

	h = np.c_[- 2. * hy, - 2. * hz]

	return 100. * h


def dipole_yz(mgz, yobs, zobs, ysrc, zsrc):
	'''
	ダイポールによる磁気異常の磁場3成分を計算(hx=0)
	mgz: 磁化ベクトル(0, jy, jz)
	yobs, zobs: 観測点座標
	ysrc, zsrc: ソース位置座標
	'''
	h = _dipole_yz_kernel_func_(mgz, ysrc - yobs, zsrc - zobs)
	return h


def _prism_yz_kernel_func_ (mgz, y, z):
	'''
	2次元磁化ブロックによる磁気異常の磁場3成分(不定積分)を計算(hx=0)
	mgz: 磁化ベクトル(0, jy, jz)
	y: y座標 (ysrc - yobs)
	z: z座標 (zsrc - zobs)
	'''
	r = np.sqrt (y**2 + z**2)

	jy = mgz[0]
	jz = mgz[1]

	hy = jy * np.arctan (z / y) + jz * np.log (r)
	hz = jy * np.log (r) + jz * np.arctan (y / z)

	h = np.c_[- 2. * hy, - 2. * hz]

	return 100. * h


def prism_yz(mgz, yobs, zobs, ysrc, zsrc, dim):
	'''
	単位磁化したブロックによる磁気異常の磁場3成分(不定積分)を計算(hx=0)
	mgz: 磁化ベクトル(0, jy, jz)
	yobs, zobs: 観測点座標
	ysrc, zsrc: ソース位置座標
	dim: ブロックのサイズ
		dim[0] = [dy1, dy2]:
				dy1: ブロック中心からy軸方向の左端までの長さ(>0)
				dy2: ブロック中心からy軸方向の右端までの長さ(>0)
		dim[1] = [dz1, dz2]:
				dz1: ブロック中心からz軸方向の下端までの長さ(>0)
				dz2: ブロック中心からz軸方向の上端までの長さ(>0)
	'''
	dy = dim[0]
	dz = dim[1]

	ys = [ysrc - dy[0], ysrc + dy[1]]
	zs = [zsrc - dz[0], zsrc + dz[1]]

	h = np.zeros([len(yobs), 2])
	flag = 1.0
	for j in range(2):
		flag = -flag
		y = ys[j] - yobs
		for k in range(2):
			flag = -flag
			z = zs[k] - zobs

			dh = _prism_yz_kernel_func_(mgz, y, z)
			h += flag * dh

	return h

def prism_yz2(mgz, yobs, zobs, ysrc, zsrc, dim):
	'''
	単位磁化したブロックによる磁気異常の磁場3成分(不定積分)を計算(hx=0)
	mgz: 磁化ベクトル(0, jy, jz)
	yobs, zobs: 観測点座標
	ysrc, zsrc: ソース位置座標
	dim: ブロックのサイズ
		dim[0] = [dy1, dy2]:
				dy1: ブロック中心からy軸方向の左端までの長さ(>0)
				dy2: ブロック中心からy軸方向の右端までの長さ(>0)
		dim[1] = [dz1, dz2]:
				dz1: ブロック中心からz軸方向の下端までの長さ(>0)
				dz2: ブロック中心からz軸方向の上端までの長さ(>0)
	'''
	dy = dim[0]
	dz = dim[1]

	ys = [ysrc - dy[0], ysrc + dy[1]]
	zs = [zsrc - dz[0], zsrc + dz[1]]

	h = np.zeros([len(ysrc), 2])
	flag = 1.0
	for j in range(2):
		flag = -flag
		y = ys[j] - yobs
		for k in range(2):
			flag = -flag
			z = zs[k] - zobs

			dh = _prism_yz_kernel_func_(mgz, y, z)
			h += flag * dh

	return h


def _dipole_kernel_func_(mgz, x, y, z):
	'''
	ダイポールによる磁気異常の磁場3成分を計算
	mgz: 磁化ベクトル(jx, jy, jz)
	x: 観測点〜ソースの位置ベクトルx座標 (xsrc - xobs)
	y: y座標 (ysrc - yobs)
	z: z座標 (zsrc - zobs)
	'''
	r  = np.sqrt (x**2 + y**2 + z**2)
	r3 = np.power (r, 3.)
	r5 = np.power (r, 5.)

	jx = mgz[0]
	jy = mgz[1]
	jz = mgz[2]

	hx = (
		- jx * (1.0 / r3 - 3. * x**2 / r5)
		+ jy * (3.0 * x * y / r5)
		+ jz * (3.0 * x * z / r5)
	)

	hy = (
		  jx * (3.0 * y * x / r5)
		- jy * (1.0 / r3 - 3. * y**2 / r5)
		+ jz * (3.0 * y * z / r5)
	)

	hz = (
		  jx * (3.0 * z * x / r5)
		+ jy * (3.0 * z * y / r5)
		- jz * (1.0 / r3 - 3. * z**2 / r5)
	)

	h = np.c_[hx, hy, hz]
	return 100. * h


def dipole(mgz, xobs, yobs, zobs, xsrc, ysrc, zsrc):
	'''
	ダイポールによる磁気異常の磁場3成分を計算
	mgz: 磁化ベクトル(jx, jy, jz)
	xobs, yobs, zobs: 観測点座標
	xsrc, ysrc, zsrc: ソース位置座標
	'''

	h = _dipole_kernel_func_(mgz, xsrc - xobs, ysrc - yobs, zsrc - zobs)
	return h


def _prism_kernel_func_(mgz, x, y, z):
	'''
	磁化ブロックによる磁気異常の磁場3成分(不定積分)を計算
	mgz: 磁化ベクトル(jx, jy, jz)
	x: 観測点〜ソースの位置ベクトルx座標 (xsrc - xobs)
	y: y座標 (ysrc - yobs)
	z: z座標 (zsrc - zobs)
	'''
	r  = np.sqrt (x**2 + y**2 + z**2)
	r3 = np.power (r, 3.)
	r5 = np.power (r, 5.)

	jx = mgz[0]
	jy = mgz[1]
	jz = mgz[2]

	idx = np.arange(len(r)).astype(int)

	lnx = np.zeros(len(r))
	idx1 = np.where(np.abs(r + x) > np.finfo(float).eps)[0]
	idx2 = list(set(idx).difference(idx1))
	lnx[idx1] = np.log(r[idx1] + x[idx1])
	lnx[idx2] = -np.log(r[idx2] - x[idx2])

	lny = np.zeros(len(r))
	idx1 = np.where(np.abs(r + y) > np.finfo(float).eps)[0]
	idx2 = list(set(idx).difference(idx1))
	lny[idx1] = np.log(r[idx1] + y[idx1])
	lny[idx2] = -np.log(r[idx2] - y[idx2])

	lnz = np.zeros(len(r))
	idx1 = np.where(np.abs(r + z) > np.finfo(float).eps)[0]
	idx2 = list(set(idx).difference(idx1))
	lnz[idx1] = np.log(r[idx1] + z[idx1])
	lnz[idx2] = -np.log(r[idx2] - z[idx2])

	hx = (
		- jx * np.arctan2 (y * z, x * r)
		+ jy * lnz
		+ jz * lny
	)

	hy = (
		  jx * lnz
		- jy * np.arctan2 (x * z, y * r)
		+ jz * lnx
	)

	hz = (
		  jx * lny
		+ jy * lnx
		- jz * np.arctan2 (x * y, z * r)
	)

	h = np.c_[hx, hy, hz]
	return 100. * h


def prism(mgz, xobs, yobs, zobs, xsrc, ysrc, zsrc, dim):
	'''
	単位磁化したブロックによる磁気異常の磁場3成分(不定積分)を計算
	mgz: 磁化ベクトル(jx, jy, jz)
	xobs, yobs, zobs: 観測点座標
	xsrc, ysrc, zsrc: ソース位置座標
	dim: ブロックのサイズ
		dim[0] = [dx1, dx2]:
				dx1:ブロック中心からx軸方向の左端までの長さ(>0)
				dx2:ブロック中心からx軸方向の右端までの長さ(>0)
		dim[1] = [dy1, dy2]:
				dy1: ブロック中心からy軸方向の下端までの長さ(>0)
				dy2: ブロック中心からy軸方向の上端までの長さ(>0)
		dim[2] = [dz1, dz2]:
				dz1: ブロック中心からz軸方向の下端までの長さ(>0)
				dz2: ブロック中心からz軸方向の上端までの長さ(>0)
	'''
	dx = dim[0]
	dy = dim[1]
	dz = dim[2]

	xs = [xsrc - dx[0], xsrc + dx[1]]
	ys = [ysrc - dy[0], ysrc + dy[1]]
	zs = [zsrc - dz[0], zsrc + dz[1]]

	h = np.zeros([len(xobs), 3])
	flag = 1.0
	for i in range(2):
		flag = -flag
		x = xs[i] - xobs
		for j in range(2):
			flag = -flag
			y = ys[j] - yobs
			for k in range(2):
				flag = -flag
				z = zs[k] - zobs

				dh = _prism_kernel_func_(mgz, x, y, z)
				h += flag * dh

	return h


def prism2(mgz, xobs, yobs, zobs, xsrc, ysrc, zsrc, dim):
	'''
	単位磁化したブロックによる磁気異常の磁場3成分(不定積分)を計算
	mgz: 磁化ベクトル(jx, jy, jz)
	xobs, yobs, zobs: 観測点座標
	xsrc, ysrc, zsrc: ソース位置座標
	dim: ブロックのサイズ
		dim[0] = [dx1, dx2]:
				dx1:ブロック中心からx軸方向の左端までの長さ(>0)
				dx2:ブロック中心からx軸方向の右端までの長さ(>0)
		dim[1] = [dy1, dy2]:
				dy1: ブロック中心からy軸方向の下端までの長さ(>0)
				dy2: ブロック中心からy軸方向の上端までの長さ(>0)
		dim[2] = [dz1, dz2]:
				dz1: ブロック中心からz軸方向の下端までの長さ(>0)
				dz2: ブロック中心からz軸方向の上端までの長さ(>0)
	'''
	
	dx = dim[0]
	dy = dim[1]
	dz = dim[2]

	xs = [xsrc - dx[0], xsrc + dx[1]]
	ys = [ysrc - dy[0], ysrc + dy[1]]
	zs = [zsrc - dz[0], zsrc + dz[1]]

	h = np.zeros([len(xsrc), 3])
	flag = 1.0
	for i in range(2):
		flag = -flag
		x = xs[i] - xobs
		for j in range(2):
			flag = -flag
			y = ys[j] - yobs
			for k in range(2):
				flag = -flag
				z = zs[k] - zobs

				dh = _prism_kernel_func_(mgz, x, y, z)
				h += flag * dh

	return h


def total_force(exf, h):
	'''
	3成分磁場から全磁力を計算
	exf: 外部磁場方向の単位ベクトル
	h: 磁場3成分
	'''
	return np.dot(h, exf)