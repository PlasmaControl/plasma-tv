import numpy as np
import pickle
import glob
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal
import pandas as pd
import matplotlib.patches as mpatches
import scipy.sparse
import scipy.signal
# from scipy.sparse.linalg.eigen.arpack import eigs as Eigens
from scipy.sparse.linalg import eigs as Eigens
# import seaborn as sns
import random
import h5py
import scipy.interpolate as interp
import cv2

# from plotly import figure_factory as ff

##############################################################################

def init_weights(input_dim, res_size, K_in, K_rec, insca, spra, bisca):
    # ---------- Initializing W_in ---------
    winexist, fname = find_saved_weights('Win', input_dim, res_size, K_in, K_rec)
    if winexist:
        with open(fname, 'rb') as f:
            W_in = pickle.load(f)
            W_in *= insca
    else:
        if K_in == -1 or input_dim < K_in:
            W_in = insca * (np.random.rand(res_size, input_dim) * 2 - 1)
        else:
            Ico = 0
            nrentries = np.int32(res_size * K_in)
            ij = np.zeros((2, nrentries))
            datavec = insca * (np.random.rand(nrentries) * 2 - 1)
            for en in range(res_size):
                Per = np.random.permutation(input_dim)[:K_in]
                ij[0][Ico:Ico + K_in] = en
                ij[1][Ico:Ico + K_in] = Per
                Ico += K_in
            W_in = scipy.sparse.csc_matrix((datavec, np.int32(ij)), shape=(res_size, input_dim), dtype='float32')
            if K_in > input_dim / 2:
                W_in = W_in.todense()

    # ---------- Initializing W_res ---------
    wrecexist, fname = find_saved_weights('Wres', input_dim, res_size, K_in, K_rec)
    if wrecexist:
        with open(fname, 'rb') as f:
            W_res = pickle.load(f)
            W_res *= spra
    else:
        converged = False
        attempts = 50
        while not converged and attempts > 0:
            if K_rec == -1:
                W_res = np.random.randn(res_size, res_size)
            else:
                Ico = 0
                nrentries = np.int32(res_size * K_rec)
                ij = np.zeros((2, nrentries))
                datavec = np.random.randn(nrentries)
                for en in range(res_size):
                    Per = np.random.permutation(res_size)[:K_rec]
                    ij[0][Ico:Ico + K_rec] = en
                    ij[1][Ico:Ico + K_rec] = Per
                    Ico += K_rec
                W_res = scipy.sparse.csc_matrix((datavec, np.int32(ij)), shape=(res_size, res_size), dtype='float32')
                if K_rec > res_size / 2:
                    W_res = W_res.todense()
            try:
                we = Eigens(W_res, return_eigenvectors=False, k=6)
                converged = True
            except:
                print("WARNING: No convergence! Redo %i times ... " % (attempts - 1))
                attempts -= 1
                pass

        W_res *= (spra / np.amax(np.absolute(we)))
    # ---------- Initializing W_bi ---------
    wbiexist, fname = find_saved_weights('Wbi', input_dim, res_size, K_in, K_rec)
    if wbiexist:
        with open(fname, 'rb') as f:
            W_bi = pickle.load(f)
            W_bi *= bisca
    else:
        W_bi = bisca * (np.random.rand(res_size) * 2 - 1)
    print("found (W_in, W_rec, W_bi) > (%s, %s, %s)" % (winexist, wrecexist, wbiexist))
    return W_in, W_res, W_bi


##############################################################################
def find_saved_weights(wname, input_dim, res_size, K_in, K_rec):
    s_r = []
    saved_reservoirs = glob.glob('/scratch/gpfs/aj17/saved_files/saved_' + wname + '*')
    for curr_file in (saved_reservoirs):
        s_r.append(curr_file[curr_file.rfind('_') + 1:curr_file.rfind('.')])
    if wname == 'Win':
        required_file_id = 'I' + str(input_dim) + 'R' + str(res_size) + 'Kin' + str(K_in)
    elif wname == 'Wres':
        required_file_id = 'R' + str(res_size) + 'Krec' + str(K_rec)
    elif wname == 'Wbi':
        required_file_id = 'R' + str(res_size)
    return (required_file_id in s_r), '/scratch/gpfs/aj17/saved_files/saved_' + wname + '_' + required_file_id + '.pkl'


##############################################################################

def res_exe(W_in, W_res, W_bi, leak, U):
    T = U.shape[0]  # size of the input vector
    nres = W_res.shape[0]  # Getting the size of the network (= 100)
    R = np.zeros((T + 1, nres),
                 dtype='float32')  # Initializing the RCN output matrix (one extra frame for the warming up)
    for t in range(T):  # for each frame
        if scipy.sparse.issparse(W_in):
            a = W_in * U[t, :]
        else:
            a = np.dot(W_in, U[t, :])

        if scipy.sparse.issparse(W_res):
            b = W_res * R[t, :]
        else:
            b = np.dot(W_res, R[t, :])
        R[t + 1, :] = np.tanh(a + b + W_bi)
        R[t + 1, :] = (1 - leak) * R[t, :] + leak * R[t + 1, :]
    R = np.concatenate((np.ones((R.shape[0], 1)), R), 1)
    return R[1:, :]  # returns the reservoir output and the desired output


##############################################################################

def res_train(xTx, xTy, xlen, regu):
    t1 = time.time()
    lmda = regu ** 2 * xlen
    inv_xTx = np.linalg.inv(xTx + lmda * np.eye(xTx.shape[0],dtype=np.float32))
    beta = np.dot(inv_xTx, xTy)  # beta is the output weight matrix
    # print ('RCN trained\tin '+str(round(time.time()-t1,2))+' sec.!')
    return beta
