import os
import numpy as np
from scipy.io import readsav
from scipy.interpolate import interp1d
import pickle
import h5py

def _make_samples():
    
    # Make R0s, Z0s, A0s, M0s
    nsp = 10

    R0s = np.zeros((nsp, 2))
    Z0s = np.zeros((nsp, 2))
    # Make R0s, Z0s, A0s, M0s
    R0s_S = np.repeat(np.repeat(np.repeat(np.linspace(1.4, 1.7, nsp), nsp), nsp), nsp)
    Z0s_S = np.tile(np.repeat(np.repeat(np.linspace(-1.3, -1.2, nsp), nsp), nsp), nsp)

    R0s_X = np.repeat(np.tile(np.tile(np.linspace(1.3, 1.6, nsp), nsp), nsp), nsp)
    Z0s_X = np.tile(np.tile(np.tile(np.linspace(-1.2, -0.9, nsp), nsp), nsp), nsp)

    R0s = np.array([R0s_S, R0s_X]).T
    Z0s = np.array([Z0s_S, Z0s_X]).T
    
    nsample = Z0s.shape[0]
    
    A0s = np.ones((nsample, 2))
    M0s = np.ones((nsample, 2)) * 0.015
    
    return R0s, Z0s, A0s, M0s, nsample

def _make_setup():

    save_name= 'synthetic_outs_v3.h5'
    chunk_size = 100
    
    Rinfo = {}
    Rinfo['outfile'], Rinfo['chunk_size'] = save_name, chunk_size
    Rinfo['R0s'], Rinfo['Z0s'], Rinfo['A0s'], Rinfo['M0s'], Rinfo['nsample'] = _make_samples()

    if Rinfo['nsample'] < Rinfo['chunk_size']:
        Rinfo['chunk_size'] = Rinfo['nsample']
    
    return Rinfo

def _load_camera(camera_save='Camera_geo.pl',filename1='geom_240perp_unwarp_2022fwd.sav',filename2='cam240perp_geometry_2022.sav'):

    #Load camera geometry

    #camera_save: Post-process camera info
    #filename1:   Camera target RZPhi info
    #filename2:   Camera CCD vertex info    

    camgeo={}    #geometry variables

    if not os.path.isfile(camera_save):

        target = readsav(filename1)
        camgeo['tar_r'] = target['newR'] / 1.e2
        camgeo['tar_z'] = target['newZ'] / 1.e2
        camgeo['tar_p'] = target['newPhi'] / 180*np.pi    

        vertex = readsav(filename2)
        location = vertex.Geom.povray[0][0][0] / 1.e2

        with open(camera_save,'wb') as f: 
            pickle.dump([camgeo['tar_r'],camgeo['tar_z'],camgeo['tar_p'],location],f)
    else:
        with open(camera_save,'rb') as f: 
            [camgeo['tar_r'],camgeo['tar_z'],camgeo['tar_p'],location] = pickle.load(f)

    camgeo['tar_x'] = camgeo['tar_r']  * np.cos(camgeo['tar_p'])
    camgeo['tar_y'] = camgeo['tar_r']  * np.sin(camgeo['tar_p'])

    [camgeo['nh'], camgeo['nw']] = camgeo['tar_x'].shape
    
    pre_ih = []
    new_ih = []
    for ih in range(camgeo['tar_x'].shape[0]):
        pre_ih.append(ih)
        new_ih.append(_calibrating_indexes(ih,camgeo))

    for iw in range(camgeo['tar_x'].shape[1]):

        camgeo['tar_x'][:,iw] = interp1d(pre_ih,camgeo['tar_x'][:,iw])(new_ih)
        camgeo['tar_y'][:,iw] = interp1d(pre_ih,camgeo['tar_y'][:,iw])(new_ih)
        camgeo['tar_z'][:,iw] = interp1d(pre_ih,camgeo['tar_z'][:,iw])(new_ih)
        camgeo['tar_r'][:,iw] = interp1d(pre_ih,camgeo['tar_r'][:,iw])(new_ih)

    camgeo['cam_x']  = np.ones((camgeo['nh'],camgeo['nw'])) * location[0]
    camgeo['cam_y']  = np.ones((camgeo['nh'],camgeo['nw'])) * location[1]
    camgeo['cam_z']  = np.ones((camgeo['nh'],camgeo['nw'])) * location[2]

    camgeo['vec_x'] = camgeo['tar_x']-camgeo['cam_x']
    camgeo['vec_y'] = camgeo['tar_y']-camgeo['cam_y']
    camgeo['vec_z'] = camgeo['tar_z']-camgeo['cam_z']
    camgeo['vec_s'] = np.sqrt(camgeo['vec_x']**2+camgeo['vec_y']**2+camgeo['vec_z']**2)

    camgeo['cam_c'] = np.zeros((camgeo['nh'],camgeo['nw']))

    ih2 = int(camgeo['nh']/2)
    iw2 = int(camgeo['nw']/2)

    for ih in range(camgeo['nh']):
        for iw in range(camgeo['nw']):
            sum0 = 0; sum1 = 0; sum2 = 0;
            for d in ['vec_x','vec_y','vec_z']:
                sum0+= camgeo[d][ih,iw]  *camgeo[d][ih,iw]
                sum1+= camgeo[d][ih2,iw2]*camgeo[d][ih2,iw2]
                sum2+= camgeo[d][ih,iw]  *camgeo[d][ih2,iw2]

            camgeo['cam_c'][ih,iw] = abs(sum2)/np.sqrt(sum0)/np.sqrt(sum1)

    camgeo['inv_x'] = np.linspace(+1.0,+2.0,201)
    camgeo['inv_y'] = np.linspace(-1.4,-0.4,201)
    
    print('>>> Synthetic Camera dim.',camgeo['tar_r'].shape)
    print('>>> Inverted  Camera dim. (%i, %i)'%(camgeo['inv_y'].shape[0],camgeo['inv_x'].shape[0]))
    return camgeo

def _integrate_image(Rinfo={},info_ind=0,camgeo={}):

    # Integrate images of different radiating rings in info_ind-th Rinfo

    R0s   = Rinfo['R0s'][info_ind]
    Z0s   = Rinfo['Z0s'][info_ind]
    A0s   = Rinfo['A0s'][info_ind]
    M0s   = Rinfo['M0s'][info_ind]

    cam_image = np.zeros(camgeo['cam_x'].shape)

    for i in range(len((R0s))):
        image = _generate_image(R0s[i],Z0s[i],A0s[i],M0s[i],cam_image,camgeo)
        cam_image += image
        
    return cam_image

def _generate_image(R0=0., Z0=0., A0=0., M0=0., cam_image=[], camgeo={}):
    
    # Make images by emission ring at (R0,Z0)[m] with A0 amplitude, M0 [m] thickness with camgeo info
    ih, iw = np.where(camgeo['tar_x'] != 0)
    tt = (Z0 - camgeo['cam_z'][ih, iw]) / camgeo['vec_z'][ih, iw]
    dt = M0 / camgeo['vec_z'][ih, iw]
    mask = (tt >= 0) & (tt <= 1)
    tot_emission = np.zeros_like(ih, dtype=float)
    tot_emission[mask] += _get_emission(camgeo, M0, R0, Z0, ih[mask], iw[mask], tt[mask] + 0.5 * dt[mask])
    tot_emission[mask] += _get_emission(camgeo, M0, R0, Z0, ih[mask], iw[mask], tt[mask])
    tot_emission[mask] += _get_emission(camgeo, M0, R0, Z0, ih[mask], iw[mask], tt[mask] - 0.5 * dt[mask])
    
    ssl = camgeo['vec_s'][ih, iw] * np.abs(dt)
    
    tot_emission = A0 * tot_emission * ssl / 3
    
    cam_image[ih, iw] += tot_emission
    
    return cam_image

def _get_emission(camgeo={},M0=0.,R0=0.,Z0=0.,ih=0,iw=0,tt=0.):

    # Make emission from radiating point at tt of LOS to [ih,iw] pixel of camgeo

    xx = camgeo['vec_x'][ih,iw] * tt + camgeo['cam_x'][ih,iw]
    yy = camgeo['vec_y'][ih,iw] * tt + camgeo['cam_y'][ih,iw]
    zz = camgeo['vec_z'][ih,iw] * tt + camgeo['cam_z'][ih,iw]

    rr = np.sqrt(xx**2+yy**2)
    dd = np.sqrt((Z0-zz)**2+(R0-rr)**2)

    return np.exp(-(dd/M0)**3)

def _calibrating_indexes(ih=0,camgeo={}):

    # Re-adjustment of vertical pixel index to match the real-image with synthetic image
    coefs= [-5.94*1.e-6,
           +1.87*1.e-3,
           -2.49*1.e-1,
           +2.70*1.e+1]
    y   = 0.
    for k in range(4): y += coefs[k] * (ih ** (3-k))
    y = max(y,15)
    y+= ih
    y = min(y,camgeo['nh']-1)
    return y
    
def _main(): # with hdf5 output

    # Main runs
    Rinfo  = _make_setup()
    camgeo = _load_camera(camera_save='Camera_geo.pl',
                            filename1='geom_240perp_unwarp_2022fwd.sav',
                            filename2='cam240perp_geometry_2022.sav')

    # Output of rnd
    output = {};
    output['image_size']= camgeo['tar_x'].shape
    
    # Inversed image  dimension
    output['inver_size']= camgeo['inv_x'].shape    
    output['inver_R']   = np.asarray(camgeo['inv_x'])
    output['inver_Z']   = np.asarray(camgeo['inv_y'])
    
    for i in range(Rinfo['nsample']):
        R0s, Z0s, A0s, M0s = Rinfo['R0s'][i], Rinfo['Z0s'][i], Rinfo['A0s'][i], Rinfo['M0s'][i]
        if not (len(R0s)==len(Z0s)==len(A0s)==len(M0s)):
            raise ValueError('>>> Given emission info is wrong!')
    
    print(f'>>> Completed checks. Generating {Rinfo["nsample"]} images with chunks of size {Rinfo["chunk_size"]}.')
        
    with h5py.File(Rinfo['outfile'], 'w') as hf:
        images = hf.create_dataset('image', (Rinfo['nsample'], output['image_size'][0], output['image_size'][1]))
        for i in range(0, Rinfo['nsample'], Rinfo['chunk_size']):
                end = min(i + Rinfo['chunk_size'], Rinfo['nsample'])
                print(f'>>> Case {i} to {end}')
                image = [_integrate_image(Rinfo, rind, camgeo) for rind in range(i, end)]
                images[i:end] = image

            # Save other output to hdf5 file
        for key, value in output.items():
            hf.create_dataset(key, data=value)

if __name__ == "__main__":
    _main()