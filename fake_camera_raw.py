import sys,os,time
import numpy as np
from scipy import sparse
from scipy.io import readsav
from scipy.interpolate import interp1d
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

def _make_setup():

#----- Sample input
    R0s      = [+1.396,+1.485];   #R of radiation point R1,R2
    Z0s      = [-1.084,-1.249];   #Z of radiation point Z1,Z2
    A0s      = [+1.000,+1.000];   #Amplitude
    M0s      = [+0.015,+0.015];   #Margins for
    do_plot  = True               #Show image plot
    save_name= 'synthetic_outs_2pnt.pl'
#-------

    #Run info
    Rinfo = {} 
    
    #Add radiation geometry info
    Rinfo['nsample'] = 0
    for key in ['R0s','Z0s','A0s','M0s']:
        Rinfo[key] = [];
    
#--- Append R-info here to do the scan 
    Rinfo['R0s'].append(R0s)
    Rinfo['Z0s'].append(Z0s)
    Rinfo['A0s'].append(A0s)
    Rinfo['M0s'].append(M0s)
    Rinfo['nsample'] += 1

    #Do plot and out(save) file name
    Rinfo['doplot']  = do_plot 
    Rinfo['outfile'] = save_name    

    return Rinfo

def _draw(cam_image=[],cam_inver=[],camgeo={}):

    #Draw synthetic/inverted images

    #cam_image: Synthetic image
    #cam_inver: Inverted image
    #cam_geo:   Camera geometry

    fig = plt.figure(1)
    
    #Draw inverted image
    plt.subplot(1,3,1)
    plt.title('Inverted')
    plt.pcolormesh(camgeo['inv_x'],camgeo['inv_y'],cam_inver)       
    plt.xlabel('R[m]')
    plt.ylabel('Z[m]')

    #Draw synthetic image
    plt.subplot(1,3,2)
    plt.title('Synthetic raw')
    plt.pcolormesh(cam_image)
    plt.xlabel('X[#]')
    plt.ylabel('Y[#]')

    #Draw synthetic image with wall picture
    xx = []; yy = [];
    plt.subplot(1,3,3)
    plt.title('Overlay')
    for ih in range(camgeo['cam_x'].shape[0]):
        for iw in range(camgeo['cam_x'].shape[1]):
            if cam_image[ih,iw]>0.01:
                xx.append(iw); yy.append(ih)
    plt.pcolormesh(camgeo['tar_r'])
    plt.scatter(xx,yy,marker='x',color='r',s=0.1)
    plt.xlabel('X[#]')
    plt.ylabel('Y[#]')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.3, 
                        hspace=0.1)
    plt.show()

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
    cam_inver = np.zeros((camgeo['inv_y'].shape[0],camgeo['inv_x'].shape[0]))

    if not (len(R0s)==len(Z0s)==len(A0s)==len(M0s)):
        print('>>> Given emission info is wrong!')
        exit()

    print('>>> Case #',Rinfo['nsample'])

    for i,R0 in tqdm(enumerate(R0s)):
        image, inver = _generate_image(R0s[i],Z0s[i],A0s[i],M0s[i],cam_image,cam_inver,camgeo)

        cam_image += image
        cam_inver += inver

    return cam_image, cam_inver

def _generate_image(R0=0.,Z0=0.,A0=0.,M0=0.,cam_image=[],cam_inver=[],camgeo={}):

    # Make images by emission ring at (R0,Z0)[m] with A0 amplitude, M0 [m] thickness
    # with camgeo info

    for iw in range(camgeo['cam_x'].shape[1]):
        for ih in range(camgeo['cam_x'].shape[0]):

            # Skip 0.0.0 pixels
            if (camgeo['tar_x'][ih,iw]==0.): continue

            # Location of emission ring along the line of sight (LOS)
            tt = (Z0-camgeo['cam_z'][ih,iw])/camgeo['vec_z'][ih,iw]
            dt =  M0/camgeo['vec_z'][ih,iw]
            # Skip if not on the LOS
            if (tt<0 or tt>1): continue       

            # Find the intersection of line of sight and emission ring
            tot_emission = 0.
            tot_emission += _get_emission(camgeo,M0,R0,Z0,ih,iw,tt+0.5*dt)
            tot_emission += _get_emission(camgeo,M0,R0,Z0,ih,iw,tt)
            tot_emission += _get_emission(camgeo,M0,R0,Z0,ih,iw,tt-0.5*dt)

            ssl  = camgeo['vec_s'][ih,iw] * abs(dt)

            tot_emission = A0 * tot_emission * ssl / 3

            # Accumulate emission on the pixel
            cam_image[ih,iw] += tot_emission

    # Generate inverted image
    for iw in range(camgeo['inv_x'].shape[0]):
        xx= camgeo['inv_x'][iw]
        for ih in range(camgeo['inv_y'].shape[0]):
            yy= camgeo['inv_y'][ih]
            dd = np.sqrt((Z0-yy)**2+(R0-xx)**2)
            cam_inver[ih,iw] += A0 * np.exp(-(dd/M0)**3)

    return cam_image, cam_inver

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

def _main():

    # Main runs

    Rinfo  = _make_setup()
    camgeo = _load_camera(camera_save='Camera_geo.pl',
                            filename1='geom_240perp_unwarp_2022fwd.sav',
                            filename2='cam240perp_geometry_2022.sav')

    # Output of rnd
    output = {};
    # Number of synthetic images
    output['run_setup'] = Rinfo
    # Synthetic image  dimension
    output['image_size']= camgeo['tar_x'].shape
    # Inverged image  dimension
    output['inver_size']= camgeo['inv_x'].shape    
    output['inver_R']   = camgeo['inv_x']
    output['inver_Z']   = camgeo['inv_y']

    output['image']     = {}
    output['inver']     = {}

    for rind in range(Rinfo['nsample']):
        output['image'][rind],output['inver'][rind] = _integrate_image(Rinfo,rind,camgeo)
        if Rinfo['doplot']: _draw(output['image'][rind],output['inver'][rind],camgeo)

    with open(Rinfo['outfile'],'wb') as f: pickle.dump(output,f)

if __name__ == "__main__":
    _main()




