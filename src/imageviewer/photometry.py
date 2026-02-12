"""
Photometric analysis functions:
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows

import pandas as pd

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.visualization import simple_norm
from astropy.visualization.wcsaxes import add_scalebar
from astroquery.sdss import SDSS
from astroquery.vizier import Vizier

import photutils.psf as psf
from photutils.detection import IRAFStarFinder, find_peaks

from scipy.spatial import cKDTree


def photo_analysis(filename,
                   xy_all = None):
    n_fig = 0
    phot_tables = []
    sky = {'mean':[], 'std':[]}
    #e_dict = {'error':[], 'B':[], 'n_im':[], 'avg_t':[],'sum1':[], 'sum2':[]}
    #SNR_pix = []

    with fits.open(filename) as hdul:
        hdu = hdul[0]
        header = hdu.header #type: ignore
        data = hdu.data #type: ignore
        wcs = WCS(header)

    # Background estimation and substraction
    sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    bkg = sky_mean
    data_sub = data - bkg
    sky['mean'].append(sky_mean)
    sky['std'].append(sky_std)
    print('Photometric analysis is not yet implemented.')
    return None

def detect_sources(filename, 
                   method = 'find_peaks',
                   sky_sigma = 3.0,
                   maxiters = 5,
                   sky_threshold = 3.0,
                   fwhm = 3.0,
                   init_table = None,
                   add_sources = False,
                   plot = True):
    
    with fits.open(filename) as hdul:
        hdu = hdul[0]
        header = hdu.header #type: ignore
        data = hdu.data #type: ignore

    # Background estimation and substraction
    print('Removing sky background...')
    sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=sky_sigma, maxiters=maxiters)
    bkg = sky_mean
    data_sub = data - bkg
    print('- Sky background mean: %s; Sky background std: %s'%(sky_mean, sky_std))
        # sky_mean, sky_median, sky_std = sigma_clipped_stats(data_sub, sigma=sky_sigma, maxiters=maxiters)
        # print('- Sky background mean: %s; Sky background std: %s'%(sky_mean, sky_std))
    if init_table is None:
        print('No initial table provided. Will perform source detection on the image.')
        
        if 'FWHM'  in header:
            fwhm = header['FWHM']
        elif 'seeing' in header:
            fwhm = header['seeing'] / header['SCALE']
        else:
            print("FWHM not found in header. Using input value fwhm = %s."%fwhm)

        print('FWHM used for source detection: %s pixels'%fwhm)

        # Source detection
        if method == 'IRAF':
            print('No initial table provided. Using IRAFStarFinder for source detection.')
            threshold_iraf =  (sky_std * sky_threshold)
            print('- IRAFStarFinder threshold: %s'%threshold_iraf)
            iraf_finder = IRAFStarFinder(threshold=threshold_iraf, fwhm=fwhm)
            print('Detecting sources...')
            iraf_stars = iraf_finder(data_sub)
            print('- Stars found by IRAFStarFinder: %d'%len(iraf_stars))
            # sorting list of found stars
            iraf_stars.sort('flux', reverse = True)
            iraf_stars['flux_id'] = 0
            iraf_stars['r_pix'] = 0
            for i in range(len(iraf_stars)): 
                iraf_stars[i]['flux_id'] = i
                iraf_stars[i]['r_pix'] = np.sqrt(iraf_stars[i]['npix'])
            init_table = iraf_stars
            # init_table['x'] = init_table['xcentroid']
            # init_table['y'] = init_table['ycentroid']
            

        elif method == 'find_peaks':
            print('No initial table provided. Using find_peaks for source detection.')
            threshold_fp = (sky_std * sky_threshold)
            print('- find_peaks threshold: %s'%threshold_fp)
            print('Detecting sources...')
            fp_sources = find_peaks(data_sub, threshold=threshold_fp, 
                                    box_size=int(fwhm),
                                    wcs = WCS(header))
            print('- Sources found by find_peaks: %d'%len(fp_sources))

            init_table = fp_sources
        
    if method =='IRAF': xlab, ylab = 'x_centroid', 'y_centroid'
    elif method == 'find_peaks': xlab, ylab = 'x_peak', 'y_peak'
    else: xlab, ylab = 'x', 'y'

    if plot:
        fig,ax = plt.subplots()
        ax.imshow(data_sub, 
            cmap = 'gray', origin = 'lower',
            vmin =  (sky_std * sky_threshold),
            vmax =  sky_std * sky_threshold,
            )
        ax.scatter(init_table[xlab], init_table[ylab], 
                   marker='x', color='red', s=20)
        ax.set_title('Detected sources')
        # plt.show()

        if add_sources:
            # Collecting clicked coordinates
            print('Click on the image to add sources. Close the image when done.')
            coords = []
            def onclick(event):
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    return
                x, y = event.xdata, event.ydata
                coords.append((x, y))
                ax.plot(x, y, 'bx', markersize=10)
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', onclick)

            ax.set_title('Click to add sources')
            plt.show(block=False)
            coords = plt.ginput(n=-1, timeout=0)
            print(f'Added {len(coords)} sources.')
            plt.close(fig)

    if add_sources:
        return init_table, coords
    else:
        return init_table