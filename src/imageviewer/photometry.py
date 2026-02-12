"""
Photometric analysis functions:
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows

import pandas as pd
from typing import cast

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
from astroquery.simbad import Simbad

import photutils.psf as psf
from photutils.detection import IRAFStarFinder, find_peaks

from scipy.spatial import cKDTree


def photo_analysis(filename,
                   init_table = None,
                   sky_background = {'sigma': 3.0, 'maxiters': 5, 'sky_threshold': 3.0},
                   phtometry_params = {'psf_fwhm_shape': 3.0, 'aperture_fwhm': 3.0, 'fitter_maxiters': 100,
                                       'qfit_filter': 8, 'cfit_filter': 0.05},
                   catalogue = 'PanSTARRS',
                   plot = True,
                   stacked = False,
                   ):
    n_fig = 0
    phot_tables = []
    sky = {'mean':[], 'std':[]}
    #e_dict = {'error':[], 'B':[], 'n_im':[], 'avg_t':[],'sum1':[], 'sum2':[]}
    #SNR_pix = []

    if init_table is None:
        print('No table of stars "init_table" provided. Use detect_sources function to detect sources and perform photometric analysis.\n')
        print('Save the resulting table to pkl format by:')
        print(' - table.to_pandas().to_pickle("./table.pkl")')
        return 
    
    with fits.open(filename) as hdul:
        hdu = hdul[0]
        header = hdu.header #type: ignore
        data = hdu.data #type: ignore
        wcs = WCS(header)

    fil = header['FILTER'] if 'FILTER' in header else 'Unknown filter'

    # Background estimation and substraction
    print('Removing sky background...')
    sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=sky_background['sigma'], maxiters=sky_background['maxiters'])
    bkg = sky_mean
    data_sub = data - bkg
    print('- Sky background mean: %s; Sky background std: %s'%(sky_mean, sky_std))

    sky['mean'].append(sky_mean)
    sky['std'].append(sky_std)

    # Known Star coordinates
    if type(init_table) == str:
            if init_table.endswith('.pkl'):
                init_table = pd.read_pickle(init_table)
            elif init_table.endswith('.csv'):
                init_table = pd.read_csv(init_table)
            else:
                print('Unsupported file format for initial table. Please provide a .pkl or .csv file.')
                return None
    if type(init_table) == pd.DataFrame:
        init_table = Table.from_pandas(init_table)
    colnames = init_table.colnames
    if 'x' not in colnames or 'y' not in colnames:
        print('Initial table should have columns "x" and "y". Please check the format of the initial table.')
        return None
    
    print('Initial number of star coordinates looked at: %i'%len(init_table))

    # Defining sky area to search in SDSS catalogue
    center = [Angle(header['CRVAL1'], u.deg), Angle(header['CRVAL2'], u.deg)]
    sc_center = SkyCoord(header['CRVAL1']*u.deg, header['CRVAL2']*u.deg)
    # RA, DEC of image corners
    ny, nx = wcs.pixel_shape # type: ignore
    corners_pix = np.array([[0, 0], [nx-1, 0], [nx-1, ny-1], [0, ny-1]])
    corners_sky = wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
    # RA/Dec bounding box
    ra_range = [corners_sky.ra.min().value, corners_sky.ra.max().value] # type: ignore
    dec_range = [corners_sky.dec.min().value, corners_sky.dec.max().value] # type: ignore
    print('Sky area observed in RA, DEC (deg): ', ra_range, dec_range)
    
    if catalogue == 'SDSS':
        # Query SDSS with the sky area, obtain psfMag values for interest filters
        sdss_table = SDSS.query_region(sc_center,                               # type: ignore
                                    width = (ra_range[1]-ra_range[0])*u.deg,
                                    height = (dec_range[1]-dec_range[0])*u.deg,
                                    fields=['ra','dec', 'psfMag_g', 'psfMag_r', 'psfMag_i']
                                    )
        print('Found %i stars catalogued in SDSS in the field'%len(sdss_table))
        # Transforming obtained sky coordinates of stars to pixels in the image
        cat_px = skycoord_to_pixel(SkyCoord(sdss_table['ra'], sdss_table['dec'], unit='deg'), wcs)
    
    elif catalogue == 'PanSTARRS':
        # Query Vizier for panstarrs1
        vizier = Vizier(columns = ['ra','dec', 'gmag', 'rmag', 'imag'])
        pstr = Vizier.query_region(sc_center,                           # type: ignore
                                    width = (ra_range[1]-ra_range[0])*u.deg,
                                    height = (dec_range[1]-dec_range[0])*u.deg,
                                    catalog = 'II/349/ps1'
                                    #fields=['ra','dec', 'gmag', 'rmag', 'imag']
                                    )
        pstr_table = pstr['II/349/ps1']['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag']
        cat_px = skycoord_to_pixel(SkyCoord(pstr_table['RAJ2000'], pstr_table['DEJ2000'], unit='deg'), wcs)
        print('Found %i stars catalogued in Panstarr in the field'%len(pstr_table))

    elif catalogue == 'Simbad':
        # Query Simbad for stars in the field
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('ra(d)', 'dec(d)', 'g', 'r', 'i', 'z')
        simbad_table = custom_simbad.query_region(sc_center, radius=Angle(np.max([ra_range[1]-ra_range[0], dec_range[1]-dec_range[0]])/2, u.deg))
        print('Found %i stars catalogued in Simbad in the field'%len(simbad_table))
        cat_px = skycoord_to_pixel(SkyCoord(simbad_table['ra'], simbad_table['dec'], unit='deg'), wcs)
    else:
        print('No catalogue provided. Available catalogues: "SDSS" and "PanSTARRS".')
        return None
    
    if plot:
        n_fig += 1
        plt.close(n_fig)
        fig, ax = plt.subplots(num=n_fig)
        ax.remove()
        ax = fig.add_subplot(111, projection=wcs)
        ax.imshow(data_sub, 
                    cmap = 'gray', origin = 'lower',
                    vmin =  - (sky_std*sky_background['sky_threshold']),
                    vmax =  + (sky_std*sky_background['sky_threshold']),
                    )
        ax.scatter(cat_px[0], cat_px[1], facecolor = 'none', edgecolor = 'green', label = 'Catalogue stars')
        ax.scatter(init_table['x'], init_table['y'], facecolor = 'none', edgecolor = 'red', marker = 's', label = 'Stars looked') # type: ignore
        ax.legend()
        title_str = 'Catalogued stars and known stars\nover image in filter %s'%header['FILTER'] if 'FILTER' in header else 'Catalogued stars and known stars'
        title_str += '\nCatalogue: %s, stars found: %i'%(catalogue, len(cat_px[0]))
        title_str += '\nStars looked at: %i'%len(init_table)
        ax.set_title(title_str)
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        plt.tight_layout()
        plt.show()

    print('\n---------------------------------------------------')
    print('\nPhotometry of image %s'%(filename))
    
    if stacked == False:
        fwhm_pix = (header['FWHM']) #8.0 
        gain = header['GAIN'] if 'GAIN' in header else 1.0
        error_map = np.sqrt(sky_std**2 + abs(data_sub) / gain)
        print('Error map average value: %s'%np.nanmean(error_map))

    else:
        print('Photometric analysis of stacked images not implemented yet.')

    # Aperture photometry
    psf_gaussian = psf.CircularGaussianPRF(fwhm=fwhm_pix)
    psf_shape_int = int(phtometry_params['psf_fwhm_shape'] * fwhm_pix)
    # make sure psf shape is even
    if psf_shape_int%2==0: psf_shape_int+=1
    psf_shape = (psf_shape_int,) * 2
    print('FWHM: %.2f px'%fwhm_pix)
    print('PSF shape: ', psf_shape)

    # PSF photometry
    psfphot = psf.PSFPhotometry(
        psf_model=psf_gaussian,
        fit_shape=psf_shape,
        aperture_radius=phtometry_params['aperture_fwhm']*fwhm_pix,
        fitter_maxiters = phtometry_params['fitter_maxiters']
    )
    phot_all = cast(Table, psfphot(data_sub, error= error_map,
                       init_params = init_table))
    phot_all.sort('flux_fit', reverse = True) 
    phot_all['flux_id'] = 0 
    phot_all['flux_id'] = np.arange(len(phot_all), dtype=int)
    for i in range(len(phot_all)): 
        phot_all[i]['flux_id'] = i 
    print('Found stars by photometry: ',len(phot_all)) 

    # filtering out the stars with bad photometry
    phot_g_all = phot_all.copy()
    # Removing stars with bad flags (saturated, truncated, etc.)
    flags_gi = np.where([x in [2,4,5,6,8,12] for x in list(phot_all['flags'])])  # type: ignore
    phot_g_all.remove_rows(flags_gi)
    # Removing stars with very high flux error compared to the flux value
    phot_g_all.remove_rows(np.where(phot_g_all['flux_err']/phot_g_all['flux_fit']>1)) # type: ignore
    big_err_gi = np.where(phot_all['flux_err']/phot_all['flux_fit']>1) # type: ignore
    # Removing stars with very high x or y error
    phot_g_all.remove_rows(np.where((phot_g_all['x_err']>10) |(phot_g_all['y_err']>10))) # type: ignore
    xy_err_gi = np.where((phot_all['x_err']>10) |(phot_all['y_err']>10)) # type: ignore
    # Removing stars with very high qfit or cfit values
    qfit_filter, cfit_filter = phtometry_params['qfit_filter'], phtometry_params['cfit_filter']
    qfit_gi = np.where((phot_all['qfit']>qfit_filter) | (phot_all['qfit']<-qfit_filter))
    cfit_gi = np.where((phot_all['cfit']>cfit_filter) | (phot_all['cfit']<-cfit_filter))
    phot_g_all.remove_rows(np.where((phot_g_all['qfit']>qfit_filter) | (phot_g_all['qfit']<-qfit_filter)))
    phot_g_all.remove_rows(np.where((phot_g_all['cfit']>cfit_filter) | (phot_g_all['cfit']<-cfit_filter)))
    
    print('Good photometry stars: %i'%len(phot_g_all))


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
            detect_table = iraf_stars
            xlab, ylab = 'xcentroid', 'ycentroid'
            
        elif method == 'find_peaks':
            print('No initial table provided. Using find_peaks for source detection.')
            threshold_fp = (sky_std * sky_threshold)
            print('- find_peaks threshold: %s'%threshold_fp)
            print('Detecting sources...')
            fp_sources = find_peaks(data_sub, threshold=threshold_fp, 
                                    box_size=int(fwhm),
                                    wcs = WCS(header))
            print('- Sources found by find_peaks: %d'%len(fp_sources)) # type: ignore
            detect_table = fp_sources
            xlab, ylab = 'x_peak', 'y_peak'

        else:
            print('\nNo method for source detection provided.')
            print(' - Available methods: "IRAF" and "find_peaks".')
            detect_table = Table({'x': [], 'y': []}, dtype=[float, float])
        
    else: 
        if type(init_table) == str:
            if init_table.endswith('.pkl'):
                init_table = pd.read_pickle(init_table)
            elif init_table.endswith('.csv'):
                init_table = pd.read_csv(init_table)
            else:
                print('Unsupported file format for initial table. Please provide a .pkl or .csv file.')
                return None
        if type(init_table) == pd.DataFrame:
            colnames = init_table.columns.to_list()
        elif type(init_table) == Table:
            colnames = init_table.colnames
        if 'x' not in colnames or 'y' not in colnames:
            print('Initial table should have columns "x" and "y". Please check the format of the initial table.')
            return None
        print('Initial table provided. Skipping source detection on the image.')
        xlab, ylab = 'x', 'y'
        detect_table = init_table

    xy_stars = Table({'x': np.asarray(detect_table[xlab]), 'y': np.asarray(detect_table[ylab])}) # type: ignore

    if plot:
        fig,ax = plt.subplots()
        ax.imshow(data_sub, 
            cmap = 'gray', origin = 'lower',
            vmin =  -(sky_std * sky_threshold),
            vmax =  sky_std * sky_threshold,
            )
        if len(xy_stars) > 0:
            ax.scatter(xy_stars['x'], xy_stars['y'], # type: ignore
                    marker='x', color='red', s=20)
        ax.set_title('Detected sources: %d'%len(xy_stars))

        if add_sources:
            # Collecting clicked coordinates
            print('Click on the image to add sources. Close the image when done.')
            ax.set_title('Detected sources: %d\nClick to add sources'%len(xy_stars))

            def onclick(event):
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    return
                x, y = event.xdata, event.ydata
                xy_stars.add_row([x, y])
                ax.plot(x, y, 'bx', markersize=10)
                ax.set_title('Detected sources: %d\nClick to add sources'%len(xy_stars))
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', onclick)
    
    return xy_stars