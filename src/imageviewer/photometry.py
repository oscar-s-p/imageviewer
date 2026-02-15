"""
Photometric analysis functions:
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows
import ipywidgets as widgets
from IPython.display import display

from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_interp

import pandas as pd
from typing import cast

from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
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

from scipy.spatial import cKDTree # type: ignore


def photo_analysis(filename,
                   init_table = None,
                   sky_background = {'sigma': 3.0, 'maxiters': 5, 'sky_threshold': 3.0},
                   photometry_params = {'psf_fwhm_shape': 3.0, 'aperture_fwhm': 3.0, 'fitter_maxiters': 100,
                                       'qfit_filter': 8, 'cfit_filter': 0.05},
                   catalogue = 'PanSTARRS',
                   matching_params = {'mag_range': (13, 18), 'max_sep_pix': 5},
                   plot = True, n_fig_init = 0,
                   stacked = False,
                   print_info = True,
                   ):
    n_fig = n_fig_init
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
    
    print('\n-----------------------------------------------------------')
    print('Photometric analysis of image %s'%(filename))

    # Background estimation and substraction
    if print_info: print('Removing sky background...')
    sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=sky_background['sigma'], maxiters=sky_background['maxiters'])
    bkg = sky_mean
    data_sub = data - bkg
    if print_info: print('- Sky background mean: %s; Sky background std: %s'%(sky_mean, sky_std))
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
    
    if print_info: print('Initial number of star coordinates looked at: %i'%len(init_table))

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
    if print_info: print('Sky area observed in RA, DEC (deg): ', ra_range, dec_range)
    
    if catalogue == 'SDSS':
        # Query SDSS with the sky area, obtain psfMag values for interest filters
        cat_table = SDSS.query_region(sc_center,                               # type: ignore
                                    width = (ra_range[1]-ra_range[0])*u.deg,
                                    height = (dec_range[1]-dec_range[0])*u.deg,
                                    fields=['ra','dec', 'psfMag_g', 'psfMag_r', 'psfMag_i']
                                    )
        if print_info: print('Found %i stars catalogued in SDSS in the field'%len(cat_table))
        # Transforming obtained sky coordinates of stars to pixels in the image
        cat_px = skycoord_to_pixel(SkyCoord(cat_table['ra'], cat_table['dec'], unit='deg'), wcs)
        cat_labels = ['ra', 'dec', 'psfMag_'+fil[-1]]
    
    elif catalogue == 'PanSTARRS':
        # Query Vizier for panstarrs1
        vizier = Vizier(columns = ['ra','dec', 'gmag', 'rmag', 'imag'])
        pstr = Vizier.query_region(sc_center,                           # type: ignore
                                    width = (ra_range[1]-ra_range[0])*u.deg,
                                    height = (dec_range[1]-dec_range[0])*u.deg,
                                    catalog = 'II/349/ps1'
                                    #fields=['ra','dec', 'gmag', 'rmag', 'imag']
                                    )
        cat_table = pstr['II/349/ps1']['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag']
        cat_px = skycoord_to_pixel(SkyCoord(cat_table['RAJ2000'], cat_table['DEJ2000'], unit='deg'), wcs)
        if print_info: print('Found %i stars catalogued in Panstarr in the field'%len(cat_table))
        cat_labels = ['ra', 'dec', fil[-1]+'mag']

    elif catalogue == 'Simbad':
        # Query Simbad for stars in the field
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('ra(d)', 'dec(d)', 'g', 'r', 'i', 'z')
        cat_table = custom_simbad.query_region(sc_center, radius=Angle(np.max([ra_range[1]-ra_range[0], dec_range[1]-dec_range[0]])/2, u.deg))
        if print_info: print('Found %i stars catalogued in Simbad in the field'%len(cat_table))
        cat_px = skycoord_to_pixel(SkyCoord(cat_table['ra'], cat_table['dec'], unit='deg'), wcs)
        cat_labels = ['ra', 'dec', fil[-1]]
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

    if print_info: print('Performing aperture and PSF photometry on the image...')
    
    if stacked == False:
        fwhm_pix = (header['FWHM']) #8.0 
        gain = header['GAIN'] if 'GAIN' in header else 1.0
        error_map = np.sqrt(sky_std**2 + abs(data_sub) / gain)
        if print_info: print('Error map average value: %s'%np.nanmean(error_map))

    else:
        print('Photometric analysis of stacked images not implemented yet.')

    # Aperture photometry
    psf_gaussian = psf.CircularGaussianPRF(fwhm=fwhm_pix)
    psf_shape_int = int(photometry_params['psf_fwhm_shape'] * fwhm_pix)
    # make sure psf shape is even
    if psf_shape_int%2==0: psf_shape_int+=1
    psf_shape = (psf_shape_int,) * 2
    if print_info: print('FWHM: %.2f px'%fwhm_pix)
    if print_info: print('PSF shape: ', psf_shape)

    # PSF photometry
    psfphot = psf.PSFPhotometry(
        psf_model=psf_gaussian,
        fit_shape=psf_shape,
        aperture_radius=photometry_params['aperture_fwhm']*fwhm_pix,
        fitter_maxiters = photometry_params['fitter_maxiters']
    )
    phot_all = cast(Table, psfphot(data_sub, error= error_map,
                       init_params = init_table))
    phot_all.sort('flux_fit', reverse = True) 
    phot_all['flux_id'] = np.arange(len(phot_all), dtype=int)
    phot_all['peak_value'] = 0
    for i in range(len(phot_all)):
        x_peak, y_peak = int(phot_all['x_fit'][i]), int(phot_all['y_fit'][i])  # type: ignore
        try:
            phot_all[i]['peak_value'] = data_sub[y_peak-int(fwhm_pix):y_peak+int(fwhm_pix), x_peak-int(fwhm_pix):x_peak+int(fwhm_pix)].max() # type: ignore
        except:
            phot_all[i]['peak_value'] = 65000 # if the peak is too close to the edge, assign a high value to flag it later
    if print_info: print('Found stars by photometry: ',len(phot_all)) 

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
    qfit_filter, cfit_filter = photometry_params['qfit_filter'], photometry_params['cfit_filter']
    qfit_gi = np.where((phot_all['qfit']>qfit_filter) | (phot_all['qfit']<-qfit_filter))
    cfit_gi = np.where((phot_all['cfit']>cfit_filter) | (phot_all['cfit']<-cfit_filter))
    phot_g_all.remove_rows(np.where((phot_g_all['qfit']>qfit_filter) | (phot_g_all['qfit']<-qfit_filter)))
    phot_g_all.remove_rows(np.where((phot_g_all['cfit']>cfit_filter) | (phot_g_all['cfit']<-cfit_filter)))
    # Removing stars with peak value above saturation limit
    sat_gi = np.where(phot_all['peak_value'] > 61000)  # type: ignore
    phot_g_all.remove_rows(np.where(phot_g_all['peak_value'] > 61000))  # type: ignore
    phot_g_all['flux_id'] = np.arange(len(phot_g_all), dtype=int)

    if print_info: print('Good photometry stars: %i'%len(phot_g_all))

    # Instrumental magnitude and error
    phot_g_all['mag_inst'] = -2.5 * np.log10(phot_g_all['flux_fit']) # type: ignore
    phot_g_all['mag_inst_err'] = 1.0857 * (phot_g_all['flux_err'] / phot_g_all['flux_fit']) # type: ignore

    if plot:
        # Stars found after filtering
        n_fig += 1
        plt.close(n_fig)
        fig,ax = plt.subplots(num = n_fig)
        ax.remove()
        ax = fig.add_subplot(111, projection=wcs)
        ax.imshow(data_sub, 
                cmap = 'gray', origin = 'lower',
                vmin = - (sky_std*sky_background['sky_threshold']),
                vmax = (sky_std*sky_background['sky_threshold']),
                )
        ax.scatter(phot_g_all['x_fit'], phot_g_all['y_fit'], facecolor = 'none',
                edgecolor = 'red',label = 'good')
        ax.scatter(phot_all['x_fit'][qfit_gi], phot_all['y_fit'][qfit_gi], marker = 's',
                facecolor = 'none', edgecolor = 'blue', label = 'bad (qfit)')
        ax.scatter(phot_all['x_fit'][cfit_gi], phot_all['y_fit'][cfit_gi], marker = 'd',
                facecolor = 'none', edgecolor = 'green', label = 'bad (cfit)')
        ax.scatter(phot_all['x_fit'][flags_gi], phot_all['y_fit'][flags_gi], marker = '*',
                facecolor = 'none', edgecolor = 'orange', label = 'bad (flag)')
        ax.scatter(phot_all['x_fit'][big_err_gi], phot_all['y_fit'][big_err_gi], marker = 'p',
                facecolor = 'none', edgecolor = 'cyan', label = 'bad (flux error)')
        ax.scatter(phot_all['x_fit'][xy_err_gi], phot_all['y_fit'][xy_err_gi], marker = 'v',
                facecolor = 'none', edgecolor = 'magenta', label = 'bad (xy error)')
        ax.scatter(phot_all['x_fit'][sat_gi], phot_all['y_fit'][sat_gi], marker = 'P',
                facecolor = 'none', edgecolor = 'purple', label = 'bad (saturation)')
        
        for i in range(len(phot_g_all)):
            plt.text(phot_g_all['x_fit'][i]+10, phot_g_all['y_fit'][i], phot_g_all['flux_id'][i]) # type: ignore
        ax.legend(ncols = 4, bbox_to_anchor = (0.5,1.08,0,0), loc='center')

        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        # ax.set_title('Good photometry stars: %i'%len(phot_g_all))
        plt.show()

    # Comparison with catalogue stars
    if catalogue == 'SDSS':
        cat_mask = ~np.isnan(cat_table[cat_labels[2]])
    else:
        cat_mask = ~np.isnan(cat_table[cat_labels[2]]).mask
    if print_info: print('Comparing with %i catalogued stars in filter %s'%(np.sum(cat_mask), fil))
    
    # Cross-correlation of queried stars and found stars
    phot_xy = np.array([phot_g_all["x_fit"], phot_g_all["y_fit"]]).T
    cat_xy = np.array([cat_px[0][cat_mask], cat_px[1][cat_mask]]).T
    # KD-tree for fast nearest-neighbor search
    kdtree = cKDTree(cat_xy)
    dist, idx_cat = kdtree.query(phot_xy, k=1) 
    # Match if separation < threshold (e.g. 3 pixels)
    max_sep_pix = matching_params['max_sep_pix']
    good = dist < max_sep_pix
    # Build matched table
    calib = Table()
    calib["phot_idx"] = phot_g_all['flux_id'][good]
    calib["cat_idx"] = idx_cat[good]
    calib["dx_pix"] = phot_xy[good, 0] - cat_xy[idx_cat[good], 0]
    calib["dy_pix"] = phot_xy[good, 1] - cat_xy[idx_cat[good], 1]
    calib["sep_pix"] = dist[good]
    # Link magnitudes
    calib["mag_inst"] = phot_g_all["mag_inst"][calib["phot_idx"]]
    calib["mag_cat"] = cat_table[cat_labels[2]][cat_mask][calib["cat_idx"]]
    # Additional filters
    calib = calib[(calib["mag_cat"] > matching_params['mag_range'][0]) & 
                  (calib["mag_cat"] < matching_params['mag_range'][1])]
    
    if print_info: print('Stars matched with catalogue after filtering by magnitude and distance: %i'%len(calib))

    # Calculate ZP
    calib["ZP"] = calib["mag_cat"] - calib["mag_inst"]  # type: ignore
    # mean_zp, med_zp, std_zp = sigma_clipped_stats(calib['ZP'], sigma=1.0, maxiters=5)
    zp_mask = sigma_clip(calib['ZP'], sigma=1.0, maxiters=3).mask  # type: ignore
    calib = calib[~zp_mask]
    ZP_mean, ZP_std = np.mean(calib['ZP']), np.std(calib['ZP']) # type: ignore

    if print_info: print('ZP = %.3f, rms = %.2e'%(ZP_mean, ZP_std))
    # print('clipped ZP = %.3f, rms = %.2e'%(mean_zp, std_zp))
    if 'ZP' in header:
        if print_info: print('- ZP in header: %.3f, EZP: %.2e'%(header['ZP'], header['EZP']))
    if print_info: print('Number of stars used: %i'%len(calib))
    else: 
        print(' - Using %i catalogued stars'%np.sum(cat_mask))
        print('   Matched stars: %i'%len(calib))
        print('   ZP = %.3f, rms = %.2e'%(ZP_mean, ZP_std))


    # Calculate calibrated magnitudes
    phot_g_all['mag_calib'] = phot_g_all['mag_inst'] + ZP_mean
    phot_g_all['mag_calib_err'] = np.sqrt(phot_g_all['mag_inst_err']**2+ZP_std**2)  # type: ignore

    if plot:
        n_fig += 1
        plt.close(n_fig)
        fig, ax = plt.subplots(figsize = (10,4),num = n_fig)
        ax.scatter(phot_g_all['flux_id'], phot_g_all['mag_calib'], marker = '.') # type: ignore
        ax.errorbar(phot_g_all['flux_id'], phot_g_all['mag_calib'], yerr = phot_g_all['mag_calib_err'], # type: ignore
                    fmt="none", color = 'black') 
        ax.set_ylabel('Measured magnitude')
        ax.set_title('Filter %s'%fil)
        ax.grid()
        plt.show()

    return phot_g_all



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
        print('If you want to save the output table: table.to_pandas().to_pickle("./table.pkl")')
        
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
            iraf_finder = IRAFStarFinder(threshold=threshold_iraf, fwhm=fwhm, peakmax=60000)
            print('Detecting sources...')
            iraf_stars = iraf_finder(data_sub)
            iraf_stars.remove_rows(np.where(iraf_stars['peak'] > 60000)) 
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
            fp_sources.remove_rows(np.where(fp_sources['peak_value'] > 60000)) # type: ignore
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

def get_coordinates(filename, 
                    scalebar_arcsec = 60,
                    rotate = True,
                    ):
    with fits.open(filename) as hdul:
        hdu = hdul[0]
        header = hdu.header #type: ignore
        data = hdu.data #type: ignore
        wcs = WCS(header)

    print('Plotting image %s for coordinate selection...'%filename)
    print('Click on the image to obtain coordinates.')
    
    out = widgets.Output()
    display(out)

    if rotate:
        wcs_out, shape_out = find_optimal_celestial_wcs((data, wcs))
        data, _ = reproject_interp((data, wcs), wcs_out, shape_out=shape_out)
    
    fig,ax = plt.subplots()
    ax.remove()
    ax = fig.add_subplot(111, projection=wcs)
    # Hardcore background estimation for visualization
    sky_mean = np.nanmedian(data)
    sky_std = np.nanstd(data)
    ax.imshow(data, 
        cmap = 'gray', origin = 'lower',
        vmin =  sky_mean-(sky_std * 3),
        vmax =  sky_mean + sky_std * 3,
        )
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')

    #scalebar
    scalebar_angle = scalebar_arcsec/3600*u.deg # type: ignore
    add_scalebar(ax, scalebar_angle, label="%s arcsec"%str(scalebar_arcsec), 
                    color='white',
                    corner = 'bottom left')
    
    # Collecting clicked coordinates
    def onclick(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        # coords_px.append([x, y])
        radec = wcs.pixel_to_world(x, y)
        ax.plot(x, y, 'bx', markersize=10)
        fig.canvas.draw_idle()
        print('x y RA DEC: %f %f %f %f'%(x, y, radec.ra.deg, radec.dec.deg))  # type: ignore
        
        with out:  # capture stdout into output widget [web:266]
            print(f"x y = {x:.2f} {y:.2f}  |  RA Dec = {radec.ra.deg:.6f}d {radec.dec.deg:.6f}d") # type: ignore

    fig.canvas.mpl_connect('button_press_event', onclick)



def get_magnitude(filename, 
                  photometry_table, 
                  coords,
                  pix_dist = 5,
                  print_info = False,
                  show_plot = False,
                  ):
    if type(coords) is not SkyCoord: sky_coords = SkyCoord(coords)
    else: sky_coords = coords
    
    with fits.open(filename) as hdul:
        hdu = hdul[0]
        header = hdu.header #type: ignore
        data = hdu.data #type: ignore
        wcs = WCS(header)

    px_coords = skycoord_to_pixel(sky_coords, wcs)
    # Find the closest photometry entry to the given coordinates
    phot_xy = np.array([photometry_table["x_fit"], photometry_table["y_fit"]]).T
    dist = np.sqrt((phot_xy[:, 0] - px_coords[0])**2 + (phot_xy[:, 1] - px_coords[1])**2)
    closest_idx = np.argmin(dist)
    if print_info: 
        print('----------- get_magnitude info -----------')
        print('Input star:   x y = %f %f  |  RA Dec = %f %f'%(px_coords[0], px_coords[1], sky_coords.ra.deg, sky_coords.dec.deg))  # type: ignore
        closest_radec = wcs.pixel_to_world(phot_xy[closest_idx, 0], phot_xy[closest_idx, 1])
        print('Closest star: x y = %f %f  |  RA Dec = %f %f'%(phot_xy[closest_idx, 0], phot_xy[closest_idx, 1], closest_radec.ra.deg, closest_radec.dec.deg))  # type: ignore
        print('Closest star magnitude: %f at distance %.2f pixels'%(photometry_table['mag_calib'][closest_idx], dist[closest_idx]))
        print('--------------------------------------------')

    if show_plot:
        fig, ax = plt.subplots()
        x, y = int(phot_xy[closest_idx, 0]), int(phot_xy[closest_idx, 1])
        xstar, ystar = int(px_coords[0]), int(px_coords[1])
        rad = int(dist[closest_idx]*1.5)
        if rad < 10: rad = 10
        ax.imshow(data[y-rad:y+rad, x-rad:x+rad], cmap='gray', origin='lower')
        ax.plot(xstar-(x-rad), ystar-(y-rad), 'rx', label='Input star')
        ax.plot(x-(x-rad), y-(y-rad), 'b+', label='Closest star')
        ax.legend()
        plt.show()

    if dist[closest_idx] < pix_dist: # if the closest star is within 5 pixels, return its magnitude
        return photometry_table['mag_calib'][closest_idx]
    else: 
        print('No star found within %d pixels of the given coordinates.'%pix_dist)
        return None

    