"""
Photometric analysis functions:
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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
from astroquery.gaia import Gaia

import photutils.psf as psf
from photutils.detection import IRAFStarFinder, find_peaks

from scipy.spatial import cKDTree # type: ignore
from scipy.ndimage import binary_dilation


def photo_analysis(filename,
                   init_table = None,
                   cat_table = None,
                   sky_background = {'sigma': 3.0, 'maxiters': 5, 'sky_threshold': 3.0},
                   photometry_params = {'psf_fwhm_shape': 3.0, 'aperture_fwhm': 3.0, 'fitter_maxiters': 100,
                                       'qfit_filter': 8, 'cfit_filter': 0.05},
                   matching_params = {'mag_range': (13, 18), 'max_sep_pix': 5},
                   plot = True, n_fig_init = 0,
                   stacked = False,
                   print_info = True,
                   print_nothing = False,
                   ):
    n_fig = n_fig_init
    sky = {'mean':[], 'std':[]}

    if init_table is None:
        print('No table of stars "init_table" provided. Use detect_sources function to detect sources and perform photometric analysis.\n')
        print('Save the resulting table to pkl format by:')
        print(' - table.to_pandas().to_pickle("./table.pkl")')
        return 
    
    if cat_table is None:
        print('No catalogue table "cat_table" provided. Use get_catalogue function to query a catalogue and perform photometric analysis.\n')
        print('Save the resulting table to pkl format by:')
        print(' - table.to_pandas().to_pickle("./cat_table.pkl")')
        return
    
    with fits.open(filename) as hdul:
        hdu = hdul[0]
        header = hdu.header #type: ignore
        data = hdu.data #type: ignore
        wcs = WCS(header)

    fil = header['FILTER'] if 'FILTER' in header else 'Unknown filter'
    if fil=='SDSSzs':
        fil = 'SDSSz'
    
    if print_info:
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

    if not set(['ra','dec']).issubset(colnames):
        print('Initial table should have columns "ra", "dec". Please check format of "init_table".')
        return None
    
    if print_info: print('Initial number of star coordinates looked at: %i'%len(init_table))

    # Catalogue table
    if type(cat_table) == str:
        if cat_table.endswith('.pkl'):
            cat_table = pd.read_pickle(cat_table)
        elif cat_table.endswith('.csv'):
            cat_table = pd.read_csv(cat_table)
        else:
            print('Unsupported file format for initial table. Please provide a .pkl or .csv file.')
            return None
        if type(cat_table) == pd.DataFrame:
            colnames = cat_table.columns.to_list()
        elif type(cat_table) == Table:
            colnames = cat_table.colnames
        if not set(['ra','dec']).issubset(colnames):
            if not set(['x','y']).issubset(colnames):
                print('Catalogue table should have columns "ra", "dec" or "x", "y". Please check format of "cat_table".')
                return None
    
    cat_px = skycoord_to_pixel(SkyCoord(cat_table['ra'], cat_table['dec'], unit='deg'), wcs) # type: ignore
    init_px = skycoord_to_pixel(SkyCoord(init_table['ra'], init_table['dec'], unit='deg'), wcs) # type: ignore
    init_px_tbl = Table({'x': init_px[0], 'y': init_px[1]})

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
        ax.scatter(init_px[0], init_px[1], facecolor = 'none', edgecolor = 'red', marker = 's', label = 'Stars looked') # type: ignore
        ax.legend()
        title_str = 'Catalogued stars and known stars\nover image in filter %s'%header['FILTER'] if 'FILTER' in header else 'Catalogued stars and known stars'
        title_str += '\nStars in catalogue: %i'%len(cat_px[0])
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
                       init_params = init_px_tbl))
    phot_all.sort('flux_fit', reverse = True) 
    phot_all['flux_id'] = np.arange(len(phot_all), dtype=int)
    phot_all['peak_value'] = 0
    for i in range(len(phot_all)):
        try: 
            x_peak, y_peak = int(phot_all['x_fit'][i]), int(phot_all['y_fit'][i])  # type: ignore
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
        
        if len(phot_g_all) < 200:
            for i in range(len(phot_g_all)):
                plt.text(phot_g_all['x_fit'][i]+10, phot_g_all['y_fit'][i], phot_g_all['flux_id'][i]) # type: ignore
        ax.legend(ncols = 4, bbox_to_anchor = (0.5,1.08,0,0), loc='center')

        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        # ax.set_title('Good photometry stars: %i'%len(phot_g_all))
        plt.show()

    
    # Cross-correlation of queried stars and found stars
    phot_xy = np.array([phot_g_all["x_fit"], phot_g_all["y_fit"]]).T
    cat_xy = np.array([cat_px[0], cat_px[1]]).T
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
    calib["mag_cat"] = cat_table[fil[-1]][calib["cat_idx"]] # type: ignore
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
        if print_nothing == False:
            print(' - Using %i catalogued stars'%len(cat_table))
            print('   Matched stars: %i'%len(calib))
            print('   ZP = %.3f, rms = %.2e'%(ZP_mean, ZP_std))


    # Calculate calibrated magnitudes
    phot_g_all['mag_calib'] = phot_g_all['mag_inst'] + ZP_mean
    phot_g_all['mag_calib_err'] = np.sqrt(phot_g_all['mag_inst_err']**2+ZP_std**2)  # type: ignore

    # Calculate fitted ra dec
    radec_phot = wcs.pixel_to_world(phot_g_all['x_fit'], phot_g_all['y_fit']) # type: ignore
    phot_g_all['ra_fit'] = radec_phot.ra.deg # type: ignore
    phot_g_all['dec_fit'] = radec_phot.dec.deg # type: ignore

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


def get_catalogue(filename, 
                  catalogue = 'Simbad',
                  filter = 'g',
                  mag_range = (13, 18),
                  print_info = True,
                  plot = False, 
                  rotate = True,
                  scalebar_arcsec = 60,
                  ):
    
    if catalogue not in ['Simbad', 'SDSS', 'PanSTARRS', 'Gaia']:
        print('ERROR: the seelected catalogue is not in the list of known catalogues.')
        print(' - %s'%['Simbad', 'SDSS', 'PanSTARRS', 'Gaia'])
        return

    with fits.open(filename) as hdul:
        hdu = hdul[0]
        header = hdu.header #type: ignore
        data = hdu.data # type: ignore
        wcs = WCS(header)

    if filter is str:
        filter_list = [filter]
    else: filter_list = filter
    
    if not set(filter_list).issubset(['g','r','i','z']):
        print('Filters chosen must be in %s'%['g','r','i','z'])
        return None
    
    return_labels = ['ra', 'dec'] + filter_list # type: ignore
    
    # Defining sky area to search in SDSS catalogue
    sc_center = SkyCoord(header['CRVAL1']*u.deg, header['CRVAL2']*u.deg)
    # RA, DEC of image corners
    ny, nx = wcs.pixel_shape # type: ignore
    corners_pix = np.array([[0, 0], [nx-1, 0], [nx-1, ny-1], [0, ny-1]])
    corners_sky = wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
    # RA/Dec bounding box
    ra_range = [corners_sky.ra.min().value, corners_sky.ra.max().value] # type: ignore
    dec_range = [corners_sky.dec.min().value, corners_sky.dec.max().value] # type: ignore
    radius = Angle(np.max([ra_range[1]-ra_range[0], dec_range[1]-dec_range[0]])/2, u.deg)
    if print_info: print('Sky area observed in RA, DEC (deg): ')
    if print_info: print('   ', ra_range, dec_range)
    
    if print_info: print('Querying %s'%catalogue)
    
    if catalogue == 'SDSS':
        # cat_filter = 'psfMag_'
        cat_filter_list = ['psfMag_' + fil for fil in filter_list]
        cat_labels = ['ra', 'dec'] + cat_filter_list
        # Query SDSS with the sky area, obtain psfMag values for interest filters
        cat_table = SDSS.query_region(sc_center,                               # type: ignore
                                    width = (ra_range[1]-ra_range[0])*u.deg,
                                    height = (dec_range[1]-dec_range[0])*u.deg,
                                    fields = cat_labels
                                    )
        # Transforming obtained sky coordinates of stars to pixels in the image
        cat_px = skycoord_to_pixel(SkyCoord(cat_table['ra'], cat_table['dec'], unit='deg'), wcs)
        cat_table = cat_table[cat_labels]
    
    elif catalogue == 'PanSTARRS':
        # Query Vizier for panstarrs1
        #vizier = Vizier(columns = ['ra','dec', 'gmag', 'rmag', 'imag'])
        cat_filter_list = [fil + 'mag' for fil in filter_list]
        cat_labels = ['RAJ2000', 'DEJ2000'] + cat_filter_list
        pstr = Vizier.query_region(sc_center,                           # type: ignore
                                    width = (ra_range[1]-ra_range[0])*u.deg,
                                    height = (dec_range[1]-dec_range[0])*u.deg,
                                    catalog = 'II/349/ps1'
                                    #fields=['ra','dec', 'gmag', 'rmag', 'imag']
                                    )
        cat_table = pstr['II/349/ps1'][cat_labels] #['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag']
        cat_px = skycoord_to_pixel(SkyCoord(cat_table['RAJ2000'], cat_table['DEJ2000'], unit='deg'), wcs)
        # cat_labels = ['ra', 'dec', filter+'mag']

    elif catalogue == 'Simbad':
        # Query Simbad for stars in the field
        cat_filter_list = filter_list
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('ra(d)', 'dec(d)', 'g', 'r', 'i', 'z')
        cat_table = custom_simbad.query_region(sc_center, radius = radius)
        cat_px = skycoord_to_pixel(SkyCoord(cat_table['ra'], cat_table['dec'], unit='deg'), wcs)
        cat_labels = ['ra', 'dec'] + cat_filter_list # type: ignore
        cat_table = cat_table[cat_labels]

    elif catalogue == 'Gaia':
        # Query Gaia for stars in the field
        Gaia.ROW_LIMIT = 1000
        # j = Gaia.cone_search_async(sc_center, 
        #                            radius=Angle(np.max([ra_range[1]-ra_range[0], dec_range[1]-dec_range[0]])/2, u.deg),
        #                            columns = ['ra', 'dec', 'phot_g_mean_mag', ])
        # t = j.get_results()
        N = 10000
        # adql = f"""
        #         SELECT TOP {N}
        #         gspc.source_id,
        #         gs.ra, gs.dec,
        #         gspc.g_sdss_mag, gspc.r_sdss_mag, gspc.i_sdss_mag,
        #         gspc.g_sdss_flux_error, gspc.r_sdss_flux_error, gspc.i_sdss_flux_error
        #         FROM gaiadr3.synthetic_photometry_gspc AS gspc
        #         JOIN gaiadr3.gaia_source AS gs USING (source_id)
        #         WHERE 1=CONTAINS(
        #         POINT('ICRS', gs.ra, gs.dec),
        #         CIRCLE('ICRS', {sc_center.ra.deg}, {sc_center.dec.deg}, {radius.deg/2})
        #         )
        #         """
        adql = f"""
                SELECT TOP {N}
                    gs.source_id, gs.ra, gs.dec,
                    gspc.g_sdss_mag, gspc.r_sdss_mag, gspc.i_sdss_mag, gspc.z_sdss_mag
                FROM gaiadr3.gaia_source_lite AS gs
                JOIN gaiadr3.synthetic_photometry_gspc AS gspc
                    USING (source_id)
                WHERE 1=CONTAINS(
                    POINT('ICRS', gs.ra, gs.dec),
                    CIRCLE('ICRS', {sc_center.ra.deg}, {sc_center.dec.deg}, {radius.deg/2})
                    )
                AND gs.phot_g_mean_mag > {mag_range[0]}
                AND gs.phot_g_mean_mag < {mag_range[1]}
                AND gspc.g_sdss_flag = 0
                AND gspc.r_sdss_flag = 0
                AND gspc.i_sdss_flag = 0
                AND gspc.z_sdss_flag = 0
                """
        job = Gaia.launch_job_async(
                                adql,
                                dump_to_file=True,
                                output_format="votable_gzip",   # small on disk
                                output_file="gaia_sdss_griz.vot.gz"
                            ) 
        # job = Gaia.launch_job_async(adql)
        # If you want the table in memory afterward (can be big):
        tbl = job.get_results()
        print(len(tbl), tbl.colnames) # type: ignore
        # t = job.get_results()
        return tbl
        #cat_table = r['ra', 'dec', 'Gmag', ]
    
    if print_info: print(' - Found %d stars in the field'%len(cat_table))

    return_table = cat_table[cat_labels].copy()
    # rename columns
    for o, n in zip(cat_labels, return_labels, strict=True):
        return_table.rename_column(o, n)
    
    # filtering out all rows without filter info
    # row is "all missing" if every kept column is masked on that row
    all_missing = np.ones(len(return_table), dtype=bool)
    for c in filter_list:
        col = return_table[c]
        m = getattr(col, "mask", None)
        if m is None:
            # Non-masked column => treat as never-missing for this test
            m = np.zeros(len(return_table), dtype=bool)
        all_missing &= np.array(m, dtype=bool)
        # remove nans too
        mag_vals = np.array(col)
        if np.issubdtype(mag_vals.dtype, np.floating):
            all_missing &= ~np.isnan(mag_vals)

    return_table = return_table[~all_missing]
    # filtering out all rows with all mags out of the selected range
    good_mag = np.ones(len(return_table), dtype=bool)
    for fil in filter_list:
        mag_fil = np.array(return_table[fil])
        good_mag &= (mag_fil > mag_range[0]) 
        good_mag &= (mag_fil < mag_range[1]) 

    return_table = return_table[good_mag]

    if print_info: print('Filtered down to %i stars catalogued in %s with filter information in the magnitude range selected.'%(len(return_table), catalogue))

    if plot:
        if rotate:
            wcs_old = wcs
            wcs, shape_out = find_optimal_celestial_wcs((data, wcs_old))
            data, _ = reproject_interp((data, wcs_old), wcs, shape_out=shape_out)
        
        return_xy = skycoord_to_pixel(SkyCoord(return_table['ra'], return_table['dec'], unit = 'deg'), wcs)

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
        
        ax.scatter(return_xy[0], return_xy[1], 
                   marker = 'o', color='red', facecolor = 'none', 
                   label = '%s stars'%catalogue)
        ax.set_title('Found %d good stars in %s'%(len(return_table), catalogue))
        ax.legend()
        
    return return_table


def explore_catalogues(filename,
                       filter,
                       mag_range=(13, 22),
                       catalogues=None,
                       include_gaia=False,
                       gaia_mag_limit=22.0,
                       ):
    """
    Compare star catalogue coverage over the actual detector footprint of a FITS image.

    For each catalogue the function:
    1. Calls get_catalogue() to query the sky bounding-box around the image.
    2. Projects every returned star to pixel coordinates via the image WCS.
    3. Labels stars that fall inside the detector boundary as on_detector.
    4. Plots the image with green (on-detector) and red (outside) circles plus
       a magnitude histogram of the usable population.

    Parameters
    ----------
    filename      : str  — path to a representative FITS file
    filter        : str  — single filter letter, e.g. 'g' for SDSSg images
    mag_range     : tuple (bright, faint) — magnitude limits for calibration stars
    catalogues    : list of str — any subset of ['SDSS', 'PanSTARRS', 'Simbad']
                    (default: all three)
    include_gaia  : bool — also query Gaia DR3 via Vizier (returns G-band magnitudes,
                    NOT the SDSS-filter magnitude; shown for coverage comparison only)
    gaia_mag_limit: float — Gaia G-mag faint limit when include_gaia=True

    Returns
    -------
    dict {catalogue_name: pandas DataFrame of stars on the detector}
    """
    if catalogues is None:
        catalogues = ['SDSS', 'PanSTARRS', 'Simbad']

    with fits.open(filename) as hdul:
        data    = hdul[0].data.astype(float)  # type: ignore
        header  = hdul[0].header               # type: ignore
        wcs_img = WCS(header)

    ny, nx = data.shape
    _, sky_med, sky_std = sigma_clipped_stats(data, sigma=3, maxiters=5)

    # ── Query each catalogue ──────────────────────────────────────────────────
    cat_results = {}   # {name: {'df': DataFrame, 'mag_col': str} | None}

    for cat_name in catalogues:
        print(f'Querying {cat_name} ...', end='  ', flush=True)
        try:
            cat_table = get_catalogue(
                filename,
                catalogue=cat_name,
                filter=[filter],
                mag_range=mag_range,
                plot=False,
                print_info=False,
            )
        except Exception as e:
            print(f'FAILED: {e}')
            cat_results[cat_name] = None
            continue
        if cat_table is None or len(cat_table) == 0:
            print('0 stars returned.')
            cat_results[cat_name] = None
            continue
        df = cat_table.to_pandas()
        print(f'{len(df)} stars in query area')
        cat_results[cat_name] = {'df': df, 'mag_col': filter}

    # ── Optional: Gaia DR3 via Vizier (G band) ───────────────────────────────
    if include_gaia:
        print('Querying Gaia DR3 (Vizier) ...', end='  ', flush=True)
        try:
            sc_ctr = SkyCoord(ra=header['CRVAL1'] * u.deg, dec=header['CRVAL2'] * u.deg)
            corners_sky = wcs_img.pixel_to_world(
                np.array([0, nx - 1, nx - 1, 0]),
                np.array([0, 0,      ny - 1, ny - 1]),
            )
            r_arcmin = max(
                sc_ctr.separation(c).to(u.arcmin).value for c in corners_sky
            ) * u.arcmin
            v = Vizier(catalog='I/355/gaiadr3',
                       columns=['RA_ICRS', 'DE_ICRS', 'Gmag', 'BP-RP'],
                       row_limit=-1,
                       column_filters={'Gmag': f'< {gaia_mag_limit}'})
            res = v.query_region(sc_ctr, radius=r_arcmin)
            if res and len(res) > 0:
                gdf = res[0].to_pandas().rename(columns={'RA_ICRS': 'ra', 'DE_ICRS': 'dec'})
                print(f'{len(gdf)} stars in query area')
                cat_results['Gaia DR3 (G)'] = {'df': gdf, 'mag_col': 'Gmag'}
            else:
                print('0 stars returned.')
                cat_results['Gaia DR3 (G)'] = None
        except Exception as e:
            print(f'FAILED: {e}')
            cat_results['Gaia DR3 (G)'] = None

    # ── Project to pixels, flag on/off detector ───────────────────────────────
    on_counts = {}
    for cat_name, entry in cat_results.items():
        if entry is None:
            on_counts[cat_name] = 0
            continue
        df = entry['df']
        coords = SkyCoord(ra=df['ra'].values * u.deg, dec=df['dec'].values * u.deg)
        x_pix, y_pix = wcs_img.world_to_pixel(coords)
        df['x_pix'] = x_pix
        df['y_pix'] = y_pix
        df['on_detector'] = ((x_pix >= 0) & (x_pix < nx) &
                             (y_pix >= 0) & (y_pix < ny))
        on_counts[cat_name] = int(df['on_detector'].sum())

    # ── Figure ────────────────────────────────────────────────────────────────
    n_cols = len(cat_results)
    if n_cols == 0:
        print('No catalogue results to display.')
        return {}

    vmin_img = sky_med - 3 * sky_std
    vmax_img = sky_med + 3 * sky_std
    best_n   = max(on_counts.values()) if on_counts else 0

    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 10))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, (cat_name, entry) in enumerate(cat_results.items()):
        ax_img  = axes[0, col]  # type: ignore
        ax_hist = axes[1, col]  # type: ignore

        ax_img.imshow(data, origin='lower', cmap='gray',
                      vmin=vmin_img, vmax=vmax_img,
                      aspect='equal', interpolation='nearest')
        ax_img.set_xlim(-nx * 0.05, nx * 1.05)
        ax_img.set_ylim(-ny * 0.05, ny * 1.05)

        if entry is not None:
            df      = entry['df']
            mag_col = entry['mag_col']
            on  = df['on_detector']
            off = ~on

            if off.sum() > 0:
                ax_img.scatter(df.loc[off, 'x_pix'], df.loc[off, 'y_pix'],
                               facecolor='none', edgecolor='tomato',
                               s=50, lw=0.9, label=f'outside ({off.sum()})', zorder=3)
            if on.sum() > 0:
                ax_img.scatter(df.loc[on, 'x_pix'], df.loc[on, 'y_pix'],
                               facecolor='none', edgecolor='lime',
                               s=70, lw=1.3, label=f'on detector ({on.sum()})', zorder=4)

            n_on    = int(on.sum())
            n_tot   = len(df)
            suffix  = ' ★' if (n_on == best_n and best_n > 0) else ''
            ax_img.set_title(f'{cat_name}{suffix}\n{n_on} / {n_tot} stars on detector',
                             fontsize=10, fontweight='bold' if suffix else 'normal')
            ax_img.legend(fontsize=7, loc='upper right')

            mag_label = (f'{filter}-band' if mag_col == filter else 'G-band (not SDSS-g)')
            lo = mag_range[0] if mag_col == filter else float(df[mag_col].min())
            hi = mag_range[1] if mag_col == filter else float(df[mag_col].max())
            bins = np.linspace(lo, hi, 30)
            if on.sum() > 0:
                ax_hist.hist(df.loc[on,  mag_col].dropna(), bins=bins,
                             color='lime', edgecolor='white', alpha=0.85,
                             label=f'on detector (n={n_on})')
            if off.sum() > 0:
                ax_hist.hist(df.loc[off, mag_col].dropna(), bins=bins,
                             color='salmon', edgecolor='white', alpha=0.6,
                             label=f'outside (n={int(off.sum())})')
            ax_hist.set_xlabel(f'{cat_name}  {mag_label}')
            ax_hist.legend(fontsize=7)
        else:
            for ax in (ax_img, ax_hist):
                ax.text(0.5, 0.5, f'{cat_name}\nquery failed\nor no stars',
                        ha='center', va='center', transform=ax.transAxes,
                        color='red', fontsize=10)
            ax_img.set_title(cat_name)

        ax_img.set_xlabel('x (pix)')
        if col == 0:
            ax_img.set_ylabel('y (pix)')
            ax_hist.set_ylabel('N stars')

    plt.suptitle(
        f'Catalogue coverage — {filename.split("/")[-1]}\n'
        f'green = on detector  ·  red = outside bounding-box',
        fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n── On-detector star count ────────────────────────────────────────────')
    print(f'   Detector: {nx} × {ny} pix')
    for cat_name, entry in cat_results.items():
        if entry is None:
            print(f'   {cat_name:16s}: query failed')
            continue
        n_on  = int(entry['df']['on_detector'].sum())
        n_tot = len(entry['df'])
        flag  = '  ← most stars' if (n_on == best_n and best_n > 0) else ''
        print(f'   {cat_name:16s}: {n_on:4d} / {n_tot} on detector{flag}')
    print('\n→ Pass the winning catalogue name to get_catalogue() or set PHOT_CATALOGUE in §0.')

    return {k: v['df'][v['df']['on_detector']].copy()
            for k, v in cat_results.items() if v is not None}


def build_known_mask(filename, coords_df, fwhm_pix=None):
    """
    Build a boolean pixel mask with True where known sources are located.

    Parameters
    ----------
    filename : str
        Path to a FITS image (used for WCS and image dimensions).
    coords_df : pd.DataFrame
        Must contain columns 'ra' and 'dec' (degrees) of known source positions.
    fwhm_pix : float, optional
        Mask radius = ceil(2 × fwhm_pix). If None, reads FWHM from FITS header (default 3.0).

    Returns
    -------
    np.ndarray  (dtype=bool, shape=(NAXIS2, NAXIS1))  True = masked (known source)
    """
    with fits.open(filename) as hdul:
        header = hdul[0].header
    wcs_obj = WCS(header)
    naxis1, naxis2 = header['NAXIS1'], header['NAXIS2']
    if fwhm_pix is None:
        fwhm_pix = header.get('FWHM', 3.0)

    coords = SkyCoord(ra=coords_df['ra'].values * u.deg,
                      dec=coords_df['dec'].values * u.deg)
    x_pix, y_pix = skycoord_to_pixel(coords, wcs_obj)

    point_mask = np.zeros((naxis2, naxis1), dtype=bool)
    for xi, yi in zip(x_pix, y_pix):
        if np.isnan(xi) or np.isnan(yi):
            continue
        xi_i, yi_i = int(round(float(xi))), int(round(float(yi)))
        if 0 <= xi_i < naxis1 and 0 <= yi_i < naxis2:
            point_mask[yi_i, xi_i] = True

    radius = int(np.ceil(2 * fwhm_pix))
    r_y, r_x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk_struct = (r_x ** 2 + r_y ** 2) <= radius ** 2
    return binary_dilation(point_mask, structure=disk_struct)


def detect_sources(filename,
                   method = 'find_peaks',
                   sky_sigma = 3.0,
                   maxiters = 5,
                   sky_threshold = 3.0,
                   fwhm = 3.0,
                   init_table = None,
                   add_sources = False,
                   mask = None,
                   plot = True,
                   print_info = True,
                   ):
    
    with fits.open(filename) as hdul:
        hdu = hdul[0]
        header = hdu.header #type: ignore
        data = hdu.data #type: ignore
        wcs = WCS(header)

    methods_dict = {'IRAF': 'IRAFStarFinder',
                    'find_peaks': 'find_peaks'}

    # Background estimation and substraction
    if print_info: print('Removing sky background...')
    sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=sky_sigma, maxiters=maxiters)
    bkg = sky_mean
    data_sub = data - bkg
    if print_info: print('- Sky background mean: %.2f; Sky background std: %.2f'%(sky_mean, sky_std))

    if init_table is None:
        if print_info: 
            print('No initial table provided. Will perform source detection on the image.')
            print('If you want to save the output table: table.to_pandas().to_pickle("./table.pkl")')
        
        if 'FWHM'  in header:
            fwhm = header['FWHM']
        elif 'seeing' in header:
            fwhm = header['seeing'] / header['SCALE']
        else:
            if print_info: print("FWHM not found in header. Using input value fwhm = %s."%fwhm)

        if print_info: print('FWHM used for source detection: %.2f pixels'%fwhm)

        if method in methods_dict.keys():
            if print_info: print('No initial table provided. Using %s for source detection.'%methods_dict[method])
        threshold_method =  (sky_std * sky_threshold)
        if print_info:
            print('- %s threshold: %.2f'%(methods_dict[method], threshold_method))
            print('Detecting sources...')
            print(' - Using mask for known sources: %s'%('Yes' if mask is not None else 'No'))
        # Source detection
        if method == 'IRAF':
            iraf_finder = IRAFStarFinder(threshold=threshold_method, fwhm=fwhm, peakmax=60000)
            iraf_stars = iraf_finder(data_sub, mask=mask)
            iraf_stars.remove_rows(np.where(iraf_stars['peak'] > 60000)) 
            if print_info: print('- Stars found by IRAFStarFinder: %d'%len(iraf_stars))
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
            fp_sources = find_peaks(data_sub, threshold=threshold_method,
                                    box_size=int(fwhm),
                                    wcs = WCS(header),
                                    mask=mask)
            if print_info: print('- Sources found by find_peaks: %d'%len(fp_sources)) # type: ignore
            fp_sources.remove_rows(np.where(fp_sources['peak_value'] > 60000)) # type: ignore
            detect_table = fp_sources
            xlab, ylab = 'x_peak', 'y_peak'

        else:
            if print_info:
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
        if not set(['ra','dec']).issubset(colnames):
            # if 'x' not in colnames or 'y' not in colnames:
            if not set(['x','y']).issubset(colnames):
                print('Initial table should have columns "ra", "dec" or "x", "y". Please check the format of the initial table.')
                return None
        else:
            init_px = skycoord_to_pixel(SkyCoord(init_table['ra'], init_table['dec'], unit = 'deg'), wcs)
            init_table = Table({'x':init_px[0], 'y': init_px[1]})
        if print_info: print('Initial table provided. Skipping source detection on the image.')
        xlab, ylab = 'x', 'y'
        detect_table = init_table

    xs, ys = np.asarray(detect_table[xlab]), np.asarray(detect_table[ylab]) # type: ignore
    xy_stars = Table({'x': xs, 'y': ys}) # type: ignore
    radec = wcs.pixel_to_world(xs, ys)
    radec_stars = Table({'ra': np.asarray(radec.ra.deg), 'dec': np.asarray(radec.dec.deg)}) # type: ignore

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
            print('  If you want to save the output table: table.to_pandas().to_pickle("./table.pkl")')
            ax.set_title('Detected sources: %d\nClick to add sources'%len(xy_stars))

            def onclick(event):
                # if event.inaxes != ax or event.xdata is None or event.ydata is None:
                #     return
                if event.inaxes is None:
                    print("Event inaxes = None")
                    return
                if event.inaxes != ax:
                    print("Clicked in different axes:", event.inaxes, "expected:", ax)
                    return
                if event.xdata is None or event.ydata is None:
                    print('Event xdata or ydata is None')
                    return
                x, y = event.xdata, event.ydata
                radec_i = wcs.pixel_to_world(x,y)
                xy_stars.add_row([x, y])
                radec_stars.add_row([radec_i.ra.deg, radec_i.dec.deg]) # type: ignore
                ax.plot(x, y, 'bx', markersize=10)
                ax.set_title('Detected sources: %d\nClick to add sources'%len(xy_stars))
                fig.canvas.draw()

            cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    return radec_stars

def get_coordinates(filename, 
                    coords = None,
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
        wcs_old = wcs
        wcs, shape_out = find_optimal_celestial_wcs((data, wcs_old))
        data, _ = reproject_interp((data, wcs_old), wcs, shape_out=shape_out)

    if type(coords) is not type(None):
        if type(coords) is not SkyCoord: sky_coords = SkyCoord(coords)
        else: sky_coords = coords
        px_coords = skycoord_to_pixel(sky_coords, wcs)
        print('  Adding coordinates: %s'%coords)

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
    
    if type(coords) is not type(None):
        ax.scatter(px_coords[0], px_coords[1],
                   marker = 'o', color = 'red',
                   facecolor = 'none',
                   label = 'Input coordinates')
        ax.legend()
    
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
        
        with out:  # capture stdout into output widget
            print(f"x y = {x:.2f} {y:.2f}  |  RA Dec = {radec.ra.deg:.6f}d {radec.dec.deg:.6f}d") # type: ignore

    fig.canvas.mpl_connect('button_press_event', onclick)



def get_magnitude(filename, 
                  photometry_table, 
                  coords,
                  multiple_stars = False,
                  pix_dist = 5,
                  print_info = False,
                  plot = False,
                  plot_rad_pix = False,
                  n_fig = 99,
                  ):
    
    if multiple_stars == False:
        if type(coords) is not SkyCoord: sky_coords = SkyCoord(coords)
        else: sky_coords = coords
        sky_coords_list = [sky_coords]
    else:
        if type(coords) is not list:
            print('If multiple_stars is True, coords should be a list of coordinates.')
            return None
        sky_coords_list = []
        for c in coords:
            if type(c) is not SkyCoord: sky_coords_list.append(SkyCoord(c))
            else: sky_coords_list.append(c)

    with fits.open(filename) as hdul:
        hdu = hdul[0]
        header = hdu.header #type: ignore
        data = hdu.data #type: ignore
        wcs = WCS(header)
    
    phot_xy = np.array([photometry_table["x_fit"], photometry_table["y_fit"]]).T

    return_mags = []
    for sky_coords in sky_coords_list:
        px_coords = skycoord_to_pixel(sky_coords, wcs)
        # Find the closest photometry entry to the given coordinates
        dist = np.sqrt((phot_xy[:, 0] - px_coords[0])**2 + (phot_xy[:, 1] - px_coords[1])**2)
        closest_idx = np.argmin(dist)
        if print_info: 
            print('----------- get_magnitude info -----------')
            print('Input star:   x y = %f %f  |  RA Dec = %f %f'%(px_coords[0], px_coords[1], sky_coords.ra.deg, sky_coords.dec.deg))  # type: ignore
            closest_radec = wcs.pixel_to_world(phot_xy[closest_idx, 0], phot_xy[closest_idx, 1])
            print('Closest star: x y = %f %f  |  RA Dec = %f %f'%(phot_xy[closest_idx, 0], phot_xy[closest_idx, 1], closest_radec.ra.deg, closest_radec.dec.deg))  # type: ignore
            print('Closest star magnitude: %f at distance %.2f pixels'%(photometry_table['mag_calib'][closest_idx], dist[closest_idx]))
            print('--------------------------------------------')

        if plot:
            pix_radius = np.sqrt(photometry_table['npixfit'][closest_idx]/np.pi)
            plt.close(n_fig)
            fig, ax = plt.subplots(num = n_fig)
            ax.remove()
            ax = fig.add_subplot(111, projection = wcs)
            x, y = int(phot_xy[closest_idx, 0]), int(phot_xy[closest_idx, 1])
            xstar, ystar = int(px_coords[0]), int(px_coords[1])
            rad = int(dist[closest_idx]*1.5)
            if rad < 10: rad = 10
            if plot_rad_pix!=False: rad = plot_rad_pix
            ax.imshow(data[y-rad:y+rad, x-rad:x+rad], cmap='gray', origin='lower')
            ax.plot(xstar-(x-rad), ystar-(y-rad), 'rx', label='Input star')
            ax.plot(x-(x-rad), y-(y-rad), 'b+', label='Closest star')
            c = Circle((rad, rad),
                        pix_radius,
                        edgecolor = 'blue',
                        facecolor = 'none')
            ax.add_patch(c)
            ax.legend()
            ax.set_xlabel('RA')
            ax.set_ylabel('DEC')
            plt.show()

        if dist[closest_idx] < pix_dist: # if the closest star is within 5 pixels, return its magnitude
            return_mags.append([photometry_table['mag_calib'][closest_idx], photometry_table['mag_calib_err'][closest_idx]])
        else:
            if print_info:
                print('No star found within %d pixels of the given coordinates.'%pix_dist)
                print('- Closest star at %.1f pixels'%dist[closest_idx])
            return_mags.append([None, None])
    if multiple_stars == False:
        return return_mags[0]
    else:
        return return_mags     