"""
Utility functions for image viewer

- `filter_df`: Filter the dataframe of FITS files based on specified criteria (e.g., seeing, moon phase, integration time, zero point).
- `final_wcs`: Compute the final World Coordinate System (WCS) for an image, taking into account any necessary corrections or adjustments.
- `stacking_wcs`: Compute the WCS for a stacked image, which may involve combining multiple exposures and accounting for any shifts or rotations between them.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip, sigma_clipped_stats, mad_std
from astropy.nddata import CCDData
import ccdproc as ccdp
from reproject import reproject_interp

def filter_df(original_df, 
            filters_dict,
            ask_all = False,
            plotting = {
                'bool': False}
        ):
    '''Method to filter the dataframe and optionally plot the distributions before and after filtering.

    Parameters
    ---------
    filters : dict
        Dictionary with filtering conditions. Possible keys: any column in the dataset ('seeing', 'moon', 'EZP', 'DUSTPLA', 'AIRMASS', 'TESSMAG', 'filter').

    ask_all : bool, optional
        If True, asks for how to appluy each filter (above, below or equal to the filter value).

    plotting : dict, optional
        Dictionary with plotting options. If 'bool' is True, plots the distributions before and after filtering.
        Possible keys: 'n_bins', 'figsize', 'group_cols'.
    '''

    plotting_kw = {'variable': None,
                    'n_bins': 100, 
                    'figsize_frame': (4,3), 
                    'group_together': None,
                    'group_separate': None,
                    'plot_all': False,
                    'x_tight': False,
                    'log': False,
                    }
    if plotting['bool']:
        for key in plotting_kw.keys():
            if key not in plotting.keys():
                plotting[key] = plotting_kw[key]
    howto_dict = {'a': '>', 'b': '<', 'e': '='}
    # Decide whether to filter values above, below or equal to each filter value
    list_filter = []
    for key in filters_dict.keys():
        if type(filters_dict[key]) is tuple:
            list_filter.append('ab')
        elif key in ['filter', 'telescope', 'camera', 'object', 'im_type'] and not ask_all:
            list_filter.append('e')
        elif key in ['seeing', 'EZP', 'DUSTPLA', 'AIRMASS'] and not ask_all:
            list_filter.append('b')
        elif key in ['moon', 'TESSMAG'] and not ask_all:
            list_filter.append('a')
        else:
            howto = ''
            while howto not in ['a','b','e', 'ab']:
                try:
                    howto = str(input('- Filtering by \"%s\" with value: %s, do you want to filter (a)bove, (b)elow or (e)qual? '%(key, filters_dict[key])))
                except:
                    print('Invalid input. Input must be a, b or e. Try again.')
                    howto = str(input('- Filtering by \"%s\" with value: %s, do you want to filter (a)bove, (b)elow or (e)qual? '%(key, filters_dict[key])))
            list_filter.append(howto)

    print('Filtering dataset with:')
    for i, key in enumerate(filters_dict.keys()):
        if type(filters_dict[key]) is tuple:
            print(' - %s < %s < %s'%(filters_dict[key][0], key, filters_dict[key][1]))
        else:
            print(' - %s %s %s'%(key, howto_dict[list_filter[i]], filters_dict[key]))

    df_filtered = original_df.copy()
    for i, key in enumerate(filters_dict.keys()):
        if key not in original_df.columns:
                print('ERROR: \"%s\" is not in the available columns: %s'%(key, original_df.columns.tolist()))
        elif list_filter[i] == 'a':
            df_filtered = df_filtered[df_filtered[key]>filters_dict[key]]
        elif list_filter[i] == 'b':
            df_filtered = df_filtered[df_filtered[key]<filters_dict[key]]
        elif list_filter[i] == 'ab':
            df_filtered = df_filtered[(df_filtered[key]> filters_dict[key][0]) & (df_filtered[key]< filters_dict[key][1])]
        elif list_filter[i] == 'e':
            if type(filters_dict[key])!=list:
                df_filtered = df_filtered[df_filtered[key]==filters_dict[key]]
            else:
                df_temp = pd.DataFrame()
                for val in filters_dict[key]:
                    df_temp = pd.concat([df_temp, df_filtered[df_filtered[key]==val]])
                df_filtered = df_temp
        else:
            print('ERROR: unrecognized filtering condition. Skipping filter by \"%s\"'%key)
        if len(df_filtered) == 0:
            print('**WARNING**: No data left after filtering by \"%s\"'%key)
            print('Aborting filtering process...')
            return
    print('Number of files before filtering: %d'%len(original_df))
    print('Number of files after filtering: %d'%len(df_filtered))

    # Plotting results
    if plotting['bool']:
        # Variables to plot separately
        if plotting['group_separate'] is not None:
            group_separate_values = df_filtered[plotting['group_separate']].unique()
            n_separate = len(group_separate_values)
        else: n_separate = 1
        # Variables to plot together
        if plotting['group_together'] is not None:
            group_together_values = df_filtered[plotting['group_together']].unique()
            n_together = len(group_together_values)
        if plotting['variable'] is None:
            print('ERROR: To plot filtering results, please provide a variable to plot in plotting[\'variable\'].')
            return
        if type(plotting['variable'])== str:
            plotting['variable'] = [plotting['variable']]
        for i, var in enumerate(plotting['variable']):
            if var not in original_df.columns:
                print('ERROR: \"%s\" is not in the available columns: %s'%(var, original_df.columns.tolist()))
            else:
                fig, ax = plt.subplots(ncols = n_separate, nrows = 1, 
                                        figsize = (plotting['figsize_frame'][0]*n_separate, plotting['figsize_frame'][1]))
                var_ext = [var]
                if plotting['group_together'] is not None:
                    var_ext.append(plotting['group_together'])

                if plotting['group_separate'] is None:
                    ax = [ax]
                    df_plot = df_filtered[var_ext]
                    df_plot_all = original_df[var_ext]

                for j in range(n_separate):
                    if plotting['group_separate'] is not None:
                        df_plot = df_filtered[var_ext][df_filtered[plotting['group_separate']]==group_separate_values[j]]
                        df_plot_all = original_df[var_ext][original_df[plotting['group_separate']]==group_separate_values[j]]
                        ax[j].set_title('%s: %s'%(plotting['group_separate'], group_separate_values[j]))

                    ax[j].hist(df_plot[var], bins = plotting['n_bins'], 
                                label = '$N_{filtered}$: %s'%len(df_plot),
                                histtype = 'step')
                    min_filt, max_filt = ax[j].get_xlim()
                    bin_width = (max_filt - min_filt)/plotting['n_bins']

                    if plotting['group_together'] is not None:
                        for k in range(n_together):
                            df_plot_together = df_plot[df_plot[plotting['group_together']]==group_together_values[k]]
                            #TODO: ERROR here when there are no rows of one of the combinations of groupt together and separate (filter and camera to test)
                            bins_together = int(plotting['n_bins'] * (df_plot_together[var].max()-df_plot_together[var].min())/(max_filt - min_filt))
                            arr_bins_together = np.arange(df_plot_together[var].min(), df_plot_together[var].max()+bin_width, bin_width)
                            ax[j].hist(df_plot_together[var], bins = arr_bins_together, 
                                        label = '$N_{%s}$: %s'%(group_together_values[k], len(df_plot_together)),
                                        histtype = 'step')
                    if plotting['plot_all']:
                        min_all, max_all = df_plot_all[var].min(), df_plot_all[var].max()
                        bins_all = int(plotting['n_bins'] * (max_all-min_all)/(max_filt - min_filt))
                        arr_bins_all = np.arange(min_all, max_all+bin_width, bin_width)
                        ax[j].hist(df_plot_all[var], bins = arr_bins_all, 
                                    label = '$N_{all}$: %s'%len(df_plot_all),
                                    histtype = 'step', color ='gray')
                        if plotting['x_tight']: ax[j].set_xlim(min_filt, max_filt)
                        else: 
                            if var in filters_dict.keys(): 
                                if list_filter[list(filters_dict.keys()).index(var)] == 'ab':
                                    ax[j].axvline(filters_dict[var][0], color = 'red', linewidth = 0.7, alpha = 0.5,label = '%s = %s'%(var, filters_dict[var][0]))
                                    ax[j].axvline(filters_dict[var][1], color = 'red', linewidth = 0.7, alpha = 0.5,label = '%s = %s'%(var, filters_dict[var][1]))
                                else:
                                    ax[j].axvline(filters_dict[var], color = 'red', linewidth = 0.7, alpha = 0.5,label = '%s = %s'%(var, filters_dict[var]))
                        if plotting['log']: ax[j].set_yscale('log')
                        ax[j].legend()
                    ax[j].set_xlabel(var)
                    
                ax[0].set_ylabel('Number of observations')
                if var in filters_dict.keys():
                    if list_filter[list(filters_dict.keys()).index(var)] == 'ab':
                        fig.suptitle('Filtering %s < \"%s\" < %s'%(filters_dict[var][0], var, filters_dict[var][1]))
                    else:
                        fig.suptitle('Filtering \"%s\" %s %s'%(var, howto_dict[list_filter[list(filters_dict.keys()).index(var)]], filters_dict[var]))
                else:
                    fig.suptitle('Plotting \"%s\" after filtering'%var)
                plt.tight_layout()
                plt.show()

    return df_filtered


def final_wcs(object, ra, dec, fov_x, fov_y, pixscale,
             name_out = "output_template.fits"):
    # Creating new folder for combined image
    if object+'_image' not in os.listdir():
        os.system('mkdir '+object+'_image')
        
    ra_center, dec_center  = Angle(ra).deg, Angle(dec).deg 
    # fov_x, fov_y, pixscale in arcsec
    fov_x_deg, fov_y_deg = Angle(fov_x).deg, Angle(fov_y).deg
    # Convert to degrees
    pixscale_deg = pixscale / 3600.0

    # Image size in pixels (rounded to integer)
    naxis1 = int(np.round(fov_x_deg / pixscale_deg))
    naxis2 = int(np.round(fov_y_deg / pixscale_deg))

    print('Creating WCS:')
    print('  Center (RA, DEC): ', 
          Angle(ra).to_string(),
          Angle(dec_center*u.deg).to_string())
    print('  Size (deg): ',
          Angle(fov_x_deg*u.deg).to_string(),
          Angle(fov_y_deg*u.deg).to_string())
    # Define WCS with TAN projection, north up, east left
    w = WCS(naxis=2)
    w.wcs.crval = [ra_center, dec_center]          # sky coord at reference pixel
    w.wcs.crpix = [naxis1 / 2.0, naxis2 / 2.0]     # reference pixel at image center
    w.wcs.cdelt = np.array([-pixscale_deg, pixscale_deg])  # RA decreases to the right
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]

    # Convert to FITS header and add NAXIS
    hdr_out = w.to_header()
    hdr_out["NAXIS"]  = 2
    hdr_out["NAXIS1"] = naxis1
    hdr_out["NAXIS2"] = naxis2
    hdr_out["SCALE"]  = pixscale
    
    shape_out = (naxis2, naxis1)
    # Empty image for later writing
    data = np.zeros((naxis2, naxis1), dtype=float)
    hdu = fits.PrimaryHDU(data=data, header=hdr_out)
    hdu.writeto(object+'_image/'+name_out, overwrite=True)
    return w, shape_out


def stacking_wcs(df, indexes, template_out, # w_out, shape_out,
                sig_clip = 3, add_name = '',
                rem_sky = True, norm = False,
                print_tests = False):
    print('Aligning %i images'%(len(indexes)))
    cam_dict = {'QHY411-1': {'rdnoise': 1.8, 'gain': 0.41, 'scale': 0.141},
                'QHY411-2': {'rdnoise': 1.81, 'gain': 0.41, 'scale': 0.142},
                'QHY411-3': {'rdnoise': 1.81, 'gain': 0.41, 'scale': 0.598},
                'QHY600-3': {'rdnoise': 1.68, 'gain': 0.33, 'scale': 0.192},
                'QHY600-4': {'rdnoise': 1.68, 'gain': 0.33, 'scale': 0.141},
                'iKon936-1': {'rdnoise': 6.2, 'gain': 1, 'scale': 0.23}}

    fil = df.iloc[0]['filter']
    object = df.iloc[0]['object']
    #int_total = 0
    with fits.open(object+'_image/'+template_out) as hdul:
        hdu = hdul[0]
        heads = hdu.header # type: ignore
        w_out = WCS(heads)
        shape_out = (heads["NAXIS1"], heads["NAXIS2"])
        pixscale = heads["SCALE"]
    # Build input list for reproject: (array, WCS) pairs
    cube = np.empty((len(indexes),) + shape_out, dtype=float)
    weights = np.empty((len(indexes)))
    for i, fn in enumerate(df.iloc[indexes]['path']):
        try:
            with fits.open(fn) as hdul:
                if print_tests: print('opening')
                hdu = hdul[0]
                head = hdu.header # type: ignore
                data = hdu.data.astype(float) # type: ignore
                wcs = WCS(hdu.header) # type: ignore
            
            if print_tests: print('reprojecting')
            data_r, _ = reproject_interp((data, wcs), w_out, shape_out=shape_out)
            # substract sky background
            if print_tests: print('sigma clip stats')
            sky_mean, _, sky_std = sigma_clipped_stats(data_r, sigma=3.0, maxiters=5, cenfunc=np.mean)
            if rem_sky:
                data_r = data_r - sky_mean #head["FLUXSKY"]
            # now data is in ADU without sky
            # convert to electrons
            if print_tests: print('converting electrons')
            cam_i_dict = cam_dict[df.iloc[indexes[i]]['camera']]
            data_r = data_r / cam_i_dict['gain']
            # calculate weights
            if print_tests: print('calculating weights')
            weights[i] = 1.0 / ((sky_std/cam_i_dict['gain'])**2 + (cam_i_dict['rdnoise'] / df.iloc[indexes[i]]['integration'])**2)
            # normalize by integration time
            if print_tests: print('normalizing by integration time')
            data_r = data_r / df.iloc[indexes[i]]['integration']
            # normalize counts
            if norm:
                data_r = data_r / np.max(data_r)
                
            cube[i] = data_r
            #int_total += df.iloc[indexes[i]]['integration']
            if i == len(indexes)//4: print('  25% done')
            if i == len(indexes)//2: print('  50% done')
            if i == len(indexes)//(3/4): print('  75% done')
        except Exception as error: print('%s in image %i'%(type(error).__name__, i))
    # stacking
    if print_tests: print('Creating CCD list')
    # Create CCDData list (no disk I/O)
    ccds = []
    for i, (data, w) in enumerate(zip(cube, weights)):
        ccd = CCDData(data, unit='electron/s')  # No header/WCS needed for combine
        ccd.header['STACKWT'] = w  
        ccds.append(ccd)
    print('Stacking images')
    """clip = sigma_clip(cube, sigma=sig_clip, axis=0, masked=True)
    stack_sigclip = np.ma.mean(clip, axis=0).filled(np.nan) # type: ignore"""
    # weighted average with sigma clipping
    stack = ccdp.combine(
                            ccds,
                            method='average',  # Weighted after clip
                            weights=weights,
                            sigma_clip=True,
                            sigma_clip_low_thresh=sig_clip,
                            sigma_clip_high_thresh=sig_clip,
                            sigma_clip_func=np.ma.median,
                            sigma_clip_dev_func=mad_std    # Robust std (import from astropy.stats)
                        )
    stack_sigclip = stack.data
    # Convert WCS to FITS header
    hdr = w_out.to_header()
    # Saving basic header
    hdr['NAXIS']  = 2
    hdr['NAXIS1'] = stack_sigclip.shape[1]
    hdr['NAXIS2'] = stack_sigclip.shape[0]
    hdr['SCALE'] = pixscale
    hdr['FLUXSKY'] = np.nanmean(stack_sigclip)
    # adding stacking parameters to header
    hdr['n_im'] = (len(df), 'Total number of images stacked')
    hdr['total_t'] = (df['integration'].sum(), 'Total integration time stacked')
    hdr['avg_see'] = (df['seeing'].mean(), 'Average seeing of stacked images')
    hdr['std_see'] = (df['seeing'].std(), 'Standard deviation of seeing')
    hdr['filter'] = fil
    # Create HDU and write to disk
    hdu = fits.PrimaryHDU(data=stack_sigclip, header=hdr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(object+'_image/'+df.iloc[indexes[0]]['filter']+add_name+".fits", overwrite=True)
    