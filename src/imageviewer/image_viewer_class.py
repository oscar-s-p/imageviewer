'''
Image viewer class
'''

import os, sys, glob
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import ceil

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
# from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.visualization import simple_norm, make_lupton_rgb #, SimpleNorm
from astropy.visualization.wcsaxes import add_scalebar, SphericalCircle
from astropy.coordinates import Angle, SkyCoord, get_body, EarthLocation
from astropy.nddata import Cutout2D, NDData
from astropy.time import Time
from astropy.stats import sigma_clip

from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_interp




class image_viewer:
    def __init__(self, directory: str = '',
               list_available = False,
               folder_list = [],
               filters = False,
               previous_df = False,
               print_error = True,
               bad_format = False,
               im_type_input = False):
        """Class to quickly open FITS images. Searches in given directory.
        
        Attributes
        ---------
        directory : str
            Directory where images are stored. If none given look in current working directory.

        list_available : bool : False
            Wether to print the resulting dataframe of found images or not

        folder_list : optional, list of str
            Extra directories to inspect for images and save their folder path from working directory
        
        previous_df : optional, pd.DataFrame or str
            Previous df with files to be added to the new one of the files found in ``folder_list``

        print_error: bool, optional
            If False, no error warnings will be printed (good for reading large datasets)
        
        Methods
        --------
        return_index()
            Returns the image path and index in the datafile given one or the other.
        
        dataframe_add()
            Add columns from FITS files to the dataset
        
        image_finder()
            Search for images by object, date, filter
        
        header_info()
            Method to view general header info.

        view_image()
            Method to view single or multiple FITS images
        
        view_RGB()
            Method to view RGB composite images
        
        data_manipulation()
            Manipulate image data (used by view_image and view_RGB)
        
        plotting()
            Advanced plotting with WCS coordinates (used by view_image and view_RGB)
        
        read_data()
            Read image data and headers
        
        get_moon_distance()
            Calculate angular distance to moon
        """
        self.im_type_dir = {'LB': 
                                {'data_i' : 0,
                                'header_i' : 0,
                                'header_WCS' : 0},
                            'HST':
                                {'data_i' : 1,
                                'header_i' : 0,
                                'header_WCS' : 1,
                                'detector' : 
                                    {'UVIS' : 
                                        {'SCALE' : 0.04},
                                    'IR' : 
                                        {'SCALE' : 0.13}}},
                            'Keck II':
                                {'data_i' : 0,
                                 'header_i' : 0,
                                 'header_WCS' : 0,
                                 'detector' : 
                                    {'Aladdin 3'}},
                            'bad':
                                {'data_i' : 0,
                                 'header_i' : 0,
                                 'header_WCS' : 0}
                                }
        self.bad_format = bad_format
        
        self.folder_list = folder_list
        print('Current working directory: ' + os.getcwd())
        if directory=='':
            directory = os.getcwd()
        if directory != os.getcwd():
            self.dir_img = os.path.join(os.getcwd(),directory)
        else: self.dir_img = directory
        print('Image directory defined: ' + self.dir_img)

        # list of images in dir_img and where were they
        files = list(Path(self.dir_img).glob('*.fits'))
        folder_found = ['']*len(files)
        # list of images in the different folders of folder_list and the corresponding folder
        if folder_list!= []:
            for fl in folder_list:
                fi = list(Path(os.path.join(self.dir_img, fl)).glob('*.fits'))
                files=files+fi
                folder_found =folder_found+[fl]*len(fi)

        files_data = []
        # creation of data dictionary
        # idea maybe future: read more info from headers: coordinates...
        for k, f in enumerate(files):
            try:
                name = f.name
                path = str(f.resolve())
                im_type = 'LB'
                try: # reading files from TTT
                    telescope, camera, date_time, object, filter = name.split('_')
                    filter = filter[:-5]
                    date_time = pd.to_datetime(date_time, format='%Y-%m-%d-%H-%M-%S-%f')
                except: 
                        if print_error: print('ERROR WITH FILENAME FORMAT CONVENTION EXPECTED: ', f)
                    # try: # reading files from HST
                        with fits.open(os.path.join(path)) as hdul: # type: ignore
                            heads = hdul[0].header # type: ignore
                            hdul.close()
                        telescope, camera, date_time, object, filter = heads['TELESCOP'], heads['INSTRUME'], heads['DATE-OBS']+' '+heads['TIME-OBS'], heads['TARGNAME'], heads['FILTER']
                        try: date_time = pd.to_datetime(date_time, format='%Y-%m-%d %H:%M:%S') # type: ignore
                        except: date_time = pd.to_datetime(date_time, format='%Y-%m-%d %H:%M:%S.%f') # type: ignore
                        im_type = telescope
                    # except:
                        # print('TTT and HST file format error...')
                        # telescope, camera, date_time, object, filter = None, None, None, None, None
                add = True
                if type(filters)==dict:
                    print('Filtering found files with: %s'%filters)
                    for ky in filters.keys(): # type: ignore
                        if type(filters[ky]) != list:
                            if filters[ky] != locals()[ky]:
                                add = add*False
                        else:
                            if locals()[ky] not in filters[ky]:
                                add = add*False
                        #object == filters['object'] or []:
                if add == True:
                    date = date_time.date()
                    if date_time.hour < 12 : date = date + dt.timedelta(days=-1)
                    date = str(date)
                    size_MB = f.stat().st_size / 1e6
                    created = pd.to_datetime(f.stat().st_ctime, unit="s")
                    files_data.append({"filename": name, "path": path, "telescope": telescope, 'camera': camera,
                                    "object": object, "filter": filter, "size_MB": size_MB,
                                    "date_time": date_time, "date": date,
                                    "folder_found": folder_found[k],
                                    "im_type" : im_type})
            except: 
                if print_error: print('Error with file: %s'%f)
                if bad_format:
                    name = f.name
                    path = str(f.resolve())
                    files_data.append({"filename": name, "path": path, "im_type" : 'bad'})
                
        if len(files_data)==0:
            print('WARNING: NO IMAGE FILES FOUND')
            #return
        # creation of dataframe
        if len(files_data)!=0:
            df_files = pd.DataFrame(files_data).sort_values("filename").reset_index(drop=True)
        # Addition of previous dataframe
        if type(previous_df) != bool:
            if type(previous_df) != pd.DataFrame:
                if type(previous_df) == str:
                    if previous_df[-3:] == 'pkl': previous_df = pd.read_pickle(previous_df)
                    elif previous_df[-3:] == 'csv' : previous_df = pd.read_csv(previous_df)
                    else: 
                        print('ERROR: unrecognized DataFrame format. Use \'.pkl\' or \'.csv\'.')
                        return
            if filters!=False and type(filters)==dict:
                print('Filtering previous dataframe with: %s'%filters)
                print(' - Initial length of previous df: ', len(previous_df))
                for k in filters.keys(): # type: ignore
                    if type(filters[k])!= list:
                        previous_df = previous_df[previous_df[k] == filters[k]]
                    else:
                        previous_df = previous_df[previous_df[k].isin(filters[k])]
                print(' - Final length of previous df: ', len(previous_df))
            if len(files_data)!=0:
                self.df_files = pd.concat([df_files, previous_df], ignore_index = True).drop_duplicates(subset = 'filename', keep= 'last')
            else: self.df_files = previous_df
        else: self.df_files = df_files
        # if im_type not previously defined in dataset
        if 'im_type' not in self.df_files.columns and im_type_input == False:
            print('ERROR, `im_type` not defined in dataset. Give it as an input parameter.')
        if im_type_input!=False:
            self.df_files['im_type'] = im_type_input
        # print available images if requested
        if list_available:
            print(self.df_files)
        print('Total number of images found: ', len(self.df_files))

        # Store gravitational lens objects
        grav_lens = ['QSO0957+561', 'Q2237+030', 'MG1654+1346', 'SDSSJ1004+4112', 'LBQS1333+0113', 'SDSSJ0819+5356',
             'EinsteinCross', 'DESI-350.3458-03.5082', 'ZTF25abnjznp']
        # EinsteinCross and Q2237+030 are the same object (?)
        grav_lens_ra = ['10 01 20.692 h', '22 40 30.234 h', '16 54 41.796 h', '10 04 34.936 h', '13:35:34.8 h', '08 19 59.764 h',
                        '22 40 30.271 h', '350.3458d', '07:16:34.5h']
        grav_lens_dec = ['+55 53 55.59 d', '+03 21 30.63 d', '+13 46 21.34 d', '+41 12 42.66 d', '+01 18 05.5 d', '+53 56 24.63 d',
                         '+03 21 31.03 d', '-03.5082d', '+38:21:08d']
        grav_data = []
        for i in range(len(grav_lens)):
            grav_data.append({
                'object' : grav_lens[i],
                'ra' : Angle(grav_lens_ra[i]),
                'dec' : Angle(grav_lens_dec[i])
                })
        self.df_grav_lens = pd.DataFrame(grav_data).sort_values('object').reset_index(drop=True)
        # filters dictionary :
        self.dict_filters = {
            'Ha' : 'Ha', 'Halpha' : 'Ha', 'H-alpha' : 'Ha', 'H' : 'Ha',
            'Lum' : 'Lum', 'L' : 'Lum',
            'OIII' : 'OIII', 'O3' : 'OIII', 'O' : 'OIII',
            'SDSSg' : 'SDSSg', 'g' : 'SDSSg',
            'SDSSr' : 'SDSSr', 'r' : 'SDSSr',
            'SDSSi' : 'SDSSi', 'i' : 'SDSSi',
            'SDSSu' : 'SDSSu', 'u' : 'SDSSu',
            'SDSSzs' : 'SDSSzs', 'z' : 'SDSSzs',
            'iz' : 'iz'
            }
        
        # print('Datafiles available:')
        # print('- df_files')
        # print('- df_grav_lens')

    
    def return_index(self, image, iloc = False):
        """
        Returns the image path and index in the datafile given one or the other.

        Parameters
        ----------
        image: int / str
            int - image index in datafile \n
            str - image path
        """
        if type(image)==int:
            if iloc: image_str = self.df_files.iloc[image].filename
            else: image_str = self.df_files.loc[image].filename
            image_int = image
        else: 
            image_str = image
            try: image_int = self.df_files.index[self.df_files['filename']==image].to_list()[0]
            except:
                try: 
                    image_int = self.df_files.index[self.df_files['path']==image].to_list()[0]
                    image_str = self.df_files.loc[image_int]['filename'] # type: ignore
                except:
                    print('\n ERROR: FILENAME NOT FOUND')
                    return
        if self.bad_format==False:
            if self.folder_list != False:
                if iloc: folder_name = self.df_files.iloc[image_int].folder_found
                else: folder_name = self.df_files.loc[image_int].folder_found
                image_str = os.path.join(folder_name, image_str) # type: ignore
        return image_str, image_int
    

    def image_finder(self, object, 
                     date = None, 
                     filter = None,
                     return_df = False,
                     printeo = False
                     ):
        """
        Method to identify the fits file that match an observation object, date and filter.
        
        Parameters
        ----------
        object : index / str
            Either the iloc or string to the object in self.df_grav_lens

        date : 'YYYY-MM-DD' (optional hh-mm-ss)
            If no date is supplied, return possible options

        filter : str
            Desired filter. If None, return possible options
        """
        try:
            if type(object) == str:
                if object in self.df_grav_lens['object']:
                    obj_int = self.df_grav_lens.index[self.df_grav_lens['object'] == object].tolist()
                    obj_str = object
                elif object in self.df_files['object'].unique(): 
                    obj_str = object
                else:
                    print('ERROR: OBJECT NAME NOT REGISTERED.\n  Try with one of: ', self.df_grav_lens['object'].tolist(), ' in the saved gravitational lens objects.')
                    print('Or see the objects found in the folder:')
                    print(self.df_files['object'].unique())
                    return
            elif type(object) in [int, np.int16, np.int32, np.int64, np.int8]:
                obj_str = self.df_grav_lens['object'].iloc[object]
                obj_int = object  
            else: print('Type of ``object`` (',type(object),') not recognized') 
        except: 
            print('ERROR: No previously known object was found.\n  Try with one of: ', self.df_grav_lens['object'].tolist())
        
        df_filtered = self.df_files[self.df_files["object"]==obj_str].copy()

        if date == None:
            print('Available date observations:')
            print(df_filtered.groupby(['object', 'date']).size())
        
        else: 
            if type(date) == str:
                df_filtered = df_filtered[df_filtered['date'] == date]
            if type(date) == list:
                df_filtered = df_filtered[df_filtered['date'] in date]

        if filter != None:
            df_filtered = df_filtered[df_filtered['filter'] == self.dict_filters[filter]]
        
        if printeo:
            print(df_filtered[['object', 'filter', 'date_time']])
        
        if return_df == True:
            print('Matching index: ')
            print(df_filtered.index.tolist())
            return df_filtered
        else:
            return df_filtered.index.tolist()
        


    def dataframe_add(self, label):
        """Method to add columns to existing `self.df_files`.
        
        Parameters
        ----------
        label : str / list(str)
            Can be `seeing`, `moon`, `integration`, `EZP`...
        """
        if type(label)==str: label = [label]
        # function to get seeing
        def get_seeing_from_filename(fname):
            with fits.open(fname) as hdul: # type: ignore
                heads = hdul[0].header # type: ignore
                hdul.close()
            try: s = float(heads['FWHM'])*float(heads['SCALE'])
            except: s = np.nan
            return s
        # function to get moon distance
        def get_moon_from_filename(fname):
            try: moon = self.get_moon_distance(fname).deg
            except: moon = np.nan
            return moon
        # 2 function to get moon distance
        def get_moon_2_from_filename(fname):
            with fits.open(fname) as hdul: # type: ignore
                heads = hdul[0].header # type: ignore
                hdul.close()
            try: m2 = heads['MOONDIST']
            except: m2 = np.nan
            return m2
        # function to get integration time
        def get_integration_from_filename(fname):
            with fits.open(fname) as hdul: # type: ignore
                heads = hdul[0].header # type: ignore
                hdul.close()
            try: t = float(heads['INTEGT'])
            except: t = np.nan
            return t
        # function to get EZP
        def get_EZP_from_filename(fname):
            with fits.open(fname) as hdul: # type: ignore
                heads = hdul[0].header # type: ignore
                hdul.close()
            try: ezp = float(heads['EZP'])
            except: ezp = np.nan
            return ezp
        # function to get other keyword from header
        def get_kw_from_filename(fname, kw = False):
            with fits.open(fname) as hdul: # type: ignore
                heads = hdul[0].header # type: ignore
                hdul.close()
            try: kw_value = float(heads[kw])
            except: kw_value = np.nan
            return kw_value
        # function to get date column
        def get_date_from_datetime(d_time):
            date = d_time.date()
            if d_time.hour < 12 : date = date + dt.timedelta(days=-1)
            #date = str(date)
            return date
        #for file in self.df_files['filename']:
        if 'seeing' in label:
            self.df_files["seeing"] = self.df_files["path"].map(get_seeing_from_filename)
        if 'moon' in label:
            self.df_files["moon"] = self.df_files["path"].map(get_moon_from_filename)
        if 'moon2' in label:
            self.df_files["moon2"] = self.df_files["path"].map(get_moon_2_from_filename)
        if 'integration' in label:
            self.df_files["integration"] = self.df_files["path"].map(get_integration_from_filename)
        if 'EZP' in label:
            self.df_files["EZP"] = self.df_files["path"].map(get_EZP_from_filename)
        if 'date' in label:
            self.df_files["date"] = self.df_files["date_time"].map(get_date_from_datetime)
        for lab in label:
            if lab not in ['seeing', 'moon', 'moon2', 'integration', 'EZP', 'date']:
                self.df_files[lab] = self.df_files["path"].map(lambda x: get_kw_from_filename(x, kw = lab)) # type: ignore


    def header_info(self, image,
                    interesting_keys = ['INSTRUME', 'OBJECT', 'FILTER', 'INTEGT', 'DATE-OBS',
                                        'RA', 'DEC', 'NAXIS1', 'NAXIS2', 'SCALE', 'FOVX', 'FOVY',
                                        'CCW', 'CRPIX1', 'CRPIX2', 'FWHM'],
                    hdul_i = 0,
                    iloc = False
                                        ):
        """Method to view general header info.
        
        Parameters
        ----------
        image : int / str
            int - index of desired file in dataframe \n
            string - path to desired fits file
            
        interesting_keys: list / 'all'
            list - list of strings with header keyword \n
            'all' - will print the whole header
        """
        image_str, image_int = self.return_index(image, iloc = iloc) # type: ignore
        
        # Extracting data from header
        with fits.open(os.path.join(self.dir_img, image_str)) as hdul: # type: ignore
            heads = hdul[hdul_i].header # type: ignore
            hdul.close()
        # printing basic header info
        print('Image: %s'%image_str)
        print('\n   --- HEADER DATA ---')
        if type(interesting_keys) == str and interesting_keys!='all':
                interesting_keys = [interesting_keys]
        try:
            if type(interesting_keys) == str and interesting_keys=='all':
                print(repr(heads))
            else:
                for k in interesting_keys:
                    if heads.comments[k]!='':
                        print(k, ' = ', heads[k], '  ---  ', heads.comments[k])
                    else:
                        print(k, ' = ', heads[k])
        except:
            print('WARNING: WRONG interesting_keys PARAMETER.')
            print('         Header parameter not recognized. Try the string \'all\' to view the full header')


    def view_RGB(self, object, date,
                 name_list = False, # use to set directly the files for R G and B channels 
                 filters = 'irg',
                 object_coordinates = None,
                 figsize = (14,10),
                 manipulation_kw = {
                       'centered' : True,
                       'zoom' : False
                       },
                 plotting_kw = {
                        'scalebar_arcsec' : 5,
                        'scalebar_frame' : False,
                        'add_circle' : None
                        },
                 RGB_kw = {
                        'stretch' : 5,
                        'Q' : 8,
                        'minimum' : None
                        },
                 RGB_norm_kw = {
                        'vmax' : None,
                        'max_sky' : False
                        }):
        """
        Method to view RGB images:
        
        -------
        Parameters:

        object : str or int or list of str or ints
            ``str`` with name of object or 
            ``int`` with object index in ``self.df_grav_lenes``

        date : str 

        filters : str
            g, r, i, z or other keywords of ``self.dict_filters`` in the desired order for RGB.
            Can be in a single string with the 1 letter abrevietion or a list of strings with the filter keyword (or full name)

        ... to be continued
        """
        # Object coordinates extraction
        if type(object) == str:
            try: 
                obj_int = self.df_grav_lens.index[self.df_grav_lens['object'] == object][0]
            except: 
                if object_coordinates == None:
                    if object not in self.df_files['object']: print('ERROR: Object not known')
                    else: print('ERROR: Object observed, coordinates are required')
                    return
                else:
                    new_object = {'object' : object, 'ra' : Angle(object_coordinates[0]), 'dec' : Angle(object_coordinates[1])}
                    self.df_grav_lens.loc[len(self.df_grav_lens)] = new_object # type: ignore
                    obj_int = self.df_grav_lens.index[self.df_grav_lens['object'] == object][0]
        else: obj_int = object
        if object_coordinates == None:
            obj_coords = (self.df_grav_lens['ra'].loc[obj_int],
                          self.df_grav_lens['dec'].loc[obj_int])
        else: obj_coords = object_coordinates

        if type(name_list) == bool:
            print('RGB image of object: ', self.df_grav_lens.loc[obj_int].object, ' taken the night of ', date)
        colors = ['R', 'G', 'B']

        fig, axes = plt.subplots(figsize = figsize)
        self.nr_nc = (1,1)

        # Loop over each image to obtain file path and header
        wcs_list = []
        data_list = []
        d_w_list = []
        headers = []
        print('Cutting out images...')
        for i in range(3):
            if type(name_list)== bool:
                print('    - ',colors[i],': ', self.dict_filters[filters[i]])
                filt_i = self.image_finder(obj_int, date = date, filter = self.dict_filters[filters[i]])
            else: filt_i = [i]
            if len(filt_i) > 1: # type: ignore
                print('More than one image matching with the object, date and filter specified.')
                for f in filt_i: print(self.df_files['filename'].loc[f]) # type: ignore
                print('Choosing first match for RGB plot.')
            if type(name_list)==bool:
                img_str = self.df_files['path'].loc[filt_i[0]] # type: ignore
            else: img_str = name_list[i]
            self.img_str = img_str
            self.img_int = self.df_files.index[self.df_files['path']==img_str].to_list()[0]
            # obtain headers for skyflux and data for title
            with fits.open(os.path.join(self.dir_img, img_str)) as hdul: # type: ignore
                headers.append(hdul[0].header) # type: ignore
                hdul.close()

            # Cutout
            cutout, _ = self.data_manipulation(img_str, **manipulation_kw) # type: ignore
            # data_list.append(cutout.data)
            # wcs_list.append(cutout.wcs)
            d_w_list.append((cutout.data, cutout.wcs))
                
        # Get optimal WCS and output shape covering all images (N up, E left)
        wcs_out, shape_out = find_optimal_celestial_wcs(d_w_list)
        # Reprojection of images to common wcs
        repr_data_list = []
        print('Rotating and aligning images...')
        for i, d_w in enumerate(d_w_list):
            repr_data, _ = reproject_interp(d_w, wcs_out, shape_out = shape_out)
            repr_data_list.append(repr_data)
        
        # Data normalization
        print('Data normalization...')
        for i, data in enumerate(repr_data_list):
            vmin = headers[i]['FLUXSKY']
            # min and max for manual norm, if max_sky is set, use it to obtain max as max_sky * sky_flux
            if 'vmax' in RGB_norm_kw.keys(): vmax = RGB_norm_kw['vmax']
            if 'max_sky' in RGB_norm_kw.keys():
                if RGB_norm_kw['max_sky'] != False: vmax = RGB_norm_kw['max_sky']*vmin
            if vmax == None: vmax = np.nanmax(repr_data)
            # manual normalization
            data_norm = (data - vmin)/(vmax-vmin)
            data_mask = data_norm < 1e-3
            data_norm[data_mask] = 1e-3
            repr_data_list[i] = data_norm

        # RGB creation
        print('Creating RGB image...')
        rgb_default = make_lupton_rgb(repr_data_list[0], repr_data_list[1], repr_data_list[2],
                                        **RGB_kw)
        if self.bad_format==False:
            title_str = (r'$\bf{Object}$: %s - $\bf{Telescope}$: %s - $\bf{Date-time est}$: %s''\n'
                        r'$\bf{Camera}$: %s - $\bf{RGB Filters}$: %s|%s|%s - $\bf{Seeings}$: %.1f|%.1f|%.1f$^{\prime\prime}$''\n'
                        r'$\bf{Integrations}$: %s|%s|%s s - $\bf{SNRs}$: %s|%s|%s -  $\bf{Moon D}$: %.1fº''\n'
                            %(self.df_files.iloc[filt_i[0]]['object'],  # type: ignore
                            self.df_files.iloc[filt_i[0]]['telescope'],  # type: ignore
                            self.df_files.iloc[filt_i[0]]['date_time'].strftime("%Y-%m-%d %H:%M"),  # type: ignore
                            self.df_files.iloc[filt_i[0]]['camera'],  # type: ignore
                            self.dict_filters[filters[0]], self.dict_filters[filters[1]], self.dict_filters[filters[2]],
                            (float(headers[0]['FWHM'])*float(headers[0]['SCALE'])),
                            (float(headers[1]['FWHM'])*float(headers[1]['SCALE'])),
                            (float(headers[2]['FWHM'])*float(headers[2]['SCALE'])),
                            headers[0]['INTEGT'], headers[1]['INTEGT'], headers[2]['INTEGT'],
                            headers[0]['OBJECSNR'], headers[1]['OBJECSNR'], headers[2]['OBJECSNR'],
                            self.get_moon_distance(filt_i[0]).deg)) # type: ignore
        else: title_str = 'RGB image'
        self.plotting(None, None, fig, axes, 0,
                        RGB = True, rgb_data = rgb_default,
                        rgb_wcs = wcs_out, title_str = title_str,  # type: ignore
                        **plotting_kw)



    def view_image(self, image, 
                    RGB = False,
                    nrows_ncols = None,
                    figsize = None,
                    save_name = None,
                    manipulation_kw = {
                       'centered' : True,
                       'zoom' : False,
                       'stretch' : 'linear',
                       'percentile' : None,
                       'vminmax' : (None, None),
                       'rotate': True
                       },
                    plotting_kw = {
                        'cmap' : 'gray',
                        'scalebar_arcsec' : 5,
                        'scalebar_frame' : False,
                        'add_circle' : None
                        },
                    RGB_kw = {
                        'stretch' : 5,
                        'Q' : 8,
                        'minimum' : None
                        },
                    RGB_norm_kw = {
                        'vmax' : None,
                        'max_percentile' : 99,
                        'max_sky' : False
                        },
                    iloc = False
                    ):
        """
        Method to view images. Takes dictionary keywords for ``data_manipulation`` and ``plotting``.
        """
        # Multiple images
        if type(image) == list and RGB == False:
            print('------\nViewing multiple images:')
            n_image = len(image)
            if nrows_ncols == None:
                if n_image <= 3: nrows_ncols = (1, n_image)
                else: nrows_ncols = (ceil(np.sqrt(n_image)), ceil(np.sqrt(n_image)))
            if n_image <= (nrows_ncols[0]-1)*(nrows_ncols[1]):
                nrows_ncols = (nrows_ncols[0]-1, nrows_ncols[1])
            image_list = image

        # Simple image Non RGB
        if type(image) != list:
            print('------\nViewing image:')
            n_image, nrows_ncols = 1, (1,1)
            image_list = [image]
        # RGB image
        if RGB == True: 
            n_image = 1
            colors = ['R', 'G', 'B']
            cutout_RGB = []
            print('------\nRGB color composite image:')
            if n_image == 1: nrows_ncols = (1,1)
            image_list = image

        self.nr_nc = nrows_ncols
        n_data = len(image_list)

        # if manipulation and plotting are dicts, use the same setup for all images
        if type(manipulation_kw) == dict:
            if 'rotate' not in manipulation_kw.keys(): manipulation_kw['rotate'] = True
            if manipulation_kw['rotate'] == False: 
                if type(plotting_kw)==dict: plotting_kw['arrows'] = False
                else: 
                    for i in range(len(plotting_kw)):
                        plotting_kw[i]['arrows'] = False
            manipulation_kw = [manipulation_kw]*n_data
        if type(plotting_kw) == dict: plotting_kw = [plotting_kw]*n_data

        fig, axes = plt.subplots(self.nr_nc[0], self.nr_nc[1], # type: ignore
                                 figsize = figsize)
        if n_image == 1: axes = [axes]
        axes = np.array(axes).reshape(-1)
        
        for i, (img, m_k, p_k) in enumerate(zip(image_list, manipulation_kw, plotting_kw)):
            self.img_str, self.img_int = self.return_index(img, iloc = iloc) # type: ignore
            if iloc: df_file = self.df_files.iloc[self.img_int]
            else: df_file = self.df_files.loc[self.img_int]
            cutout, norm = self.data_manipulation(self.img_str, iloc = iloc, **m_k) # type: ignore

            if RGB == False:
                if self.bad_format==False:
                    print('    Object: ', df_file['object'],
                        '  -  Filter: ', df_file['filter'])
                self.plotting(cutout, norm, fig, axes[i], i,
                              iloc = iloc,
                              **p_k)
            else:
                # Extracting data from header
                with fits.open(os.path.join(self.dir_img, self.img_str)) as hdul: # type: ignore
                    heads = hdul[0].header # type: ignore
                    hdul.close()
                if i==0: print('    Object: ', df_file['object'])
                print('    - ',colors[i],': ', df_file['filter'])
                # min and max for manual norm, if max_sky is set, use it to obtain max as max_sky * sky_flux
                vmin = heads['FLUXSKY']
                if 'vmax' in RGB_norm_kw.keys(): vmax = RGB_norm_kw['vmax']
                if RGB_norm_kw['max_sky'] != False: vmax = RGB_norm_kw['max_sky']*heads['FLUXSKY']
                if vmax == None: vmax = np.max(cutout.data) # type: ignore
                # manual normalization
                data = (cutout.data - vmin)/(vmax-vmin)
                data_mask = data < 1e-3
                data[data_mask] = 1e-3
                cutout_RGB.append(data)

                if i == len(image_list)-1:                        
                    rgb_default = make_lupton_rgb(cutout_RGB[0].data, cutout_RGB[1].data, cutout_RGB[2].data,
                                                  **RGB_kw)
                    self.plotting(cutout, norm, fig, axes[0],0,
                                  RGB = True, rgb_data = rgb_default,
                                  iloc = iloc,
                                  **plotting_kw[i])
        plt.tight_layout()
        if type(save_name) == str:
            plt.savefig(save_name, dpi=300)
        plt.show()

    def data_manipulation(self, image_str,
                          centered = True, 
                          zoom = False,
                          stretch = 'linear',
                          percentile = None,
                          vminmax = (None, None),
                          rotate = True,
                          iloc = False
                          ):
        """
        Method to prepare images for manipulation. It is internally called. Crops the image and sets visualization normalization and stretch.

        Parameters
        ---------
        image : int / string / list
            int - index of desired file in dataframe \n
            string - path to desired fits file \n

        centered : True or tuple, optional
            (x,y) - int for pix coordinates \n
            (RA, DEC) - wcs coordinates. Accepting both strings or angle values

        zoom : False or Value or Tuple, optional
            int / (int, int) - pixel size in x and y axis \n
            Angle / (Angle, Angle) - angular size in RA and DEC
        
        stretch : str, optional
            Image stretch to enhance detail visualization \n
            ``linear``, ``sqrt``, ``power``, ``log``, ``sinh``, ``asinh``
        
        percentile : int or tuple, optional
            ``int`` - Middle percentile of values to consider for normalization; 
            ``tuple`` - Lower and upper percentile of values to consider for normalization
        
        vminmax : tuple, optional
            Min and max pixel values for normalization. Overrides ``percentile``.
            If set as None, keeps the absolute min or max of image
        """
        

        # Extracting data from header
        if iloc: im_type = self.df_files.im_type.iloc[self.img_int]
        else: im_type = self.df_files.im_type.loc[self.img_int]

        with fits.open(os.path.join(self.dir_img, image_str)) as hdul:
            print(image_str)
            data = hdul[self.im_type_dir[im_type]['data_i']].data.astype(np.float32) # type: ignore
            heads = hdul[0].header # type: ignore
            heads_WCS = hdul[self.im_type_dir[im_type]['header_WCS']].header  # type: ignore
            if im_type == 'Keck II':
                hdr = hdul[0].header # type: ignore
                # Fix CD matrix elements: convert strings to floats
                for key in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
                    if key in hdr:
                        hdr[key] = float(hdr[key])
                # Replace deprecated RADECSYS by RADESYSa
                if 'RADECSYS' in hdr:
                    hdr['RADESYSa'] = hdr['RADECSYS'].strip()
                    hdr['RADECSYS']
                heads_WCS = hdr

            wcs = WCS(heads_WCS)
            hdul.close()
        
        # obtaining central px coordinates
        x_shape = data.shape[1]
        y_shape = data.shape[0]
        if centered == True:
            center_px = (x_shape//2, y_shape//2)
        if type(centered)==tuple:
            if type(centered[0]) == int: # input in px units
                center_px = tuple(centered)
            elif type(centered[0]) == str: # input in str to be converted to deg
                center_angle = SkyCoord(centered[0], centered[1], frame = 'icrs')
                center_px = skycoord_to_pixel(center_angle, wcs, origin=0)
            else:
                center_angle = SkyCoord(centered[0], centered[1], frame = 'icrs')
                center_px = skycoord_to_pixel(center_angle, wcs, origin=0)
        
        # setting zoom
        if zoom == False:
            zoom = (x_shape, y_shape)
            size = (x_shape, y_shape)
        if im_type == 'LB' or im_type=='bad':
            scale = float(heads['SCALE'])
        elif im_type == 'HST':
            scale = float(self.im_type_dir[im_type]['detector'][heads['DETECTOR']]['SCALE'])
        elif im_type == 'Keck II':
            scale = float(heads['PIXSCALE'])
        else:
            print('No pixel scale defined, check image type.')
            return
        if type(zoom) == str:
            zoom = Angle(zoom)
            size = (zoom.deg / scale * 3600, zoom.deg / scale * 3600)  # type: ignore
        if type(zoom)== tuple:
            if type(zoom[0]) == str:
                zoom = (Angle(zoom[0]), Angle(zoom[1]))
                size = (zoom[0].deg / scale * 3600,zoom[1].deg / scale * 3600)  # type: ignore
        if type(zoom)==tuple:
            zoom = zoom[::-1]
            size = size[::-1]

        # slicing image
        try:
            cutout = Cutout2D(data, position = center_px, size = size, #zoom,
                              wcs = wcs, mode = 'partial')
        except:
            print('\n --- \nERROR: the cutout region is outside of the image.')
            return
        
        # rotating image
        if rotate:
            wcs_out, shape_out = find_optimal_celestial_wcs((cutout.data, cutout.wcs))
            data_oriented, _ = reproject_interp((cutout.data, cutout.wcs), wcs_out, shape_out=shape_out)
            cutout = NDData(data_oriented, wcs = wcs_out)
        
        # norm definition
        if type(percentile) == int or percentile == None:
            percentile_minmax = (None, None)
        if type(percentile) == tuple:
            percentile_minmax = percentile
            percentile = None
        if stretch not in {'linear', 'sqrt', 'power', 'log', 'sinh', 'asinh'}:
            print('ERROR: Stretch should be one of \'linear\', \'sqrt\', \'power\', \'log\', \'sinh\', \'asinh\'')
            plt.close()
            return
        norm = simple_norm(cutout.data, stretch = stretch, 
                           vmin = vminmax[0], vmax = vminmax[1],
                           percent = percentile,
                           min_percent = percentile_minmax[0],
                           max_percent = percentile_minmax[1])
        
        return cutout, norm
        
    def plotting(self,
                 cutout, norm, fig, ax, ax_i,
                cmap = 'gray',
                scalebar_arcsec = 5, scalebar_frame = False,
                add_circle = None,
                RGB = False,
                rgb_data = False, rgb_wcs = False, title_str = False,
                arrows = True,
                iloc = False
                ):
        """
        Method to plot images, obtains edited data from ``self.data_manipulation()``.

        Parameters
        ---------
        cutout : Cutout2D
            Selected cutout object from ``data_manipulation``

        norm : Norm
            Selected norm from ``data_manipulation``

        cmap : str, optional
            Select the desired colormap for the image

        scalebar_arcsec : int, optional
            Angular size of scalebar in arcsec units
        
        scalebar_frame : bool, optional
            Add frame or not

        add_circle : dict, list of dicts or None, optional
            Parameters to plot a circle overlay. If None, no circle is plotted. If multiple circles are desired, enter a list of dicts.\n
            Expected keys: \n
                'center' : tuple 
                    (RA, DEC) coordinates as astropy Angle or SkyCoord
                'size' : astropy.units.Quantity
                    Angular size (e.g., astropy Angle with units).
                'color' : str, optional
                    Circle edge color.
                'label' : str, optional
                    Label for the circle to use in legend.
            
        fig_kwrds : None or dict, optional
            Dict with all the keywords desired to insert in ``plt.subplots()``

        figure : None or dict ..... tuple or axis
            Dict used by view_multiple method. Expected keys: \n
                'is_simple' : bool
                'create_fig' : bool
                    True or False
                'figsize' : tuple
                    Looked at if ``create_fig = True``
                'nrows_ncols' : tuple
                    Looked at if ``create_fig = True``
                'fig' : plt.figure object
                    Looked at if ``create_fig = False``
                'ax' : plt.axis object
                    Looked at if ``create_fig = False``
                'im_i' : int
                    Subplot index (image index). Looked at if ``create_fig = False``

            None - creates normal figure, does not return nothing \n
            tuple (int, int) - creates figure with specified conditions. Returns (fig, ax) \n
            tuple (ax, int, int) - plots image in specified ax[int,int]
        """
        if RGB == False: 
            with fits.open(os.path.join(self.dir_img, self.img_str)) as hdul: # type: ignore
                heads = hdul[0].header # type: ignore
                hdul.close()
            wcs = cutout.wcs
        else: 
            wcs = rgb_wcs
        ax.remove()
        ax = fig.add_subplot(self.nr_nc[0], self.nr_nc[1], ax_i+1, projection = wcs) # type: ignore
        if RGB == False:
        # colorbar
            im = ax.imshow(cutout.data,
                            norm = norm, origin = 'lower',
                            cmap = cmap)
            divider = make_axes_locatable(ax) # ensure colorbar height equal to inner axis of plot
            cax = divider.append_axes("right", size = "5%", pad = "7%")
            cbar = fig.colorbar(im, cax = cax)
            # hide the entire x-axis (ticks, ticklabels, spine)
            cbar.ax.tick_params(axis='x', which='both', 
                    labelbottom=False, labeltop=False,
                    bottom=False, top=False, length=0)
            # ticks on the right
            # cbar.ax.set_ylabel('ADU')

            cbar.ax.tick_params(axis='y', which='both', 
                    labelright=True, labelleft=False,
                    right=True, left=False, length=3,
                    labelsize = 10)
            # cbar.ax.set_ylabel('ADU', loc = 'lower', labelpad=-5.2)#,  **{'rotation':90})

            cbar.ax.yaxis.set_ticks_position('right')
            cbar.ax.yaxis.set_label_position('left')

            # 4) create the label (use cbar.set_label so colorbar internals are consistent)
            cbar.set_label("ADU")   # do not pass rotation here

            # 5) now get the Text object and force rotation + coords
            lbl = cbar.ax.yaxis.get_label()   # returns the Text instance
            lbl.set_rotation(90)               # rotate text (90 or 270 as you like)
            lbl.set_va('center')
            lbl.set_ha('center')

            # move the label into the gap between image and colorbar
            # x < 0 moves left; y=0.5 centers vertically
            cbar.ax.yaxis.set_label_coords(-0.45, 0.5)  # tune -0.45 to taste

            # 6) If you use tight_layout or constrained_layout, run these BEFORE step 5,
            #    or run a final draw after everything so the Text object is not overwritten:
            fig.canvas.draw()  


        else:
            ax.imshow(rgb_data, origin = 'lower')

        # Scale bar choosing color depending on luminance of cmap
        scalebar_angle = scalebar_arcsec/3600*u.deg # type: ignore
        rgba = plt.get_cmap(cmap)(0.0)
        luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
        scalebar_color = 'white' if (luminance < 0.5 and scalebar_frame == False) else 'black'
        add_scalebar(ax, scalebar_angle, label="%s arcsec"%str(scalebar_arcsec), 
                     color=scalebar_color, frame=scalebar_frame,
                     corner = 'bottom left')
        # Axis and title
        ax.set(xlabel='RA', ylabel='DEC')
        ax.coords.grid(color='gray', alpha=1, linestyle='solid')
        if iloc: df_file_loc = self.df_files.iloc[self.img_int]
        else: df_file_loc = self.df_files.loc[self.img_int]
        if title_str == False and df_file_loc['im_type']=='LB': #  # type: ignore -- if not false, was created by view_RGB method
            title_str = (r'$\bf{Object}$: %s - $\bf{Telescope}$: %s - $\bf{Seeing}$: %.1f$^{\prime\prime}$''\n'
                        r'$\bf{Camera}$: %s - $\bf{Filter}$: %s - $\bf{Integration}$: %s s''\n'
                        r'$\bf{SNR}$: %s - $\bf{Date time}$: %s - $\bf{Moon D}$: %.1fº'
                        %(df_file_loc['object'],
                        df_file_loc['telescope'],
                        (float(heads['FWHM'])*float(heads['SCALE'])),
                        df_file_loc['camera'],
                        df_file_loc['filter'],
                        heads['INTEGT'], heads['OBJECSNR'],
                        df_file_loc['date_time'].strftime("%Y-%m-%d %H:%M"),
                        self.get_moon_distance(self.img_int, iloc = iloc).deg))
        if title_str == False and df_file_loc['im_type']=='HST':  # type: ignore
            title_str = (r'$\bf{Object}$: %s - $\bf{Telescope}$: %s''\n'
                        r'$\bf{Camera}$: %s - $\bf{Filter}$: %s - $\bf{Integration}$: %s s''\n'
                        r'$\bf{Date time}$: %s - $\bf{Moon D}$: %.1fº - $\bf{Sun D}$: %.1fº'
                        %(df_file_loc['object'],
                        df_file_loc['telescope'],
                        df_file_loc['camera'],
                        df_file_loc['filter'],
                        heads['EXPTIME'],
                        df_file_loc['date_time'].strftime("%Y-%m-%d %H:%M"),
                        heads['MOONANGL'], heads['SUNANGLE']))
        if title_str == False and df_file_loc['im_type']=='Keck II': # type: ignore
            title_str = (r'$\bf{Object}$: %s - $\bf{Telescope}$: %s''\n'
                        r'$\bf{Camera}$: %s - $\bf{Filter}$: %s - $\bf{Integration}$: %s s''\n'
                        r'$\bf{Date time}$: %s'
                        %(df_file_loc['object'],
                        df_file_loc['telescope'],
                        df_file_loc['camera'],
                        df_file_loc['filter'],
                        heads['EXPOSURE'],
                        df_file_loc['date_time'].strftime("%Y-%m-%d %H:%M")))
        ax.set_title(title_str)
        ax.minorticks_on()

        # Optional plot of circles
        if add_circle is not None:
            if type(add_circle) != list:
                add_circle = [add_circle]
            for d_circle in add_circle:
                center = d_circle.get('center')
                size = d_circle.get('size')
                if 'color' not in d_circle: color = 'white'
                else: color = d_circle.get('color')
                label = d_circle.get('label')
                c = SphericalCircle((Angle(center[0]), Angle(center[1])),
                                    Angle(size),
                                    edgecolor = color,
                                    facecolor = 'none',
                                    transform = ax.get_transform('icrs'))
                ax.add_patch(c)
        
        # optional plot of arrows
        if arrows:
            arrs = AnchoredDirectionArrows(ax.transAxes, 'E', 'N', loc='lower right',
                                           pad = 0, color=scalebar_color, frameon=False,
                                           length=-0.1, aspect_ratio=-1,
                                           back_length=0, tail_width=0.2,  # type: ignore
                                           head_width=1.5, head_length=2,  # type: ignore
                                           sep_y = 0.02, sep_x = -0.01,  # type: ignore
                                           fontsize=0.03
                                           )
            ax.add_artist(arrs)

    def read_data(self, image, header = False, iloc = False):
        """Method to view images."""
        image_str, image_int = self.return_index(image, iloc = iloc) # type: ignore
        print('Reading ', image_str)

        # Extracting data from header
        with fits.open(os.path.join(self.dir_img, image_str)) as hdul: # type: ignore
            if iloc: data = hdul[self.im_type_dir[self.df_files.im_type.iloc[image_int]]['data_i']].data.astype(np.float32) # type: ignore
            else: data = hdul[self.im_type_dir[self.df_files.im_type.loc[image_int]]['data_i']].data.astype(np.float32) # type: ignore
            head = hdul[0].header # type: ignore
            hdul.close()

        if header == False: return data
        else: return head
    
    def get_moon_distance(self, image, iloc = False):
        """
        Method to calculate the angular separation with the Moon in degrees for a given observation.
        
        Parameters
        ---------
        image : int or str
            index of observation image or string to .fits file.
            
        Returns:
            astropy.Angle object with angular separation"""

        image_str, image_int = self.return_index(image, iloc = iloc) # type: ignore
        with fits.open(os.path.join(self.dir_img, image_str)) as hdul: # type: ignore
            heads = hdul[0].header # type: ignore
            hdul.close()
        RA = str(heads['RA']) + ' d'
        DEC = str(heads['DEC']) + ' d'
        if iloc: time = Time(self.df_files.iloc[image_int]['date_time'])
        else: time = Time(self.df_files.loc[image_int]['date_time'])
        loc = EarthLocation.of_site('Observatorio del Teide')
        moon_coords = get_body('moon', time = time, location = loc)
        moon_coords = SkyCoord(ra = moon_coords.ra, dec = moon_coords.dec, frame = 'icrs', unit = u.deg) # type: ignore
        obj_coords = SkyCoord(ra = Angle(RA), dec = Angle(DEC), frame = 'icrs', unit = u.deg) # type: ignore
        sep = obj_coords.separation(moon_coords)
        return sep



"""
Utility functions for WCS handling and image reprojection for stacking.
"""


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


"""
def filtering_df(iv_class, filters, plotting = False):
    print('Filtering dataset')
    # Filter seeing and moon distance from original dataframe
    df_filtered = iv_class.df_files
    df_original = iv_class.df_files
    n_fil = len(filters)
    mult_filters = False
    mult_filters_n = 1
    if 'filter' in filters.keys():
        if n_fil!=1: n_fil = n_fil-1
        if type(filters['filter'])!=str:
            mult_filters = True
            mult_filters_n = len(filters['filter'])
    if plotting:
        fig, ax = plt.subplots(ncols=n_fil, nrows=mult_filters_n+1, figsize=(n_fil*3,3*mult_filters_n))
        if n_fil==1: ax = np.array([ax])
        if mult_filters_n>1:
            ax = ax.flatten()
        i=0
        ax[0].set_ylabel('All observations')
        for fil in filters.keys():
            if fil!='filter':
                ax[i].hist(df_filtered[fil], bins=100)
                ax[i].axvline(df_filtered[fil].mean(), color = 'gray', label = 'mean %s: %.3f'%(fil, df_filtered[fil].mean()))
                ax[i].axvline(filters[fil], color = 'red', label = 'filtering value: %.3f'%(filters[fil]))
                ax[i].set_xlabel(fil)
                ax[i].legend()
                #plt.suptitle('All observations statistics')
                i+=1
    for fil in filters.keys():
        if fil in ['seeing', 'EZP', 'DUSTPLA', 'AIRMASS']:
            df_filtered = df_filtered[df_filtered[fil]<filters[fil]]
            df_filtered = df_filtered[df_filtered[fil] > 0]   
        elif fil in ['moon', 'TESSMAG']:
            df_filtered = df_filtered[df_filtered[fil]>filters[fil]]
        elif fil == 'filter' and mult_filters == False:
            df_filtered = df_filtered[df_filtered[fil]==filters[fil]]
        elif fil == 'filter' and mult_filters == True:
            df_list = []
            for fi in filters[fil]:
                df_list.append(df_filtered[df_filtered['filter']==fi])
        else: print('ERROR: UNRECOGNIZED FILTER')

    if plotting:
        if mult_filters == False:
            i=0
            for fil in filters.keys():
                if fil!='filter':
                    ax[i].hist(df_filtered[fil].to_numpy(), bins=100)
                    ax[i].axvline(df_filtered[fil].mean(), color = 'red', label = 'filtered mean %s: %.3f'%(fil, df_filtered[fil].mean()))
                    ax[i].set_xlabel(fil)
                    ax[i].legend()
                    i+=1
        else:
            #fig, ax = plt.subplots(ncols=n_fil, figsize=(n_fil*3,3))
            #if n_fil==1: ax = [ax]
            i=0
            for fil in filters.keys():
                if fil!='filter':
                    for j, df_i in enumerate(df_list):
                        index = i + (j+1)*n_fil
                        ax[index].hist(df_i[fil], bins=50)#, label = filters['filter'][j], alpha = 0.5)
                        ax[index].axvline(df_i[fil].mean(), color = plt.get_cmap('tab10')(j), 
                                      label = 'filtered mean: %.3f'%(df_i[fil].mean()))
                        ax[index].set_xlabel(fil)
                        ax[index].legend()
                        ax[(j+1)*n_fil].set_ylabel('Filtered '+filters['filter'][j])
                    i+=1
        plt.tight_layout()
    if mult_filters == False:
        df_filtered.index = np.arange(len(df_filtered))
        print('Total images after filtering: ', len(df_filtered))
        return df_filtered
    else:
        for i in range(len(df_list)):
            print('Total images after filtering in %s: %i, total time: %s s'%(filters['filter'][i], len(df_list[i]), df_list[i]['integration'].sum()))
        return tuple(df_list)
"""


"""def stacking(df, indexes, w_out, shape_out,
             sigma =3, min_area = 5):
    print('Aligning and stacking images')
    fil = df.iloc[0]['filter']
    object = df.iloc[0]['object']
    # Creating new folder for combined image
    if object+'_image' not in os.listdir():
        os.system('mkdir '+object+'_image')
    if fil not in os.listdir(object+'_image'):
        os.system('mkdir '+object+'_image/'+fil)
    # Build input list for reproject: (array, WCS) pairs
    cube = np.empty((len(indexes),) + shape_out, dtype=float)
    for i, fn in enumerate(df.iloc[indexes]['path']):
        with fits.open(fn) as hdul:
            hdu = hdul[0]  # adjust if image is in another extension
            data = hdu.data.astype(float)
            data = data/df.loc[i]['integration']
            wcs = WCS(hdu.header)
        reproj, _ = reproject_interp((data, wcs), w_out, shape_out=shape_out)
        if i==0: cube[0] = reproj
        # align all to first image
        try:
            registered_image, _ = aa.register(reproj, cube[0],
                                             detection_sigma = sigma,
                                             min_area = min_area)
            cube[i] = registered_image
        except ValueError as e:
            print('Cannot align index', indexes[i], ' - ', e)
        except aa.MaxIterError as e:
            print('Cannot align index', indexes[i], ' - ', e)
        if i == len(indexes)//4: print('  25% done')
        if i == len(indexes)//2: print('  50% done')
        
    # Sigma-clipping
    clip = sigma_clip(cube, sigma=sigma, axis=0, masked=True)
    stack_sigclip = np.ma.mean(clip, axis=0).filled(np.nan)
    print('\nStacking finished!')
    return stack_sigclip"""

def stacking_wcs(df, indexes, template_out, # w_out, shape_out,
                sig_clip = 3, add_name = '',
                rem_sky = True, norm = False):
    print('Aligning %i images'%(len(indexes)))
    fil = df.iloc[0]['filter']
    object = df.iloc[0]['object']
    int_total = 0
    with fits.open(object+'_image/'+template_out) as hdul:
        hdu = hdul[0]
        heads = hdu.header # type: ignore
        w_out = WCS(heads)
        shape_out = (heads["NAXIS1"], heads["NAXIS2"])
        pixscale = heads["SCALE"]
    # Build input list for reproject: (array, WCS) pairs
    cube = np.empty((len(indexes),) + shape_out, dtype=float)
    for i, fn in enumerate(df.iloc[indexes]['path']):
        try:
            with fits.open(fn) as hdul:
                hdu = hdul[0]
                head = hdu.header # type: ignore
                data = hdu.data.astype(float) # type: ignore
                # substract sky background
                if rem_sky:
                    data = data - head["FLUXSKY"]
                data = data/df.iloc[indexes[i]]['integration']
                # normalize counts
                if norm:
                    data = data / np.max(data)
                wcs = WCS(hdu.header) # type: ignore
            cube[i], _ = reproject_interp((data, wcs), w_out, shape_out=shape_out)
            int_total += df.iloc[indexes[i]]['integration']
            if i == len(indexes)//4: print('  25% done')
            if i == len(indexes)//2: print('  50% done')
            if i == len(indexes)//(3/4): print('  75% done')
        except Exception as error: print('%s in image %i'%(type(error).__name__, i))
    # stacking
    print('Stacking images')
    clip = sigma_clip(cube, sigma=sig_clip, axis=0, masked=True)
    stack_sigclip = np.ma.mean(clip, axis=0).filled(np.nan) # type: ignore
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
    # Create HDU and write to disk
    hdu = fits.PrimaryHDU(data=stack_sigclip, header=hdr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(object+'_image/'+df.iloc[indexes[0]]['filter']+add_name+".fits", overwrite=True)
    
    #fig, ax = plt.subplots()
    #ax.imshow(stack_sigclip, origin = 'lower', norm = 'log', cmap = 'gray')
    #return stack_sigclip, cube

"""def stacking_align(df, indexes, template_out, #w_out, shape_out,
                    ref_s,
                    sigma = 3, min_area = 5,
                    sig_clip = 3,
                    show_ref = False,
                   add_name = ''):
    
    with fits.open(template_out) as hdul:
        hdu = hdul[0]
        heads = hdu.header
        w_out = WCS(heads)
        shape_out = (heads["NAXIS1"], heads["NAXIS2"])
        pixscale = heads["SCALE"]
        
    print('Aligning %i images'%(len(indexes)))
    print('Finding transform to image indexes[0]')
    # Build input list for reproject: (array, WCS) pairs
    ref_s_c = []
    for i, rs in enumerate(ref_s):
        ref_s_c.append(SkyCoord(ra=rs.split()[0]+'d', dec=rs.split()[1]+'d',frame = 'icrs'))
    cube = np.empty((len(indexes),) + shape_out, dtype=float)
    for i, fn in enumerate(df.iloc[indexes]['path']):
        with fits.open(fn) as hdul:
            hdu = hdul[0]  # adjust if image is in another extension
            data = hdu.data.astype(float)
            data = data/df.loc[i]['integration']
            wcs = WCS(hdu.header)
        # Not reprojecting with WCS, only  to the first image, the resti with transformation with stars
        # obtain pixel coordinates of ref_stars
        ref_px = []
        for rs in ref_s_c:
            px_x, px_y = skycoord_to_pixel(rs, wcs)
            ref_px.append([px_x,px_y])
        img_pos_xy = np.array(ref_px)
        """
"""
        print(img_pos_xy)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(data, origin='lower', norm='log', cmap='gray')
        for k, px_xy in enumerate(img_pos_xy):
            circ = plt.Circle((px_xy[0], px_xy[1]), 20, fill=False, edgecolor=plt.get_cmap('tab10')(k), linewidth=1)
            ax.add_patch(circ)"""
"""
        if i==0: 
            reference, _ = reproject_interp((data, wcs), w_out, shape_out=shape_out)
            print('Transforming images to reference WCS with selected star alignment.')
            # obtain reference stars in px coordinates 
            """
"""
            if show_ref:
                fig, ax = plt.subplots()
                ax.imshow(reference, origin='lower', cmap = 'gray', norm = 'log')
                for k, px_xy in enumerate(img_pos_xy):
                    circ = plt.Circle((px_xy[0], px_xy[1]), 20, fill=False, edgecolor=plt.get_cmap('tab10')(k), linewidth=1)
                    ax.add_patch(circ)"""
"""
        # align all to first image
        try:
            transform, (pos_img, pos_img_rot) = aa.find_transform(img_pos_xy, reference,
                                                                  detection_sigma = sigma,
                                                                  min_area = min_area)
            #print('Rotation, scale, translation : ', transform.rotation, transform.scale, transform.translation)
            cube[i], _ = aa.apply_transform(transform, data, reference)
            #print(data.mean())
            #print(cube[i].mean(), cube[i].shape)

        except ValueError as e:
            print('Cannot align index', indexes[i], ' - ', e)
        except aa.MaxIterError as e:
            print('Cannot align index', indexes[i], ' - ', e)
        if i == len(indexes)//4: print('  25% done')
        if i == len(indexes)//2: print('  50% done')
        if i == len(indexes)//(3/4): print('  75% done')
    # stacking
    print('Stacking images')
    
    clip = sigma_clip(cube, sigma=sig_clip, axis=0, masked=True)
    stack_sigclip = np.ma.mean(clip, axis=0).filled(np.nan)
    # Convert WCS to FITS header
    hdr = w_out.to_header()
    # Make sure header knows the image size
    hdr['NAXIS']  = 2
    hdr['NAXIS1'] = stack_sigclip.shape[1]
    hdr['NAXIS2'] = stack_sigclip.shape[0]
    hdr['SCALE'] = pixscale
    hdr['FLUXSKY'] = np.mean(stack_sigclip)
    # Create HDU and write to disk
    hdu = fits.PrimaryHDU(data=stack_sigclip, header=hdr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(object+'_image/'+df.loc[indexes[0]]['filter']+add_name+".fits", overwrite=True)
    #return cube"""

"""
final_data = np.zeros((naxis1, naxis2))
for i, fname in enumerate(df['path']):
    #with fits.open(fname) as hdul: # type: ignore
    #    head = hdul[0].header # type: ignore
    #    data = hdul[0].data
    #    hdul.close()
    repr_data, _ = reproject_interp(fname, hdr_out, shape_out = (naxis1, naxis2))
    repr_data = repr_data / df.loc[i]['integration']
    final_data +=repr_data
"""
