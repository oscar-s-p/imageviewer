# Image Viewer

A Python package for viewing and analyzing astronomical FITS images from various telescopes including the Two-Meter Twin Telescope (TTT) of Light Bridges (Tenerife, Spain), Hubble Space Telescope (HST), and Keck II.

Utility functions to align and stack multiple observations of the same object are also included, along with a full photometric analysis pipeline.

## Features

- Automatic discovery and cataloging of FITS images in specified directories
- Support for multiple telescope formats (LB, HST, Keck II)
- Image viewing with matplotlib and astropy visualization
- RGB image composition with automatic WCS alignment
- Header information extraction
- Moon distance calculations
- Data manipulation and plotting utilities (stretches, zoom, WCS reprojection)
- Alignment and reprojection of multiple images with WCS reference frame
- Image stacking with sigma-clipped weighted average
- PSF and aperture photometry with reference catalogue calibration
- Source detection and interactive coordinate selection

## Installation

### Prerequisites

- Python 3.7+

### Install from GitHub into IPython kernel
```python
%pip install --upgrade "git+https://github.com/oscar-s-p/imageviewer.git"
```

### Install from source

1. Clone the repository:
```bash
git clone https://github.com/oscar-s-p/imageviewer.git
cd imageviewer
```

2. Install the package:
```bash
pip install -e .
```

## Usage

### Image viewer

```python
from imageviewer import image_viewer

# Initialize viewer with image directory
viewer = image_viewer(directory='/path/to/fits/images')

# Find images of a specific object
indices = viewer.image_finder('QSO0957+561', date='2024-01-15', filter='r')

# View an image
viewer.view_image(indices[0])

# View multiple images in a grid
viewer.view_image(indices, nrows_ncols=(2, 3))

# View RGB composite
viewer.view_RGB('QSO0957+561', '2024-01-15', filters='irg')

# Get header information
viewer.header_info(indices[0])

# Calculate moon distance
moon_dist = viewer.get_moon_distance(indices[0])
```

### Image stacking

```python
from imageviewer import filter_df, final_wcs, stacking_wcs

# Filter observations by quality criteria
filtered = filter_df(viewer.df_files, {'seeing': ('below', 2.0), 'filter': 'r'})

# Create a WCS template for the output stack
wcs_template, shape = final_wcs('QSO0957+561', ra=149.74, dec=55.89,
                                 fov_x=120, fov_y=120, pixscale=0.23)

# Stack the selected images
stacking_wcs(viewer.df_files, filtered.index, 'output_template.fits')
```

### Photometry

```python
from imageviewer import detect_sources, get_catalogue, photo_analysis, get_magnitude, get_coordinates

# Detect sources in an image
sources = detect_sources('stacked.fits', method='find_peaks', fwhm=3.0)

# Query a reference catalogue
catalogue = get_catalogue('stacked.fits', catalogue='PanSTARRS', filter='r', mag_range=(14, 19))

# Run full PSF + aperture photometry pipeline
phot_table = photo_analysis('stacked.fits', init_table=sources, cat_table=catalogue,
                             photometry_params={'fwhm': 3.0, 'aperture_radius': 5.0})

# Get the magnitude at a specific coordinate
mag, mag_err = get_magnitude('stacked.fits', phot_table, coords=(ra, dec))

# Interactively select coordinates by clicking on an image
coords = get_coordinates('stacked.fits')
```

## Supported Telescopes

- **LB (Light Bridges - TTT)**: Standard FITS format with metadata encoded in filename (`telescope_camera_date-time_object_filter.fits`)
- **HST (Hubble Space Telescope)**: FITS with header metadata; supports WFC3/UVIS (0.04 arcsec/px) and WFC3/IR (0.13 arcsec/px)
- **Keck II**: Supported with detector-specific parameters and CD-matrix WCS

## API Reference

### `image_viewer` class

**Constructor**

```python
image_viewer(directory, list_available=True, folder_list=None,
             previous_df=None, filters=None, bad_format=False,
             im_type_input=None, list_files=None)
```

**Methods**

| Method | Description |
|---|---|
| `return_index(image, iloc=False)` | Return image path and integer index given either one |
| `image_finder(object, date=None, filter=None, return_df=False, printeo=False)` | Find images matching object name, date, and filter |
| `dataframe_add(label)` | Add a metadata column to the image DataFrame (`'seeing'`, `'moon'`, `'integration'`, `'EZP'`, `'date'`, or any FITS header keyword) |
| `header_info(image, interesting_keys=[...], hdul_i=0, iloc=False, return_value=False)` | Display or return FITS header information |
| `view_image(image, RGB=False, nrows_ncols=None, figsize=None, save_name=None, ...)` | Display one or more FITS images; accepts all `data_manipulation` and `plotting` kwargs |
| `view_RGB(object, date, name_list=False, filters='irg', object_coordinates=None, ...)` | Compose and display an RGB image from three single-band observations |
| `data_manipulation(image_str, centered=True, zoom=False, stretch='linear', percentile=None, vminmax=(None,None), rotate=True, iloc=False)` | Prepare image data for display (centering, zoom, stretch, WCS reprojection) |
| `plotting(cutout, norm, fig, ax, ax_i, cmap='gray', scalebar_arcsec=5, scalebar_frame=False, add_circle=None, RGB=False, ...)` | Low-level plotting with WCS axes, scale bar, directional arrows, and optional circle overlays |
| `read_data(image, header=False, iloc=False)` | Read and return raw image array and optionally its FITS header |
| `get_moon_distance(image, iloc=False)` | Calculate the angular distance between the observation pointing and the Moon |

**`data_manipulation` stretch options:** `'linear'`, `'sqrt'`, `'power'`, `'log'`, `'sinh'`, `'asinh'`

### Utility functions

```python
from imageviewer import filter_df, final_wcs, stacking_wcs
```

**`filter_df(original_df, filters_dict, ask_all=False, plotting={})`**
Filter the image DataFrame by quality criteria. `filters_dict` keys can be: `seeing`, `moon`, `EZP`, `airmass`, `magnitude`, `filter`, `telescope`, `camera`. Values are `(comparison, threshold)` tuples where comparison is `'above'`, `'below'`, or `'equal'`.

**`final_wcs(object, ra, dec, fov_x, fov_y, pixscale, name_out='output_template.fits')`**
Create a WCS template FITS file defining the output grid (position, field of view in arcsec, pixel scale in arcsec/px) for subsequent stacking. Returns `(WCS, output_shape)`.

**`stacking_wcs(df, indexes, template_out, sig_clip=3, add_name='', rem_sky=True, norm=False, print_tests=False)`**
Reproject and stack all images referenced by `indexes` onto the WCS template. Performs sky subtraction, electron-count conversion (camera-specific gain), and sigma-clipped weighted-average combination. Returns the stacked FITS file path and stacking metadata header.

### Photometry functions

```python
from imageviewer import photo_analysis, detect_sources, get_catalogue, get_magnitude, get_coordinates
```

**`detect_sources(filename, method='find_peaks', sky_sigma=3.0, maxiters=5, sky_threshold=3.0, fwhm=3.0, init_table=None, add_sources=False, plot=True)`**
Detect point sources in a FITS image. `method` can be `'find_peaks'` or `'IRAF'`. Returns a table of RA/DEC coordinates.

**`get_catalogue(filename, catalogue='Simbad', filter='g', mag_range=(13,18), print_info=True, plot=False, rotate=True, scalebar_arcsec=60)`**
Query a reference catalogue for stars in the image field. Supported catalogues: `'SDSS'`, `'PanSTARRS'`, `'Simbad'`, `'Gaia'`. Returns a filtered table with positions and magnitudes.

**`photo_analysis(filename, init_table=None, cat_table=None, sky_background={}, photometry_params={}, matching_params={}, plot=True, n_fig_init=0, stacked=False, print_info=True)`**
Full PSF and aperture photometry pipeline with catalogue-based magnitude calibration. Returns a photometry table with calibrated magnitudes and uncertainties.

**`get_magnitude(filename, photometry_table, coords, pix_dist=5, print_info=False, plot=False, n_fig=99)`**
Retrieve the calibrated magnitude at a given RA/DEC coordinate from a photometry table. Returns `[magnitude, error]` or `[None, None]` if no source is found within `pix_dist` pixels.

**`get_coordinates(filename, coords=None, scalebar_arcsec=60, rotate=True)`**
Display the image and allow interactive click-based RA/DEC coordinate selection. Returns a coordinate table.

## Dependencies

- numpy
- pandas
- matplotlib
- astropy
- reproject
- ccdproc
- astroquery
- photutils
- scipy
- ipywidgets
- IPython

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contact

Through GitHub: https://github.com/oscar-s-p/imageviewer
