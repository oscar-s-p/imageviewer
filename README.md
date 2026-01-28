# Image Viewer

A Python package for viewing and analyzing astronomical FITS images from various telescopes including the Two-Meter Twin Telescope (TTT) of Light Bridges (Tenerife, Spain), Hubble Space Telescope (HST), and Keck II.

## Features

- Automatic discovery and cataloging of FITS images in specified directories
- Support for multiple telescope formats (LB, HST, Keck II)
- Image viewing with matplotlib and astropy visualization
- RGB image composition
- Header information extraction
- Moon distance calculations
- Data manipulation and plotting utilities

## Installation

### Prerequisites

- Python 3.7+
- Required packages: numpy, pandas, matplotlib, astropy, reproject

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

Or install in development mode:
```bash
pip install -e .[dev]
```

## Usage

```python
from imageviewer import image_viewer

# Initialize viewer with image directory
viewer = image_viewer(directory='/path/to/fits/images')

# View an image
viewer.view_image('image_name.fits')

# View RGB composite
viewer.view_RGB(object_name, date)

# Get header information
header = viewer.header_info('image_name.fits')
```

## Supported Telescopes

- **LB (Ligth Bridges - TTT)**: Standard FITS format with metadata in filename
- **HST (Hubble Space Telescope)**: FITS with header metadata
- **Keck II**: Supported with detector-specific parameters

## Methods

- `return_index()`: Get image path and index
_ `dataframe_add()`: Add columns from FITS files to the dataset
- `image_finder()`: Search for images by object, date, filter
- `header_info()`: Display header information
- `view_image()`: Display single or multiple FITS image
- `view_RGB()`: Create and display RGB composite images
- `data_manipulation()`: Manipulate image data (used by `view_image` and `view_RGB`)
- `plotting()`: Advanced plotting with WCS coordinates (used by `view_image` and `view_RGB`)
- `read_data()`: Read image data and headers
- `get_moon_distance()`: Calculate angular distance to moon

## Dependencies

- numpy
- pandas
- matplotlib
- astropy
- reproject

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

Ask for permission to use

## Contact

Through github