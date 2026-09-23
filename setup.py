from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='Cycles-utils',
    version='4.0.6.post',
    author='Yuning Shi',
    author_email="shiyuning@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    description='Python scripts to build Cycles input files and post-process Cycles output files',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PSUmodeling/Cycles-utils',
    license='MIT',
    python_requires='>=3.10',
    # Core deps only: what `import cycles` / running simulations / reading
    # output needs. cartopy/geopandas/fiona moved to the `plot` extra so a
    # minimal install doesn't require the GDAL/cartopy stack.
    install_requires=['pandas>=1.2.4', 'numpy>=1.19.5', 'matplotlib>=3.4.2'],
    extras_require={
        # Needed for Cycles.plot_yield / plot_operations / plot_map /
        # plot_satellite_map and cycles_tools.read_geospatial_file
        'plot': ['cartopy>=0.18.0', 'geopandas>=0.9.0', 'fiona>=1.8.20', 'shapely>=1.7.1'],
        'soilgrids':  ['geopandas>=0.9.0', 'rioxarray>=0.5.0', 'owslib>=0.24.1', 'rasterio>=1.2.3', 'shapely>=1.7.1'],
        'gssurgo': ['geopandas>=0.9.0', 'shapely>=1.7.1', 'fiona>=1.8.20'],
        'weather': ['netCDF4>=1.5.7', 'tqdm>=4.60.0'],
        # Convenience bundle for contributors/CI running the full test suite
        'all': ['cartopy>=0.18.0', 'geopandas>=0.9.0', 'fiona>=1.8.20', 'shapely>=1.7.1',
                'rioxarray>=0.5.0', 'owslib>=0.24.1', 'rasterio>=1.2.3', 'netCDF4>=1.5.7',
                'tqdm>=4.60.0'],
    },
)
