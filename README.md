# alaska_catalog
Code to investigate publicly available AACSE earthquake catalog for possible projects.

/data_acquisition:
The datafiles in /catalog_css, the earthquake catalog information for the 2018-2019 AACSE experiment, are downloaded from https://scholarworks.alaska.edu/handle/11122/11967.

The find_templates notebook reads in the earthquake catalog information, retrieves waveforms of the earthquakes from IRIS, and saves the traces as templates in an h5 file if they meet the criteria specified within the notebook.
