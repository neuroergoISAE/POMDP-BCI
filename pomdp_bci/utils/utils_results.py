"""
Utils for results I/O and analysis

Author: Juan Jesús Torre
Mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pandas as pd


def save_results(filename, data, metadata):
    """
    Save the results of the analysis on a hdf5 with metadata using pandas

    Arguments
    ---------

    filename: str or path object
        Path to save the results file to

    data: pd.DataFrame
        Data to save

    metadata: dict
        Information about the performed analysis relevant to the results
    """

    store = pd.HDFStore(filename)
    store.put('data', data)
    store.get_storer('data').attrs.metadata = metadata
    store.close()


def load_results(filename):
    """
    Load the results and metadata of the analysis from a hdf5 using pandas

    Arguments
    ---------

    filename: str or path object
        Path to save the results file to

    Returns
    -------

    data: pd.DataFrame
        Results from the experiment

    metadata: dict
        Information about the performed analysis relevant to the results
    """

    with pd.HDFStore(filename) as store:
        data = store['data']
        metadata = store.get_storer('data').attrs.metadata

    return data, metadata
