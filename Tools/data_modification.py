import numpy as np
import pandas as pd

from numpy.linalg import norm
from scipy.signal import resample

from data_analysis import calc_cost, calc_cost_wrong, calc_cost_rms

'''
Functions in file

calc_cost_patient(patient, cost)
normalize_patient(patient)
trim_tat_patient(patient, ignore_scar=False)
scale_at_patient(patient, length=None)
resample_patient(patient, length=None)
create_data_matrix(patient, transforms=None)

'''
def calc_cost_patient(patient, cost='theta'):

    """
    Calculates the cost function between the simulated VCG and the clinical VCG
        Cost function is modified from Villongco et al 2014 PBMB so that identical vectors
        will give a cost function of zero (previous version did not)
        
    Arguments
    ---------
    patient : pandas.Series
        The dataframe containing all information about the patient
    cost: string
        String defining the type of cost function.  {'theta,theta_old,rms'}
        
    Returns
    -------
    patient : pandas.Series
        The dataframe containing all information about the patient with the following added:
            patient['theta']: time series of theta for each 'vcg_model'
    """
    # Need some input filtering for cost

    if cost == 'theta':
        cost_function = calc_cost
    elif cost == 'theta_old':
        cost_function = calc_cost_wrong
    elif cost == 'rms':
        cost_function = calc_cost_rms
    else:
        raise ValueError('Cost function must be one of the following: {theta|theta_old|rms}')

    patient[cost] = [None]*len(patient['vcg_model'])
    
    for i in range(0,len(patient['vcg_model'])):
        if isinstance(patient['vcg_model'][i], pd.DataFrame):
            patient[cost][i] = cost_function(patient['vcg_model'][i], patient['vcg_real'])
        else:
            patient[cost][i] = np.nan
    return patient

def normalize_patient(patient):
    """Normalises the VCG by dividing by the maximum heart vector.
        Creates a new 
    """
#    mag = np.sqrt(x.dot(x)) is much faster, consider refactoring
#   https://stackoverflow.com/questions/9171158/how-do-you-get-the-magnitude-of-a-vector-in-numpy

    patient['vcg_real'] = patient['vcg_real']  / norm(patient['vcg_real'],axis=1).max()
    patient['vcg_model_norm'] = [None]*len(patient['vcg_model'])
    for i, vcg_model in enumerate(patient['vcg_model']):

        # Standardise VCG signal:
        if isinstance(patient['vcg_model'][i], pd.DataFrame):
            patient['vcg_model_norm'][i] = patient['vcg_model'][i] / norm(patient['vcg_model'][i],axis=1).max()
        else:
            patient['vcg_model_norm'][i] = np.nan

    patient['vcg_model'] = patient['vcg_model_norm']
    patient.drop(labels=['vcg_model_norm'])
    return patient


def trim_tat_patient(patient, ignore_scar=False):
    '''
    Trim each simulated VCG to be the length of the total activation time

    '''
    if ignore_scar:
        nodes_file = '/Users/victoriavincent/Dropbox/VCG_MachineLearning/Data/%s/%s_nodes.txt'%(patient['pt_id'],patient['pt_id'])
        nodes = np.loadtxt(nodes_file, skiprows=1, usecols=(39))
        not_scar = ~(nodes > 0.6)
        not_scar = not_scar.astype(int)
        df_ignore_scar = patient['at_maps'].mul(not_scar, axis=0)

        for i, vcg_model in enumerate(patient['vcg_model']):

            # Standardise VCG signal:
            if isinstance(patient['vcg_model'][i], pd.DataFrame):
                sim_str = '%03d'%(i+1)
                tat = df_ignore_scar[sim_str].max()
                patient['vcg_model'][i] = patient['vcg_model'][i].loc[0:tat]
            else:
                patient['vcg_model'][i] = np.nan

    else:
        for i, vcg_model in enumerate(patient['vcg_model']):

            # Standardise VCG signal:
            if isinstance(patient['vcg_model'][i], pd.DataFrame):
                sim_str = '%03d'%(i+1)
                tat = patient['at_maps'][sim_str].max()
                patient['vcg_model'][i] = patient['vcg_model'][i].loc[0:tat]
            else:
                patient['vcg_model'][i] = np.nan

    return patient


def scale_at_patient(patient, length=None):
    '''
    Scale the total activation time of each activation map

    '''
    if isinstance(length,int):
        qrs_dur = length
    else:
        qrs_dur = len(patient['vcg_real'])

    patient['at_maps'] = patient['at_maps'] / patient['at_maps'].max() * qrs_dur

    return patient


def resample_patient(patient, length=None):
    '''
    Resample all of the simulated VCGs to have the same length and 
    number of samples as the clinical VCG

    '''
    if isinstance(length,int):
        qrs_dur = length
    else:
        qrs_dur = len(patient['vcg_real'])



    for i, vcg_model in enumerate(patient['vcg_model']):

            # resample simulated VCGs:
            if isinstance(patient['vcg_model'][i], pd.DataFrame):

                patient['vcg_model'][i] = pd.DataFrame(data=resample(patient['vcg_model'][i],qrs_dur), columns=['VCGx', 'VCGy', 'VCGz'])
            else:
                patient['vcg_model'][i] = np.nan

    return patient


def create_data_matrix(patient, transforms=None):
    """Returns a datamatrix for the patients where each column is a simulation.
    Arguments
    ---------
    patient : pandas.DataFrame
        The dataframe containing all information about the patient
    transforms : Array like
        List containing three transformations to be applied to the different
        axis of the VCG signal.
    
    Returns
    -------
    data_matrix : np.ndarray
        A matrix where each column is a VCG signal for each simulation (the
        three dimensions are just concatenated to one vector).
    """
    data_matrix = np.zeros((len(patient['vcg_model']), np.prod(patient['vcg_model'][0].shape)))
    for i, simulation in enumerate(patient['vcg_model']):
        sim = simulation.values.copy()

        if transforms is not None:
            for j in range(3):
                if transforms[j] is not None:
                    sim[j, :] = transforms[j](sim[j, :])

        data_matrix[i] = sim.reshape(np.prod(sim.shape))

    return data_matrix

