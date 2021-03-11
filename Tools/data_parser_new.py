import os
import glob as gb
import pandas as pd
import numpy as np
import re


'''
Functions in file

numericalSort(value)
parse(initial_path = "/Users/kevin/Dropbox/VCG_MachineLearning/Data/New_Simulations/", verbose=False, vcg_select='*', at_map=False, patient_number=0)

'''


numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def parse(initial_path="/Users/kevin/Dropbox/VCG_MachineLearning/Data/New_Simulations/", verbose=False, vcg_select='*', at_map=False, patient_number=0):
    """
    Get all data from patient number 'patient_number' in folder LBBB_LVdyssync stored into the 'patient'variable
    Returns:
    --------
    patient: 
        pandas.Series containing all data of the patient. See comments in __main__ for indexing.
    
    https://open.spotify.com/artist/5BvJzeQpmsdsFp4HGUYUEx
    Warning: python3 is __next__, python2 is next
    Credit: Yngve Moe wrote an original version of this script
    """

    vcg_type = 'kVCG'

    patients_folders = [os.path.join(initial_path, sub_path) for sub_path in next(os.walk(initial_path))[1]]

    # for patient_folder in patients_folders[patient_number:]:
    for patient_folder in [patients_folders[patient_number]]:

        # include patient identifier
        pt_id = patient_folder.rsplit('/', 1)[1]

        alt_path = initial_path.rsplit('/', 2)[0] + '/' + pt_id

        # load the parameters of each simulation
        eval_values = pd.read_csv(gb.glob(patient_folder + '/*eval*')[0], header=0,  #HEADER=1 is a bug!?!
                                  usecols=(3, 4, 5, 11, 12, 13), names=['x', 'y', 'z', 'cond', 'L', 'R'], sep=',', index_col=False)
        # search for nodes file
        if os.path.isfile(alt_path+'/'+pt_id+'_nodes.txt'):
            nodes = np.loadtxt(alt_path+'/'+pt_id+'_nodes.txt', skiprows=1, usecols=(0, 8, 16))
            # find unique pacing locations
            eval_values['pacing_nodes'] = list(zip(eval_values['x'], eval_values['y'], eval_values['z']))
            unique_nodes = eval_values['pacing_nodes'].unique()
            # match unique nodes to their node numbers
            node_list = []
            for n in unique_nodes:
                node_list.append(np.linalg.norm(np.asarray(n)-nodes, axis=1).argmin()+1)

            # map nodes back into dataframe
            xmap = dict(zip(eval_values['pacing_nodes'].unique(), node_list))
            # make these int?
            eval_values['pacing_nodes'] = eval_values['pacing_nodes'].map(xmap)

        # load the experimental VCG

        vcg_real = pd.read_csv(gb.glob(alt_path + '/*' + vcg_type + '*.txt')[0],
                               header=None, names=['VCGx', 'VCGy', 'VCGz'],
                               sep=' ', index_col=False)

        # load the VCG of each simulation
        vcg_reading = gb.glob(patient_folder + '/VCGs/' + vcg_select + '.txt')
        vcg_reading = sorted(vcg_reading, key=numericalSort)
        vcg_sims = [int(filename.split('_')[-1].strip('.txt')) for filename in vcg_reading]

        # vcg_model = [np.nan]*max(vcg_sims)
        vcg_model = [np.nan]*181  # hack for now because BiV2 is missing sim 181 #FIXME
        for i in range(len(vcg_reading)):
            vcg_model[vcg_sims[i]-1] = pd.read_csv(vcg_reading[i], header=None, usecols=[2, 3, 4], names=['VCGx', 'VCGy', 'VCGz'], sep='\t', index_col=False)

        # load the activation maps if requested
        if at_map:
            at_files = gb.glob(patient_folder + '/ATmap/' + vcg_select + '.txt')
            at_files = sorted(at_files, key=numericalSort)
            at_maps = pd.concat([pd.read_csv(item, header=None, names=[item.split('_')[-1].strip('.txt')]) for item in at_files], axis=1)
            at_maps.index += 1

        # Print some info about data
        if False:
            print(pt_id + '\n' + ''.join(str(node_list)))
        if verbose:
            print(pt_id +
                  '\n\tSimulations in pts_to_eval: %i \
                  \n\tVCG files: %i \
                  \n\tMax VCG number: %i \
                  \n\tNumber of pacing nodes %i'
                  %(len(eval_values), len(vcg_reading), max(vcg_sims), len(unique_nodes)))

        # store all into a panda series
        if at_map:
            patient = pd.Series([pt_id, eval_values, vcg_real, vcg_model, at_maps], index=['pt_id', 'eval_values', 'vcg_real', 'vcg_model', 'at_maps'])
        else:
            patient = pd.Series([pt_id, eval_values, vcg_real, vcg_model], index=['pt_id', 'eval_values', 'vcg_real', 'vcg_model'])

    return patient



if __name__ == "__main__":

    #patient = parse(initial_path = "/Users/victoriavincent/Dropbox/VCG_MachineLearning/Data/New_Simulations/", at_map = True, patient_number=0)
    patient = parse(initial_path = "/Users/victoriavincent/CMRG/VCG_MachineLearning/Troubleshooting/BiV5_noScar/Data/New_Simulations/", at_map = True, verbose = True, patient_number=0)

    #mean_plot(patient)
            
    # example calls
   # print patient['opt_desync'][0]
    #print patient['desync']
    #print patient['eval_values']
    #print patient['vcg_real'] # ['px'],['py'],['pz']
    #print patient['vcg_model']  [1] # ['px'],['py'],['pz']
    print(patient['pt_id'])
