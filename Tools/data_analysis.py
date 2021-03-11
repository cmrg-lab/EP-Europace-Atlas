import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from numpy.linalg import norm
from sklearn.decomposition import PCA


'''
Functions in file:

calc_cost_wrong(vcg1, vcg2)
calc_cost(vcg1, vcg2)
calc_cost_rms(vcg1, vcg2)
calc_cost_rms_norm(vcg1, vcg2)
split_vcg_vector(vcg)
df_vcg_vector(vcg):
plot_vcgs(labels, vcg1, vcg2=None, vcg3=None, ref_lines=False, arrows=False, alt_lines=False)
add_arrow(line, frequency=5, size=20, color=None)
prol2cart(lambd, mu, theta, d)
cart2prol(x, y, z, d)
prolateHammer(mu, theta)
    
'''


def calc_cost_wrong(vcg1, vcg2):
    """Calculates the cost function between the simulated VCG and the clinical VCG
        Cost function is defined in Villongco et al 2014 PBMB

    Arguments
    ---------
    vcg1,vcg2 : pandas.DataFrame
        The dataframe the patient VCG in three columns

    Returns
    -------
    theta: float
        cost function evaluation between the two input VCGs
    """

    max_vcg1 = max(norm(vcg1, axis=1))
    max_vcg2 = max(norm(vcg2, axis=1))
    t_tot = min(len(vcg1), len(vcg2))
    theta_t = np.zeros(t_tot)
    for t in range(0, t_tot):
        theta_t[t] = (norm(vcg1.iloc[t, :])/max_vcg1)*(np.arccos(np.dot(vcg2.iloc[t, :], vcg1.iloc[t, :])/(max_vcg1*max_vcg2))/np.pi)
    theta_t = theta_t/t_tot
    theta = theta_t.sum()

    return theta


def calc_cost(vcg1, vcg2):
    """Calculates the cost function between the simulated VCG and the clinical VCG
        Cost function is modified from Villongco et al 2014 PBMB so that identical vectors
        will give a cost function of zero (previous version did not)

    Arguments
    ---------
    vcg1,vcg2 : pandas.DataFrame
        The dataframe the patient VCG in three columns

    Returns
    -------
    theta: float
        cost function evaluation between the two input VCGs
    """

    max_vcg1 = max(norm(vcg1, axis=1))
    max_vcg2 = max(norm(vcg2, axis=1))
    t_tot = min(len(vcg1), len(vcg2))
    theta_t = np.zeros(t_tot)
    for t in range(0, t_tot):
        cos = np.dot(vcg2.iloc[t, :], vcg1.iloc[t, :])/(norm(vcg2.iloc[t, :])*norm(vcg1.iloc[t, :]))
        cos = np.clip(cos, -1, 1)
        theta_t[t] = (norm(vcg1.iloc[t, :])/max_vcg1)*(np.arccos(cos)/np.pi)
    theta_t = theta_t/t_tot
    theta = theta_t.sum()

    return theta


def calc_cost_rms(vcg1, vcg2, comp=all):
    """Calculates the RMS error between two VCGs. Total RMS error is
        the sum of the three individual leads

    Arguments
    ---------
    vcg1,vcg2 : pandas.DataFrame
        The dataframe the patient VCG in three columns

    Returns
    -------
    rmse: float
        RMSE evaluation between the two input VCGs
    """

    rmsx = ((vcg1['VCGx']-vcg2['VCGx']) ** 2).mean() ** 0.5
    rmsy = ((vcg1['VCGy']-vcg2['VCGy']) ** 2).mean() ** 0.5
    rmsz = ((vcg1['VCGz']-vcg2['VCGz']) ** 2).mean() ** 0.5
    rmse = rmsx + rmsy + rmsz

    if comp == all:
       rmse = rmsx + rmsy + rmsz
    if comp == 'VCGx':
        rmse = rmsx
    if comp == 'VCGy':
        rmse = rmsy
    if comp == 'VCGz':
        rmse = rmsz 

    return rmse


def calc_cost_rms_norm(vcg1, vcg2):
    """Calculates the RMS error between two magnitude normalized VCGs. Total
        RMS error is sum of the three individual leads

    Arguments
    ---------
    vcg1,vcg2 : pandas.DataFrame
        The dataframe the patient VCG in three columns

    Returns
    -------
    rmse: float
        RMSE evaluation between the two input VCGs after normalizing magnitudes
    """

    vcg1_norm = vcg1/max(norm(vcg1, axis=1))
    vcg2_norm = vcg2/max(norm(vcg2, axis=1))
    rmse = calc_cost_rms(vcg1_norm, vcg2_norm)

    return rmse


def split_vcg_vector(vcg):
    """Split a vectorized VCG with a single function call

    Arguments
    ---------
    vcg : numpy.array
        The array containing a vcg vector

    Returns
    -------
    vcg_x,vcg_y,vcg_z : numpy.array
        The seperated vcg components
    """
    if len(vcg) % 3 != 0:
        raise ValueError("Input vector length must be divisible by zero")

    lead_len = int(len(vcg)/3)
    vcg_x = vcg[0:lead_len]
    vcg_y = vcg[lead_len:lead_len*2]
    vcg_z = vcg[lead_len*2:]

    return vcg_x, vcg_y, vcg_z


def df_vcg_vector(vcg):
    """Split a vectorized VCG with a single function call and return a df

    Arguments
    ---------
    vcg : numpy.array
        The array containing a vcg vector

    Returns
    -------
    df_vcg : pandas.Dataframe
        dataframe with the three seperated vcg components
    """

    # Need to test and make more robust to bad input
    if isinstance(vcg, pd.DataFrame):
        vcg = vcg.to_numpy()

    vcg_x, vcg_y, vcg_z = split_vcg_vector(vcg)
    df_vcg = pd.DataFrame({'VCGx': vcg_x, 'VCGy': vcg_y, 'VCGz': vcg_z})

    return df_vcg


def plot_vcgs(labels, vcg1, vcg2=None, vcg3=None, ref_lines=False, arrows=False, title=False, alt_lines=False):
    """Split a vectorized VCG with a single function call

    Arguments
    ---------
    vcg1 : list
        A list containing the x,y, and z components
        of a vcg (i.e. [vcg_x,vcg_y,vcg_z])

    vcg2 : list
        A list containing the x,y, and z components
        of a vcg (i.e. [vcg_x,vcg_y,vcg_z])

    labels : list
        A list containing the labels for plotting

    Returns
    -------
    vcg_x,vcg_y,vcg_z : numpy.array
        The seperated vcg components
    """

    # sns.reset_orig()
    fig = plt.figure(figsize=(9, 9))
    fig.patch.set_facecolor('white')
    ax1 = plt.subplot2grid((12, 12), (1, 0), colspan=7, rowspan=2)
    ax2 = plt.subplot2grid((12, 12), (3, 0), colspan=7, rowspan=2)
    ax3 = plt.subplot2grid((12, 12), (5, 0), colspan=7, rowspan=2)
    ax4 = plt.subplot2grid((12, 12), (7, 0), colspan=7, rowspan=2)
    ax5 = plt.subplot2grid((12, 12), (0, 8), colspan=4, rowspan=4)
    ax6 = plt.subplot2grid((12, 12), (4, 8), colspan=4, rowspan=4)
    ax7 = plt.subplot2grid((12, 12), (8, 8), colspan=4, rowspan=4)
    ax8 = plt.subplot2grid((12, 12), (9, 2), colspan=4, rowspan=3)

    # p_txt = fig.text(0.35,.89,f.split('/')[-1], fontsize = 16, ha='center')
    if title:
        fig.suptitle(title, fontsize=18,fontweight='bold')

    t = np.arange(len(vcg1['VCGx']))

    ax1.plot(t, vcg1['VCGx'], 'k-')
    if ref_lines: ax1.axhline(y=0.0, xmin=0.0, xmax=1.0, color='0.5', linewidth='1.0')
    ax1.set_title('X',x=0,y=0.8, fontsize=16)
    ax1.axis('off')

    ax2.plot(t, vcg1['VCGy'], 'k-')
    if ref_lines: ax2.axhline(y=0.0, xmin=0.0, xmax=1.0, color='0.5', linewidth='1.0')
    ax2.set_title('Y',x=0,y=0.8, fontsize=16)
    ax2.axis('off')

    ax3.plot(t, vcg1['VCGz'], 'k-')
    if ref_lines: ax3.axhline(y=0.0, xmin=0.0, xmax=1.0, color='0.5', linewidth='1.0')
    ax3.set_title('Z', x=0, y=0.8, fontsize=16)
    ax3.axis('off')

    ax4.plot(t, np.linalg.norm(vcg1, axis=1), 'k-', label=labels[0])
    ax4.set_title('Mag.', x=-0.05, y=0.5, fontsize=16)
    ax4.set_xlabel('Time (ms)', fontsize=16)
    # ax4.axis('off')
    ax4.spines['right'].set_color('none')
    ax4.spines['left'].set_color('none')
    ax4.axes.get_yaxis().set_visible(False)
    ax4.spines['top'].set_color('none')
    ax4.set_facecolor('w')

    lines = []  # gather lines to add arrows if needed
    lines.append(ax5.plot(vcg1['VCGx'], vcg1['VCGz']*-1, 'k-'))
    ax5.set_title('Transverse', fontsize=16)
    ax5.axis('equal')
    ax5.set_facecolor('w')

    lines.append(ax6.plot(vcg1['VCGx'], vcg1['VCGy']*-1, 'k-'))
    ax6.set_title('Frontal', fontsize=16)
    ax6.axis('equal')
    ax6.set_facecolor('w')

    lines.append(ax7.plot(vcg1['VCGz']*-1, vcg1['VCGy']*-1, 'k-'))
    ax7.set_title('Left Sagittal', fontsize=16)
    ax7.axis('equal')
    ax7.set_facecolor('w')

    for ax in [ax5, ax6, ax7]:
        for axis in ['top', 'right']:
            ax.spines[axis].set_color('none')
        for axis in ['left', 'bottom']:
            if ref_lines:
                ax.spines[axis].set_position('center')
                ax.spines[axis].set_color('0.5')
                ax.spines[axis].set_linewidth(1.0)
                ax.set_yticks([])
                ax.set_xticks([])
                # plt.text(axmax,0-0.05,'X', color='0.5',size='xx-small')
                # plt.text(0-0.05,-axmax-0.11,'Y', color='0.5',size='xx-small')
            else:
                ax.spines[axis].set_color('none')
                ax.set_yticks([])
                ax.set_xticks([])

    ax8.axis('off')

    if alt_lines:
        line_style = alt_lines
    else:
        line_style = ['b-','r-']

    if isinstance(vcg2, pd.DataFrame):
        t_vcg = np.arange(len(vcg2['VCGx']))
        ax1.plot(t_vcg, vcg2['VCGx'], line_style[0])
        ax2.plot(t_vcg, vcg2['VCGy'], line_style[0])
        ax3.plot(t_vcg, vcg2['VCGz'], line_style[0])
        ax4.plot(t_vcg, np.linalg.norm(vcg2, axis=1), line_style[0], label=labels[1])
        lines.append(ax5.plot(vcg2['VCGx'], vcg2['VCGz']*-1, line_style[0]))
        lines.append(ax6.plot(vcg2['VCGx'], vcg2['VCGy']*-1, line_style[0]))
        lines.append(ax7.plot(vcg2['VCGz']*-1, vcg2['VCGy']*-1, line_style[0]))
    if isinstance(vcg3, pd.DataFrame):
        t_vcg = np.arange(len(vcg3['VCGx']))
        ax1.plot(t_vcg, vcg3['VCGx'], line_style[1])
        ax2.plot(t_vcg, vcg3['VCGy'], line_style[1])
        ax3.plot(t_vcg, vcg3['VCGz'], line_style[1])
        ax4.plot(t_vcg, np.linalg.norm(vcg3, axis=1), line_style[1], label=labels[2])
        lines.append(ax5.plot(vcg3['VCGx'], vcg3['VCGz']*-1, line_style[1]))
        lines.append(ax6.plot(vcg3['VCGx'], vcg3['VCGy']*-1, line_style[1]))
        lines.append(ax7.plot(vcg3['VCGz']*-1, vcg3['VCGy']*-1, line_style[1]))

    ax4.legend(prop={'size': 16}, frameon=False, bbox_to_anchor=(1.0, -0.5))

    if arrows:
        tspace = int(20)
        freq = tspace  # assumes 1ms/sample
        for line in lines:
            add_arrow(line[0], frequency=freq, size=20)


def add_arrow(line, frequency=5, size=20, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    frequency:  number of data points between arrows
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    num_arrows = len(xdata)//frequency
    for rows in range(1, num_arrows):
        start_ind = rows*frequency
        line.axes.annotate('',
                           xytext=(xdata[start_ind], ydata[start_ind]),
                           xy=(xdata[start_ind+1], ydata[start_ind+1]),
                           arrowprops=dict(arrowstyle="->", color=color),
                           size=size
                           )


def prol2cart(lambd, mu, theta, d):
    '''
    Convert prolate spheroidal coordinates into cartesian
    '''
    
    x = d*np.cosh(lambd)*np.cos(mu)
    y = d*np.sinh(lambd)*np.sin(mu)*np.cos(theta)
    z = d*np.sinh(lambd)*np.sin(mu)*np.sin(theta)
    return x, y, z


def cart2prol(x, y, z, d):
    '''
    convert cartesian coordinates into prolate spheroidal
    calculated using wolfram lab (https://lab.open.wolframcloud.com)
    >>> CoordinateTransformData["Cartesian"->"ProlateSpheroidal","Mapping",{x,y,z}]
    note wolfram has a different ordering from Hunter. Wolfram->Hunter =  {x,y,z}->{y,z,x}

    '''

    lambd = np.arccosh(np.sqrt(d**2+y**2+z**2+x**2+np.sqrt(y**2+z**2+(-d+x)**2)*np.sqrt(y**2+z**2+(d+x)**2))/(np.sqrt(2)*d))
    mu = np.pi/2-np.arctan(np.sqrt(2)*x*np.sqrt(1/(d**2+y**2+z**2-x**2+np.sqrt((y**2+z**2+(d-x)**2)*(y**2+z**2+(d+x)**2)))))
    theta = np.arctan2(z, y)

    return lambd, mu, theta

def prolateHammer(mu, theta):
    '''
    Calculate hammer projectsion from prolate spheroidal coordinates
    '''

    k = (1+np.cos(mu-np.pi/2)*np.cos((theta-np.pi)/2))**(-0.5)
    x = -k*np.cos(mu-np.pi/2)*np.sin((theta-np.pi)/2)
    y = k*np.sin(mu-np.pi/2)
    return x, y



if __name__ == "__main__":

    from sklearn.decomposition import PCA
    import seaborn as sns; sns.set()
    import sys
    
    sys.path.insert(-1,'/Users/victoriavincent/Dropbox/VCG_MachineLearning/Results_and_analysis/Tools')
    from data_parser_new import parse
    
    data_path = '/Users/victoriavincent/Dropbox/VCG_MachineLearning/Data/New_Simulations/'
    patients = [
            parse(initial_path=data_path, patient_number=patient_no, at_map = True, verbose = False) 
            for patient_no in range(0,8)
        ]
    # Concatinate data
    for patient in patients:
        vcg_list = [pd.concat([x['VCGx'],x['VCGy'],x['VCGz']],axis=0,ignore_index=True).to_frame(name='VCG_%i'%(i+1)) for i,x in enumerate(patient['vcg_model']) if isinstance(x, pd.core.frame.DataFrame)]
        patient['vcg_df'] = pd.concat([x for x in vcg_list],axis=1)
    
    all_vcg = pd.concat([patient['vcg_df'].add_prefix(patient['pt_id']+'_') for patient in patients],axis=1)
    
    print(all_vcg.shape)
    failed = all_vcg.columns[~all_vcg.any()]
    print(failed)
    all_at = all_vcg.drop(failed,axis=1,)
    print(all_vcg.shape)
    X_train = all_vcg.to_numpy().transpose()
    print(X_train.shape)
    
    pca = PCA(n_components=20)
    pca.fit(X_train) # fit trains the PCA
    eigvals = pca.singular_values_
    components = np.array(pca.components_)
    
    ncomp = components.shape[0]
    nfeat = components.shape[1]
    X_variance = np.cumsum(pca.explained_variance_ratio_)
    component_number = np.arange(len(X_variance)) + 1

    X_weights = pca.transform(X_train)    
    mean = pca.mean_
    df_mean = df_vcg_vector(mean)
    
    comps =[0]
    s = np.array([1.0])
    d = s*(np.sqrt(eigvals[comps]).T)
    recon_VCG = pca.mean_ + np.dot(d,components[comps])
    df_recon_VCG = df_vcg_vector(recon_VCG)
    
    c = calc_cost(df_mean,df_recon_VCG)
    c_rms = calc_cost_rms(df_mean,df_recon_VCG)
    c_rms_norm = calc_cost_rms_norm(df_mean,df_recon_VCG)
    