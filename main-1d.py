import numpy as np
import scipy
import pandas as pd

import matplotlib.pyplot as plt

import os, re, logging, argparse

# Constants
k_B = 0.0019872041  # kcal/mol/K

def get_bins(hist_min, hist_max, num_bins=200):
    bins = np.linspace(hist_min, hist_max, num_bins+1)
    mids = (bins[:-1] + bins[1:]) / 2
    return {'bins': bins, 'mids': mids, 'min': hist_min, 'max': hist_max, 'n': num_bins}

def guess_file_table(dir='.', mask='dist_(.*?).dat', k=None, T=None):
    """
    Generate the file table from the directory and mask on the fly
    :param dir: Directory to search for files
    :param mask: File search mask, e.g. dist_(.*?).dat
    :param k: force constant, corresponding to the harmonic restraint E = k/2 * (x - x0)^2
    :param T: temperature in Kelvin
    :return:
    """
    assert k is not None and T is not None, "Force constant and temperature are mandatory"

    # if mask is enclosed in '', remove it
    mask = mask.strip("'")
    files = [f for f in os.listdir(dir) if re.match(mask, f)]

    restraints = [float(re.match(mask, f).group(1)) for f in files]
    file_paths = [os.path.join(dir, f) for f in files]
    df = pd.DataFrame({'file': file_paths, 'R_0': restraints})
    df = df.sort_values(by='R_0')
    df['k'] = k
    df['T'] = T
    return df


def read_all_runs(file_table, bins, params):

    m = np.zeros((len(file_table), bins['n']))
    cols0 = [c-1 for c in [params.column,]]

    for i, file in enumerate(file_table['file']):
        m_file = np.loadtxt(file, comments=list(params.comment), skiprows=params.skip, usecols=cols0)
        m[i], _ = np.histogram(m_file, bins=bins['bins'])
    return m


def generate_funs_roux(m, dist, MID_BINS, k, Temperature, cyclic=False):
    RT = k_B * Temperature
    b1 = np.subtract.outer(np.array(dist), MID_BINS)  # For angles, it should use the cyclic distance!!!

    if cyclic:
        # apply minimum image convention
        b1 = b1 - 360 * np.rint(b1 / 360)

    b2 = np.einsum('i,ij->ij', k/2 / RT, b1**2)  # biasing potential
    B = np.exp(-b2)  # Boltzmann factor of the biasing potential

    NJ = m.sum(axis=1)  # NRj in the paper; total number of samples in each Umb.Sampl. run
    NI = m.sum(axis=0)  # NCi in the paper; total number of samples in each output bin


    def calc_vecs(f):
        """
        Calculate the partition functions, probability distribution, and error vector
        :param f: a vector of free energy displacements
        :return: a dictionary with results
        """

        F = np.exp(f / RT)  # Exponential of the free energy displacements divided by RT
        D = (NJ * F) @ B  # Partition functions, [1xI], eq. 6 in the paper
        P = NI / D          # Unnormalized and unbiased probability distribution, eq. 5 in the paper

        Fn = P @ B.T        # New vector F(=exp(f/RT)), computed from P, eq. 10 in the paper
        ER = -RT * np.log(Fn) - f   # Error vector, e1. 11 in the paper
        R2 = np.dot(ER, ER)   # Norm of the error vector
        return {'F': F, 'D': D, 'P': P, 'Fn': Fn, 'ER': ER, 'R2': R2, 'B': B}

    def calc_grads(f, v=None):
        if v is None:
            v = calc_vecs(f)
        dD = np.einsum('i,i,ij->ij', NJ, v['F']/RT, B)  # dD[j,i] = dD_i/df_j, eq. 20 in the paper
        dFn = -np.einsum('i,ji,mi->jm', NI / v['D']**2, B, dD) # dFn[j,i] = dFn_i/df_j, eq. 19 in the paper
        dER = -np.einsum('i,ij->ij', RT/v['Fn'], dFn) - np.eye(len(v['Fn'])) # dER[j,i] = dER_i/df_j, eq. 18 in the paper

        return {'dER': dER}

    def A_f(f, T=300):
        P = calc_vecs(f)['P']
        RT = k_B * T
        A = -RT * np.log(P)
        A -= np.min(A)
        return pd.DataFrame({'R': MID_BINS, 'P': P, 'A': A})

    ER_f = lambda f: calc_vecs(f)['ER']
    R2_f = lambda f: calc_vecs(f)['R2']

    grad_ER_f = lambda f: calc_grads(f)['dER']

    return {
        'ER':ER_f, 'grad_ER':grad_ER_f, 'R2':R2_f, 'A':A_f, 'B':B
    }


def NR_SVD(f_start, R2, ER, Jacobian, alpha=1, maxit=1000, THRESHOLD=1e-5, G_THRESHOLD=1e-10, k0=2, Cmax=1e9, base=1e3,
           DEBUG=False, DOWN=50):
    """
    Minimize R2(f) using the Newton-Raphson method with SVD
    :param f_start: initial guess for the free energy profile
    :param R2: function that returns the norm of the error vector
    :param ER: function that returns the error vector
    :param Jacobian: function that returns the Jacobian matrix
    :param alpha: initial step size
    :param maxit: maximum number of iterations
    :param THRESHOLD: convergence threshold for the norm of the error vector
    :param G_THRESHOLD: convergence threshold for the norm of the gradient
    :param k0: initial value for the parameter k
    :param Cmax: maximum value for the parameter C
    :param base: base value for the parameter C
    :param DEBUG: print debug information
    :param DOWN: downscale factor for the step size
    :return: a dictionary with the results
    """
    f = f_start.copy()
    bconv = False
    conv, conv_old = None, 1e8
    conv_hist = np.zeros(maxit)
    R2_hist = np.zeros(maxit)

    k_max = int(np.ceil(np.log(Cmax) / np.log(base)))
    k_min = 1
    base_min = 1.01

    for iter in range(maxit):
        if DEBUG:
            print(f'Starting iteration {iter + 1}')

        try:
            J = Jacobian(f)
            U, s, Vt = np.linalg.svd(J, full_matrices=False)

            if iter % DOWN == 0:
                if base > base_min:
                    base = np.sqrt(base)
                    k0 = 2
                    f -= np.mean(f)
                    if DEBUG:
                        print(f'Iteration {iter + 1}: Slow convergence; reduce base value {base:.3f}')

        except np.linalg.LinAlgError:
            print(f'SVD error at iteration {iter + 1}; stop')
            if base > base_min:
                if DEBUG:
                    print(f'SVD error at iteration {iter + 1}, restart with reduced base value {base:.3f}')
                base = np.sqrt(base)
                k0 = 2
                f = f_start.copy()
                conv_hist[iter] = conv_old
                R2_hist[iter] = R2(f)
                continue
            else:
                print(f'SVD error at iteration {iter + 1}; base value is below the lowest possible value; stop')
                break


        dp = 1 / s
        dp[dp / dp[0] > base ** k0] = 0      # http://www.wag.caltech.edu/publications/sup/pdf/341.pdf
        G_minus = Vt.T @ np.diag(dp) @ U.T

        h = -alpha * G_minus @ ER(f)
        f += h
        conv = np.dot(h, h) # The step size norm

        conv_hist[iter] = conv
        R2_hist[iter] = R2(f)

        if DEBUG:
            print(k0)

        if conv < conv_old:
            k0 = min(k0 + 1, k_max)
        if conv > THRESHOLD and conv > conv_old:
            k0 = max(k0 - 1, k_min)

        if conv < THRESHOLD and np.sum(ER(f - h) ** 2) > G_THRESHOLD:
            k_max = int(np.ceil(np.log(1e15) / np.log(base)))
            k0 = k_max

        bconv = conv < THRESHOLD and np.sum(ER(f - h) ** 2) < G_THRESHOLD and k0 >= k_max
        if bconv:
            break

        alpha = 1  # Reset alpha after the first iteration
        conv_old = conv

    return {
        'convergence': bconv,
        'iter': iter,
        'norm': conv,
        'optim': f,
        'conv_hist': conv_hist[:iter],
        'R2_hist': R2_hist[:iter],
        'k0': k0
    }

def parse_command_line():
    parser = argparse.ArgumentParser(description='Free Energy Profile Calculation using WHAM equations and Newton-Raphson optimization with SVD')

    parser.add_argument('--input', type=str, help='Input file with the list of files and restraints')
    # Possible ways to provide the table of input files:
    # 1. Provide the directory and the mask to search for files, as well as the force constant and temperature
    # This method is limited to a 1D case
    # --dir required only if --input=='guess'
    parser.add_argument('--mask', type=str, help='Mask to search for files', default='dist_(.*?).dat')
    parser.add_argument('--dir', type=str, help='Directory to search for files', default='.')
    parser.add_argument('--k', type=float, help='Force constant, as in E = k/2 * (R-R_0)**2. Units: kcal/mol/Unit**2, where Unit is the unit of the sampled param', default=None)
    parser.add_argument('--T', type=float, help='Temperature in Kelvin', default=None)

    # 2. Provide the table of files directly as a text file with columns:
    # 1D case: file, R1_0, k, T, where E = k/2 * (x - R1_0)^2
    # 2D case: file, R1_0, R2_0, k1, k2, T, where E = k1/2 * (x1 - R1_0)^2 + k2/2 * (x2 - R2_0)^2

    # Format of the files with the samples
    # there might be one or two columns
    parser.add_argument('--skip', type=int, help='Number of lines to skip in the samples files', default=0)
    parser.add_argument('--comment', type=str, help='Comment characters in the samples files', default='#')
    parser.add_argument('--column', type=int, help='Column with the samples', default=2)

    # Parameters for the WHAM equations
    parser.add_argument('--hist_min', type=float, help='Minimum value for the histogram')
    parser.add_argument('--hist_max', type=float, help='Maximum value for the histogram')
    parser.add_argument('--num_bins', type=int, help='Number of bins in the histogram', default=100)

    parser.add_argument('--maxit', type=int, help='Maximum number of iterations', default=1000)
    parser.add_argument('--threshold', type=float, help='Convergence threshold for the norm of the error vector', default=1e-20)

    parser.add_argument('--cyclic', action='store_true', help='Use cyclic distance for angles, assuming the angles are in degrees', default=False)

    return parser.parse_args()

if __name__ == '__main__':

    # Set up the logging
    logging.basicConfig(level=logging.INFO)

    cli = parse_command_line()

    bins = get_bins(hist_min=cli.hist_min, hist_max=cli.hist_max, num_bins=cli.num_bins)

    if cli.input:
        file_table = pd.read_csv(cli.file_table)
    else:
        file_table = guess_file_table(dir=cli.dir, mask=cli.mask, k=cli.k, T=cli.T)

    m = read_all_runs(file_table=file_table, bins=bins, params=cli)

    funs = generate_funs_roux(m, file_table['R_0'], bins['mids'], k=file_table['k'], Temperature=file_table['T'], cyclic=cli.cyclic)

    # for each consecutive pair of rows in m, find the first bin where the difference between the values in two rows becomes positive
    # Additional condition: The row i should be beyond the maximum, and row i+1 should be before the maximum
    est_f = np.zeros(m.shape[0])

    # Initialize f0 with zeros based on the number of rows in m
    f0 = np.zeros(m.shape[0])

    OUT_SVD = NR_SVD(f0, R2=funs['R2'], ER=funs['ER'], Jacobian=funs['grad_ER'], THRESHOLD=cli.threshold, maxit=cli.maxit, DEBUG=True)

    # Assuming 'A_f' returns a pandas DataFrame
    results = funs['A'](OUT_SVD['optim'])
    plt.figure()
    plt.plot(results['R'], results['A'], label='Free Energy Profile', color='red')

    plt.xlabel('Reaction Coordinate')
    plt.ylabel('Free Energy (kcal/mol)')
    plt.title('Free Energy Profile')
    plt.legend()
    plt.show()

