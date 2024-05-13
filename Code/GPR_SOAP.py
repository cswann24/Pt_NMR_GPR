

# import necessary libraries

import numpy as np
import os
import pandas as pd
from dscribe.descriptors import SOAP
from rdkit.Chem import AllChem
from ase import Atoms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Exponentiation
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit, learning_curve
from sklearn.preprocessing import Normalizer

# set up class for generating, reading and using SOAP descriptors as input for polynomial GPR

class SOAP_GPR:
    def __init__(self, SOAP_parameters, SOAP_directory = None, XYZ_directory = None, XYZ_base = None, central_atom = None):

        """
        Initialize the SOAP_GPR class for generating and reading SOAPs and evaluating
        polynomial kernel GPR for Pt chemical shift prediction

        :param SOAP_parameters: List containing SOAP parameters (r_cut, n_max, l_max)
        :param SOAP_directory: Directory to store SOAP descriptors
        :param XYZ_directory: Directory containing XYZ files of each sample of the dataset
        :param XYZ_base: Base filename for XYZ files
        :param central_atom: Atomic symbol of the central atom (Pt)
        """

        self.central_atom = central_atom
        self.SOAP_directory = SOAP_directory
        self.SOAP_parameters = SOAP_parameters
        self.XYZ_directory = XYZ_directory
        self.XYZ_base = XYZ_base


# function for generating SOAPs from XYZ files
    def generate_SOAPs(self):

        """
        Generate SOAP descriptors from every XYZ file of the dataset stored in XYZ_directory
        SOAP descriptors for every structure are stored as array

        :return:
        N x P array of SOAPs for every sample in the dataset (N = number of samples, P = length of SOAP array)
        """

        xyz_path = self.XYZ_directory

        xyz_filenames = sorted(os.listdir(xyz_path), key=lambda x: int(x.replace(self.XYZ_base, '').split('.')[0]))

        set_of_species = set()

        for xyz_filename in xyz_filenames:
            xyz_file_path = os.path.join(xyz_path, xyz_filename)

            try:

                if os.path.getsize(xyz_file_path) == 0:
                    raise Warning(f'XYZ file {xyz_filename} is empty')

                with open(xyz_file_path, 'r') as xyz_file:
                    lines = xyz_file.readlines()[2:]

                    for line in lines:
                        line_elements = line.split()

                        if line_elements:
                            set_of_species.add(line_elements[0])

            except Exception as e:
                print(e)

                pass

        species = list(set_of_species)
        print('Species present in dataset:', species)

    # Setting up SOAPs with DScribe library
        SOAP_dataset = []

        for xyz_filename in xyz_filenames:

            xyz_file_path = os.path.join(xyz_path, xyz_filename)

            try:
                mol = AllChem.MolFromXYZFile(xyz_file_path)
                central_atom_index = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == self.central_atom]
                atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
                atom_positions = mol.GetConformer().GetPositions()

                atoms = Atoms(symbols=atom_symbols, positions=atom_positions)


                soap = SOAP(
                    species=species,
                    periodic=False,
                    r_cut=float(self.SOAP_parameters[0]),
                    n_max=int(self.SOAP_parameters[1]),
                    l_max=int(self.SOAP_parameters[2])
                )

                soap_power_spectrum = soap.create(atoms, centers=central_atom_index)

                SOAP_dataset.append(soap_power_spectrum.flatten())

                descriptor_path = os.path.join(self.SOAP_directory,
                                f'r{self.SOAP_parameters[0]}_n{self.SOAP_parameters[1]}_l{self.SOAP_parameters[2]}/')

                os.makedirs(descriptor_path, exist_ok=True)
                SOAP_file = f'{int(xyz_filename.replace(self.XYZ_base, "").split(".")[0])}'
                np.savetxt(f'{descriptor_path}{SOAP_file}.txt', soap_power_spectrum)

            except Exception as e:
                print(e)

                pass

        return np.array(SOAP_dataset)


# read generated SOAPs saved as txt
    def read_SOAPs(self):

        """
        Read generated SOAPs for all samples from SOAP_directory

        :return:
         N x P array of SOAPs for every sample in the dataset (N = number of samples, P = length of SOAP array)
        """

        descriptor_path = os.path.join(self.SOAP_directory,
                        f'r{self.SOAP_parameters[0]}_n{self.SOAP_parameters[1]}_l{self.SOAP_parameters[2]}/')

        SOAP_dataset = []

        SOAP_filenames = sorted(os.listdir(descriptor_path), key=lambda x: int(x.split('.')[0]))
        SOAP_memory = 0
        file_count = 0

        for SOAP_filename in SOAP_filenames:
            try:
                SOAP_file = os.path.join(descriptor_path, SOAP_filename)
                SOAP_array = np.loadtxt(SOAP_file)
                SOAP_dataset.append(SOAP_array)
                SOAP_memory += os.path.getsize(SOAP_file)

                file_count += 1

            except os.path.getsize(SOAP_file) == 0:
                raise Warning(f'File No. {file_count} is empty.')

                pass


        print(f'SOAP files read: {len(SOAP_filenames)} \nAverage size: {round((SOAP_memory / file_count) / 1024, 3)} kB')

        return SOAP_dataset

# function for polynomial GPR taking SOAPs and response variable (chemical shifts) as input
    def predict(self, mode, regressor, kernel_degree, target_path, target_name, alpha, normalization,
                lc=None):

        """
        Use SOAPs to do predictions with different regression models and evaluate performance using
        cross-validated errors and learning curves

        :param mode: either 'read' for using already existing SOAPs or 'write' for generating new SOAPS to use in GPR
        :param regressor: either use 'KRR' for kernel ridge regression or 'GPR' for Gaussian process regression (sklearn)
        :param kernel_degree: integer number to specify degree of polynomial kernel
        :param target_path: path to csv with target data (chemical shifts)
        :param target_name: name of the column containing chemical shift values
        :param alpha: regularization parameter (sigma²_n)
        :param normalization: whether to normalize SOAPs before use (boolean)
        :param lc: whether to display learning curves (boolean)

        :return:
        Tuple of mean absolute error, standard deviation of mean absolute error,
        mean root mean squared error, standard deviation of root mean squared error
        """

        if mode == 'read':

            X_data = self.read_SOAPs()
            print('Dimension of data matrix:', np.shape(X_data))

        elif mode == 'write':

            X_data = self.generate_SOAPs()
            print('Dimension of data matrix:', np.shape(X_data))

        else:

            raise Exception('mode has to be specified as "read" for reading \n '
                            'already existing SOAPs or "write" for generating SOAPs.')

        target_data = pd.read_csv(f'{target_path}.csv')[str(target_name)]


        if normalization is False:

            pass

        else:
            X_data = Normalizer(norm='l2').fit_transform(X_data)

        randomSeed = 42
        train_X, test_X, train_target, test_target \
            = train_test_split(X_data, target_data, random_state=randomSeed, test_size=0.25, shuffle=True)


        if regressor == 'GPR':


            if kernel_degree == 1:
                estimator = GaussianProcessRegressor(kernel=DotProduct(),
                                                     random_state=randomSeed, alpha=float(alpha), optimizer=None)


            elif kernel_degree > 1:
                estimator = GaussianProcessRegressor(kernel=Exponentiation(DotProduct(), int(kernel_degree)),
                                               random_state=randomSeed, alpha=float(alpha), optimizer=None)

            else:
                raise Exception('Degree of polynomial kernel has to be specified.')

        elif regressor == 'KRR':

            if kernel_degree == 1:
                estimator = KernelRidge(kernel='linear', degree=1, alpha=float(alpha))

            elif kernel_degree > 1:
                estimator = KernelRidge(kernel=Exponentiation(DotProduct(), int(kernel_degree)), alpha=float(alpha))

            else:
                raise Exception('Degree of polynomial kernel has to be specified.')


        else:
            raise Exception('Regressor type has to be specified. \n '
                            '"GPR" for Gaussian Process Regression or "KRR" for Kernel Ridge Regression')


        estimator.fit(train_X, train_target)

        cv = KFold(n_splits=4, random_state=randomSeed, shuffle=True)
        model = estimator

        scores_rmse = cross_val_score(model, X_data, target_data, scoring='neg_root_mean_squared_error',
                                      cv=cv, n_jobs=-1)

        print('--------------------------------\nCross-validated error values:\n--------------------------------')
        print('RMSE (4-fold CV):')
        print(np.mean(np.absolute(scores_rmse)), '[ppm]')
        print(np.std(np.absolute(scores_rmse)), '[ppm]  (STDEV)')

        print('\n')

        scores_mae = cross_val_score(model, X_data, target_data, scoring='neg_mean_absolute_error',
                                     cv=cv, n_jobs=-1)
        print('MAE (4-fold CV):')
        print(np.mean(abs(scores_mae)), '[ppm]')
        print(np.std(abs(scores_mae)), '[ppm],  (STDEV)')
        print('--------------------------------')


        if lc is True:

            mae_list = []
            split_list = []

            for split in range(10, 100, 10):

                train_X, test_X, train_target, test_target \
                    = train_test_split(X_data, target_data, random_state=randomSeed, test_size=1-(split/100), shuffle=True)

                estimator.fit(train_X, train_target)
                prediction = estimator.predict(test_X)

                mae = mean_absolute_error(test_target, prediction)
                mae_list.append(mae)

                split_list.append(split)

            plt.scatter(split_list, mae_list)
            plt.plot(split_list, mae_list)
            plt.xlabel('Training data [%]')
            plt.ylabel('MAE [ppm]')
            plt.title(f'Learning Curve for Polynomial {regressor} with degree={kernel_degree} \n '
                     f'($r_{{cut}}$={float(self.SOAP_parameters[0])}, $n_{{max}}$={self.SOAP_parameters[1]}, '
                      f'$l_{{max}}$={self.SOAP_parameters[2]})')
            plt.show()

            fig, ax = plt.subplots()


            lc_plot_params = {
            "X": X_data,
            "y": target_data,
            "train_sizes": np.linspace(0.25, 1.0, 5),
            "cv": ShuffleSplit(n_splits=4, test_size=0.25, random_state=42),
            "score_type": "both",
            "n_jobs": -1,
            "line_kw": {"marker": "o"},
            "std_display_style": "fill_between",
            "score_name": "Correlation ($R^{2}$)",
            "scoring": "r2"
            }

            LearningCurveDisplay.from_estimator(estimator, **lc_plot_params, ax=ax)
            #train_sizes, train_scores, test_scores = learning_curve(estimator, X_data, target_data, train_sizes = np.linspace(0.25, 1, 5), cv=ShuffleSplit(n_splits=4, test_size=0.25, random_state=42), scoring='neg_mean_absolute_error')
            handles, label = ax.get_legend_handles_labels()
            ax.legend(handles[:2], ["Training Score", "Test Score"], loc='lower right')

            ax.set_title('SOAP', fontsize=16)
            ax.set_ylabel('$R^{2}$', fontsize=16)

            plt.show()

        else:
            pass


        return np.mean(np.abs(scores_mae)), np.std(np.abs(scores_mae)), np.mean(np.abs(scores_rmse)), np.std(
            np.abs(scores_rmse))


SOAP_directory = '/home/alex/ML/SOAP_GPR_NMR/final_dataset/SOAPs/'
XYZ_directory = '/home/alex/ML/SOAP_GPR_NMR/final_dataset/xyz_files_final_set/'

XYZ_base = 'st_'

target_name = 'Experimental'
target_path = '/home/alex/ML/SOAP_GPR_NMR/final_dataset/final_data_corrected'

SOAP_ML = SOAP_GPR(SOAP_parameters=[2.0, 3, 7], SOAP_directory=SOAP_directory, XYZ_directory=XYZ_directory,
                            XYZ_base=XYZ_base, central_atom='Pt')

errors_std = SOAP_ML.predict(mode='read', regressor='GPR', kernel_degree=2, target_path=target_path,
             target_name=target_name, alpha=1e-3, normalization=True)


