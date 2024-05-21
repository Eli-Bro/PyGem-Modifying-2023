import datetime

import dateutil.utils
from sklearn.preprocessing import StandardScaler
import xml.etree.ElementTree as ET
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import ttest_ind
import numpy as np
import itertools
from math import cos, radians, sin, hypot
from copy import copy
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
#import cv2
from scipy import interpolate
from subprocess import call
import re
import time
import seaborn as sns
import numpy as np
import csv
import glob
#import generate_int_csvs as gic
import PostProcess_FeBio as proc
import PCA_data
import pandas as pd
import re
import Bottom_Tissue_SA_Final as bts

current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)
Results_Folder = "D:\\Gordon\\Automate FEB Runs\\2024_5_6_NewModel"  # INTERMEDIATE CSV ENDS UP HERE
Target_Folder = "D:\\Gordon\\Automate FEB Runs\\2024_5_6_NewModel\\*.feb"  # LOOK HERE FOR THE FEB FILES
csv_filename = Results_Folder + '\\' + date_prefix + '_intermediate.csv'
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)

object_list = ['Object17']  # MAKE SURE THIS MATCHES THE OBJECTS IN THE CURRENTLY USED MODEL
obj_coords_list = []
file_num = 0
numCompPCA = 2

first_file_flag = True
GENERATE_INTERMEDIATE_FLAG = True
final_csv_flag = False


if GENERATE_INTERMEDIATE_FLAG:

    for feb_name in glob.glob(Target_Folder):

        int_log_name = feb_name.split(".f")
        int_log_name[1] = ".log"
        log_name = int_log_name[0]+int_log_name[1]

        csv_row = []

        # Get the pure file name that just has the material parameters
        file_params = int_log_name[0].split('\\')[-1]

        proc.generate_int_csvs(file_params, object_list, log_name, feb_name, first_file_flag, csv_filename)

        if first_file_flag:
            first_file_flag = False

        # sleep to give the file time to reach directory
        time.sleep(1)
        file_num += 1
        print(str(file_num) + ": " + file_params)
        obj_coords_list = []

if final_csv_flag:
    print('Generating PC File')
    proc.process_features(csv_filename, Results_Folder, date_prefix, numCompPCA)



# use the generated csv to get the 2 PC scores
#TODO: IGNORE THIS FUNCTION, USE THE ONE IN "PostProcess_FeBio"
def process_features(csv_file):
    int_df = pd.read_csv(csv_file)
    pc_df = int_df.iloc[:, 4:len(int_df.columns)]
    # int_df = pd.read_csv("intermediate_pc_data", header=None)
    total_result_PC, pca = PCA_data.PCA_(pc_df)

    PC_scores = total_result_PC[['principal component 1', 'principal component 2']]
    print(PC_scores)

    final_df = pd.concat([int_df.loc[:, ["File Name", "Apex", "E1", "E2"]], PC_scores], axis=1)
    final_df.to_csv(Results_Folder + '\\' + date_prefix + "_features.csv", index=False)