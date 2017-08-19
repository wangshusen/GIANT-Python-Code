# GIANT in Python

Globally Improved Approximate NewTon (GIANT) Method


## Demos

### 1. Prepare: download and process data

Download the "Year Prediction Million Song Dataset":

>- go to the directory "Resource/"
>- Linux: bash LinuxDownloadData.sh
>- Mac: bash MacDownloadData.sh
>- Now you have "YearPredictionMSD" and "covtype" in "./Resource/"

Convert the ".txt" files to ".npz" files:
>- cd Resource/
>- python txt2npz.py
>- Now you have "YearPredictionMSD.npz" and "covtype.npz" in "./Resource/"

Optional: generate synthetic data:
>- cd Resource/
>- python toydata.py
>- Now you have "N8.npz" in "./Resource/"


### 2. GIANT for Ridge Regression 

>- Edit "./Algorithm/Solver.py"
>- Make sure to use "from Algorithm.ExecutorQuadratic import Executor" 
>- cd ExperimentQuadratic
>- python demo.py


### 3. GIANT for Logistic Regression 

>- Edit "./Algorithm/Solver.py"
>- Make sure to use "from Algorithm.ExecutorLogistic import Executor" 
>- cd ExperimentLogistic
>- python demo.py
>- The results will be saved to "./Output/"


### 4. Generate Random Fourier Features

>- cd Resource/
>- Edit "./Resource/rfm.py" to adjust the data name
>- python rfm.py
>- Now you have "rfm_covtype.npz" OR "rfm_YearPredictionMSD.npz" in "./Resource/"



