README — MOSFET SPICE Parameter Extraction with Physics-Informed MLP 

This project uses neural networks to predict SPICE parameters  
of MOSFETs from simulated electrical curves.  
It contains two main notebooks :  
    1. mosfet_surrogate.ipynb  
    2. mosfet_mlp.ipynb  
The first one trains a forward (“surrogate”) model that reconstructs  
electrical curves from SPICE parameters.  
The second one trains an inverse MLP that predicts SPICE parameters  
from electrical curves, with physical constraints on Vth,  
Idlin, Idsat and Gmmax.  

1. DATASET  

This version of the project uses a dataset of 3,000 samples.  
This dataset is a portion of the initial dataset of 50,000 samples.  
Each sample corresponds to a simulated MOSFET with :  
    - 101 gate-voltage Vg points ;  
    - 12 electrical-curve channels ;  
    - 11 associated SPICE parameters ;  
    - 4 threshold-voltage Vth values ;  
    - 7 Idlin, Idsat and Gmmax characteristics.  
The data are split reproducibly with SEED = 42 :  
    - 80 % training   : 2,400 samples ;  
    - 10 % validation :   300 samples ;  
    - 10 % testing    :   300 samples.  

2. INSTALLATION  

Preferably create a Python virtual environment.  
On Windows :  
    python -m venv mosfet_env  
    mosfet_env\Scripts\activate  
On Linux or macOS :  
    python3 -m venv mosfet_env  
    source mosfet_env/bin/activate  
Install the dependencies with :  
    pip install -r requirements.txt  
The main libraries are NumPy, Pandas, Scikit-learn,  
Matplotlib, Joblib and PyTorch.  
The project automatically uses CUDA if a compatible NVIDIA GPU is  
available. Otherwise, the notebooks are executed on CPU. [2] [4]  

3. REQUIRED DATA FILES  

The data files must be placed in the folder :  
    dataset_3000/  
Required files :  
    dataset_3000/mosfet_X_dataset_3000.npz  
        Contains the electrical curves under the key "X".  
        Expected shape :  
            (3000, 101, 12)  
    dataset_3000/Y_3000.csv  
        Contains the 11 reference SPICE parameters.  
        Format : semicolon (;) separator, without a header.  
        Expected shape :  
            (3000, 11)  
    dataset_3000/vth_extracted_3000.csv  
        Contains the four threshold-voltage Vth values for the  
        conditions Vbs = 0 V, -2 V, -6 V and -10 V.  
        Expected shape :  
            (3000, 4)  
    dataset_3000/currents_and_gmmax_extracted_3000.csv  
        Contains seven electrical characteristics :  
            idlin_0vbs  
            idlin_m2vbs  
            idlin_m6vbs  
            idlin_m10vbs  
            idsat_0vbs  
            idsat_m2vbs  
            max_dId_dVg_0vbs  
        Expected shape :  
            (3000, 7)  

4. PREDICTED SPICE PARAMETERS  

The 11 predicted SPICE parameters are :  
    vth0  
    k1  
    k2  
    nfactor  
    u0  
    ua  
    ub  
    a0  
    ags  
    uc  
    keta  
The parameters u0, ua, ub and uc are transformed using :  
    log10(abs(value))  
before normalization. The inverse transformation is applied during  
prediction evaluation. [2] [3]  

5. ELECTRICAL CURVES  

Each sample contains 12 electrical channels :  
    0.  logId_Vds_0p1_Vbs_0  
    1.  logId_Vds_0p1_Vbs_2  
    2.  logId_Vds_0p1_Vbs_6  
    3.  logId_Vds_0p1_Vbs_10  
    4.  Id_Vds_0p1_Vbs_0  
    5.  Id_Vds_0p1_Vbs_2  
    6.  Id_Vds_0p1_Vbs_6  
    7.  Id_Vds_0p1_Vbs_10  
    8.  Id_Vds_5p0_Vbs_0  
    9.  Id_Vds_5p0_Vbs_2  
    10. dId_dVg_Vds_0p1_Vbs_0  
    11. Vg  
The curves are normalized channel by channel in the [0, 1] range.  
Before being provided to the MLP, they are flattened in  
“channel-major” order :  
    (N, 101, 12) -> (N, 12, 101) -> (N, 1212)  
Each MLP input therefore contains 1,212 values. [2] [3]  

6. STEP 1 — SURROGATE TRAINING  

Open the notebook :  
    mosfet_surrogate.ipynb  
Run the cells in the following order :  
    1. Config  
    2. Model  
    3. Dataprep  
    4. Train  
    5. Test  
The surrogate learns the forward relationship :  
    Normalized SPICE parameters (11)  
        ->  
    Normalized electrical curves (12 × 101 = 1212)  
Its architecture is a dense network :  
    11 -> 512 -> 1024 -> 2048 -> 2048 -> 1212  
The notebook creates the folder :  
    mosfet_surrogate_3000_10_10_25C/  
The best model is saved under :  
    mosfet_surrogate_3000_10_10_25C/surrogate_best.pth  
This file is required to train and evaluate the inverse MLP.  
The test cell calculates the global MAE, RMSE and R² metrics, as well  
as errors per channel. It is recommended to check surrogate  
performance before moving on to the next step. [3]  

7. STEP 2 — INVERSE MLP TRAINING  

Open the notebook :  
    mosfet_mlp.ipynb  
Run the cells in the following order :  
    1. config.py  
    2. model_mlp.py  
    3. data_prep.py  
    4. surrogate_vth_loss.py  
    5. train.py  
    6. predict.py  
The MLP learns the inverse relationship :  
    Normalized electrical curves (1212)  
        ->  
    Normalized SPICE parameters (11)  
The model contains five hidden layers of 2,048 neurons with BatchNorm,  
ReLU and Dropout.  
The output is constrained by physical bounds applied in normalized  
space using a rescaled sigmoid. [2]  
The notebook creates the folder :  
    mosfet_mlp_3000_10_10_25C/  
The best model is saved under :  
    mosfet_mlp_3000_10_10_25C/mlp_best.pth  

8. LOSS FUNCTION  

The total MLP loss is composed of three terms :  
    Total Loss =  
        Parameter Loss  
      + LAMBDA_VTH × Vth Loss  
      + LAMBDA_CONDUCTANCE × Conductance Loss  
1. Parameter Loss  
A Huber Loss compares the predicted SPICE parameters with the  
reference parameters.  
2. Vth Loss  
The predicted parameters pass through the frozen surrogate to  
reconstruct the log(Id) curves.  
Differentiable linear interpolation enforces the condition :  
    logId(Vth) = -7  
for the four Vbs bias conditions.  
3. Conductance Loss  
The conductance loss compares the following characteristics :  
    - Idlin at Vg = 5 V for Vbs = 0, -2, -6 and -10 V ;  
    - Idsat at Vg = 5 V for Vbs = 0 and -2 V ;  
    - Gmmax, corresponding to the maximum of dId/dVg.  
The configured coefficients are :  
    LAMBDA_VTH = 0.285  
    LAMBDA_CONDUCTANCE = 0.285  
The surrogate is frozen : its weights are not updated during  
MLP training, but the gradient flows through the surrogate in order  
to optimize the MLP. [2]  

9. MLP EVALUATION  

The predict.py cell performs the following operations :  
    - loading mlp_best.pth ;  
    - predicting SPICE parameters on the test set ;  
    - denormalizing the parameters ;  
    - reconstructing curves with the surrogate ;  
    - calculating MAE, RMSE and relative errors ;  
    - extracting and comparing Vth ;  
    - comparing Idlin, Idsat and Gmmax ;  
    - generating “prediction vs. true value” plots.  
The results are saved in :  
    mosfet_mlp_3000_10_10_25C/  
Generated files :  
    Y_pred_test_mlp.npy  
    Y_true_test_mlp.npy  
    Vth_pred_test_mlp.npy  
    Vth_true_test_mlp.npy  
    Conductance_pred_test_mlp.npy  
    Conductance_true_test_mlp.npy  
Generated figures :  
    grouped_parameters_pred_vs_true.png  
    grouped_vth_pred_vs_true.png  
    grouped_currents_pred_vs_true.png  

10. TESTING AN INDIVIDUAL SAMPLE  

The predict_one_Xtest_sample.py cell allows you to visually test an  
individual sample from the test set.  
Modify the following variable :  
    sample_index = 100  
The index must be between 0 and 299.  
The cell displays :  
    - the predicted and actual SPICE parameters ;  
    - absolute and relative errors ;  
    - the reference curves ;  
    - the curves reconstructed by the surrogate.  
The predicted parameters are saved in :  
    parameters_Xtest_sample_<index>_mlp.txt  

11. QUICK START  

    1. Create and activate a Python virtual environment.  
    2. Install the dependencies :  
           pip install -r requirements.txt  
    3. Check that the dataset_3000/ folder contains the four required  
       data files.  
    4. Run mosfet_surrogate.ipynb :  
           Config -> Model -> Dataprep -> Train -> Test  
    5. Check for the presence of the file :  
           mosfet_surrogate_3000_10_10_25C/surrogate_best.pth  
    6. Run mosfet_mlp.ipynb :  
           config.py  
           model_mlp.py  
           data_prep.py  
           surrogate_vth_loss.py  
           train.py  
           predict.py  
    7. Review the results in :  
           mosfet_mlp_3000_10_10_25C/
