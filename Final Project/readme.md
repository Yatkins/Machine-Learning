[Google Drive Link](https://drive.google.com/drive/folders/1Vsf_e1JLEnxz9k0yWecq_sbel0ohWPgV?usp=sharing)
Please run 'nnU-Net.ipynb' file under 'Colab Notebook' with 'nnUNet_raw_data' under 'Colab Notebook/nnUNet/nnunet/nnUNet_raw_data_base'
(Everything is already in place.)

Colab Notebook
 |- nnU_Net.ipynb : This is the file to execute
 |- nnUNet
 |  |- nnunet
 |     |- nnUNet_raw_data_base : Where input files are
 |     |   |- nnUNet_raw_data
 |     |   |   |- Task101_COVID
 |     |   |       |- imagesTr: Training Nitfi image dataset 
 |     |   |       |    ex) volume-covid19-A-0003_0000.nii.gz
 |     |   |       |    file name: 'volume-covid19-A-0003', modality: '_0000'
 |     |   |       | 
 |     |   |       |- labelsTr: Training Nitfi label dataset
 |     |   |       |    ex) volume-covid19-A-0003.nii.gz
 |     |   |       |    matching data name 'volume-covid19-A-0003' with files under 'imagesTr' folder
 |     |   |       | 
 |     |   |       |- imagesTs: Test Nitfi image dataset
 |     |   |       |    ex) volume-covid19-A-0004_0000.nii.gz
 |     |   |       |    file name: 'volume-covid19-A-0004', modality: '_0000'
 |     |   |       |
 |     |   |       |- dataset.json: dataset config generated during Jupyter Notebook session
 |     |   |
 |     |   |- nnUNet_cropped_data: preprocessed data generated during Jupyter Notebook session
 |     |
 |     |- nnUNet_trained_models
 |     |  |- nnUNet
 |     |     |- 3d_fullres
 |     |        |- Task101_COVID
 |     |           |- nnUNetTrainerV2__nnUNetPlansv2.1
 |     |              |- fold_0 : Trained with 0 fold
 |     |                 |- model_best.model : epoch 57 created the best model  
 |     |                 |- model_best.model.pkl
 |     |                 |
 |     |                 |- progress.png: loss function over epoch
 |     |                 |- network_architecture.pdf
 |     |
 |     |- nnUNet_Prediction_Results
 |        |- Task101_COVID: prediction files with test Nifti images
 |           ex) volume-covid19-A-0004.nii.gz
 |   
 |- apex : required library, you will install during Jupyter Notebook session
