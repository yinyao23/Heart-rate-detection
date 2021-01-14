# Video-Based Heart Rate Measurement Using 3D-Convolutional Attention Networks
Deep learning (neural network) based remote photoplethysmography: how to extract heart rate signal from video using ica-based tool and deep learning tools.

## How to Run the Implemented Pipelines

### 1. Generate the pre-processed files for the 3 pipelines.
(1) Download the MANHOB-HCI dataset under the "./dataset/" directory. For example, the path of one video file would be "./dataset/Sessions/Session-1/2/P1-Rec1-2009.07.09.17.53.46_C1 trigger_C_Section_2".

(2) Run the preprocess-for-hci-tagging-db.py to generate the following files: \\
	instan_hr_groundtruth_hci_tagging.json 
	mean_hr_groundtruth_hci_tagging.json
	peak_timestamps_groundtruth_hci_tagging.json
Note: This step can be skipped since the generated 3 files has already been uploaded.

(3) Run the corrensponding functions in construct_dataset.py for deep learning based methods.
	1D-CNN-based pipeline: construct_HCI_dataset_for_ica_method() && construct_HCI_dataset_for_1dcnn_model()
	3D-CNN-based pipeline: construct_HCI_dataset_for_3dcnn_model()
Note: This step can be skipped for ICA-based pipeline.

### 2. Run the corresponding python file
ICA-based pipeline: Run the ica-based-method.py.
1D-CNN-based pipeline: Run the 1dcnn-based-method.py
3D-CNN-based pipeline: Run the 3dcnn-based-method.py

# Abstract of the corresponding report
## titled "Video-Based Heart Rate Measurement Using 3D-Convolutional Attention Networks" 
Non-contact video-based physiological measure-ment has many applications in health care and human-computer interaction. Over the last few years, remoteheart rate measurement from facial video has gainedparticular attention.  Preliminary results demonstratethat this convolutional 3D network can effectively ex-tract pulse rate from video without the need for anyprocessing of frames. 
Thus, we propose a deep 3D-attention-convolutional network for video-based heartrate detection. The model is based on a skin reflectionmodel and attention mechanism, which utilizes appear-ance information to guide the rPPG signal extraction. Our approach significantly outperforms the ICA-basedpipeline on public MAHNOB-HCI data.

