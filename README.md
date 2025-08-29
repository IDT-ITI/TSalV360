# TSalV360: A Method and Dataset for Text-driven Saliency Detection in 360-Degrees Videos

* This repository provides code, dataset usage instructions, and trained model from our paper **"TSalV360: A Method and Dataset for Text-driven Saliency Detection in 360-Degrees Videos"**, written by Ioannis Kontostathis, Evlampios Apostolidis and Vasileios Mezaris, and accepted for publication in the Proceedings of the IEEE Int. Conf. on Content-Based Multimedia Indexing (CBMI 2025), Dublin, Ireland, Oct. 2025.
* This software can be used to obtain our TSV360 dataset for training and objective evaluation of methods for text-driven saliency detection in 360-degrees video, and for training our TSalV360 method. After training, this method can be used to find the salient events in a 360-degrees video that are most relevant to a user-provided text, and generate the associated saliency maps for the frames of this video.
* This repository includes:
  * Installation instructions for cloning the repository.
  * Details on how we constructed the TSV360 dataset and how to download it.
  * Instructions on how to train and run the TSalV360 method for text-driven 360-degrees video saliency detection.
  * Other details (license, citation, acknowledgements)

## Installation
To generate the TSV360 dataset and/or use the TSalV360 method, first clone the repository:
```
git clone https://github.com/IDT-ITI/TSalV360
```
Create and activate Conda environment

```
conda env create -f env.yml
conda activate tsalv360
```
## A. The TSV360 dataset

TSV360 is a dataset for training and objective evaluation of text-driven 360-degrees video saliency detection methods.
It contains textual descriptions and the associated ground-truth saliency maps, for 160 videos (up to 60 seconds long) sourced from the VR-EyeTracking and Sports-360 benchmarking datasets. These datasets cover a wide and diverse range of 360-degrees visual content, including indoor and outdoor scenes, sports events, and short films.

We constructed the dataset as follows:

We utilized EquiRectangular Projection (ERP) frames and their corresponding ground-truth saliency maps from the original datasets. An algorithm processed these inputs to generate multiple 2D video segments, each centered on different events within the same panoramic scene. For each 2D segment, we extracted and assigned event-specific saliency maps derived from the original ground-truth data. Following, these 2D video segments passed through a state-of-the-art video-language model (LlaVA-Next-7B) to generate textual descriptions that capture the depicted events. Finally, we manually curated the generated content to validate and refine it, resulting in 160 videos in total. For each video, there are multiple triplets of ERP frames, saliency maps, and text descriptions, each corresponding to a different event.

To obtain the data of the TSV360 dataset:

*	Download the original videos from the VR-EyeTracking dataset, by following the instructions [here](https://github.com/xuyanyu-shh/VR-EyeTracking) or [here](https://github.com/mtliba/ATSal/tree/master). The subset of the videos that are included in our TSV360 dataset can be found [here](dataset/vreyetracking.json). To extract frames from these videos, please run the following command:
``` 
python dataset/frames_extractor.py --videos_path="path_to_videos"
```
* Download the frames of the videos from the Sports-360 dataset, by following the instructions [here](https://github.com/vhchuong/Saliency-prediction-for-360-degree-video/tree/main). The subset of the videos that are included in our TSV360 dataset, can be found [here](dataset/sports360.json).

* Download the textual descriptions and the associated ground-truth saliency maps of our TSV360 dataset from Zenodo (here)(zenodo link)

In case you have troubles accessing any of the above, please contant us at ioankont@iti.gr. 

## B. The TSalV360 method

This section provides details on how to setup the data of our TSV360 dataset in order to train and evaluate our TSalV360 method for text-driven 360-degrees video saliency detection.

### Dataset Preparation

After downloading the TSV360 dataset and extracting the frames from the utilized videos of the VR-EyeTracking dataset (following the instructions in Section **A. The TSV360 dataset**), place all folders and subfolders according to the following structure:

```Text
/Parent Directory
    /video_name_0
        /0000.jpg
        ...
    /video_name_1
        /0000.jpg
    ...
```

### Training stage
To train a model, please edit the [train.yml](configs/train.yml) file, by updating: a) `path_text_saliency_maps` to link to the generated "TSV360_gt" folder after unpacking the **TSV360_gr.rar** file from Zenodo, and b) `path_to_erp_frames` to link to the folder containing the subfolders with the frames of each video. Then, please run the following command:

```
python train.py
```

Training is performed following the 5-fold cross validation approach. After the end of the training stage in each of the 5 folds, the performance of the trained model is evaluated on the videos of the associated test set. A checkpoint is saved at the end of each fold inside the 'model' folder. The evaluation results, including metrics and losses, are recorded in a log file named evaluation_logs.txt, located in the TSalV360 directory.
### Inference stage

To use your own trained model — or the provided [pretrained TSalV360 model](https://drive.google.com/file/d/1oMyNRPtgtDMHkCpttPXaSyGj45CG8HS-/view?usp=sharing) — set the path to the model's checkpoint, the path to the ERP-formatted 360-degrees video, the input text, and the path to the folder where the generated saliency maps will be saved, as parameters to the following command:

```
python inference.py --model_path="path_to_model" --video_path="path_to_video" --text_input="text_description" --output_path="path_to_save_generated_saliency_maps"
```

The output consists of saliency maps for the video frames, saved as .png files in the folder specified above.

## Licence

This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Citation
If you find our work, code or trained models useful in your work, please cite the following publication:

I. Kontostathis, E. Apostolidis, V. Mezaris, "TSalV360: A Method and Dataset for Text-driven Saliency Detection in 360-Degrees Videos", IEEE Int. Conf. on Content-Based Multimedia Indexing (CBMI 2025), Special Session on Multimedia Indexing for XR (MmIXR), Dublin, Ireland, Oct. 2025.

BibTex:
````
@inproceedings{kontostathisTSalV360,
  title={TSalV360: A Method and Dataset for Text-driven Saliency Detection in 360-Degrees Videos},
  author={Kontostathis, Ioannis
    and Apostolidis, Evlampios
    and Mezaris, Vasileios},
  booktitle={2025 Int. Conf. on Content-Based Multimedia Indexing (CBMI)},
  year={2025},
  organization={IEEE}
}
````
## Acknowledgements
This work was supported by the EU's Horizon Europe programme under grant aggreement 101070109 TransMIXR.



