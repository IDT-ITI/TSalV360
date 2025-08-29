# TSalV360: A Method and Dataset for Text-driven Saliency Detection in 360-Degrees Videos

* This repository provides code and trained models from our paper "TSalV360: A Method and Dataset for Text-driven Saliency Detection in 360-Degrees Videos", by Ioannis Kontostathis, Evlampios Apostolidis and Vasileios Mezaris, accepted for publication in the Proceedings of the IEEE Int. Conf. on Multimedia Indexing for XR (CBMI 2025), Special Session on Multimedia Indexing for XR Dublin, Ireland, Oct. 2025.
* This software can be used to train our text-driven 360-degree video saliency detection model. We provide the TSV360 dataset, which was constructed for training and evaluating deep-learning models on the text-driven 360-degree video saliency task. Once trained, the model takes a text prompt and a 360-degree video in ERP frames as input, and generates saliency maps based on the given description.
* This repository includes:
  * Installation instructions for cloning the repository
  * Details on how we constructed the TSV360 dataset and how to download it
  * Instructions on how to train and run inference with the TSV360 model
  * Other details (license, citation, acknowledgements)

## Installation
To generate the TSV360 dataset and/or use the TSalV360 approach, first clone the repository:
```
git clone https://github.com/IDT-ITI/TSalV360
```
Create and activate Conda environment

```
conda env create -f env.yml
conda activate tsalv360
```
## A. TSV360 Dataset

TSV360 is a dataset for text-driven 360-degree video saliency detection.
It comprises 160 videos, each up to 60 seconds in duration, sourced from the benchmark saliency datasets VR-EyeTracking and Sports-360. The dataset covers a wide and diverse range of visual content, including indoor and outdoor scenes, sports events, and short films.

We constructed the dataset as follows:
We utilized EquiRectangular Projection (ERP) frames and their corresponding ground-truth saliency maps from the original datasets. An algorithm processed these inputs to generate multiple 2D video segments, each centered on different events within the same panoramic scene. For each 2D segment, we extracted and assigned event-specific saliency maps derived from the original ground-truth data. Following, these 2D video segments passed through a state-of-the-art video-language model (LlaVA-Next-7B) to generate textual descriptions that capture the depicted events.
Finally, we manually curated the generated content to validate and refine it, resulting in 160 videos in total. For each video, there are multiple triplets of ERP frames, saliency maps, and text descriptions, each corresponding to a different event.

To download the TSV360 dataset:

*	For the original videos from the VR-EyeTracking dataset, download them by following the instructions [here](https://github.com/xuyanyu-shh/VR-EyeTracking) or [here](https://github.com/mtliba/ATSal/tree/master). The list of videos from this dataset that are included in TSV360 dataset, can be found [here](dataset/vreyetracking.json). To extract frames from the VR-EyeTracking videos, run the following command:
``` 
python dataset/frames_extractor.py --videos_path="path_to_videos"
```
* For the original videos from the Sports-360 dataset, download them by following the instructions [here](https://github.com/vhchuong/Saliency-prediction-for-360-degree-video/tree/main). The list of videos from this dataset that are included in TSV360 dataset, can be found [here](dataset/sports360.json). The Sports-360 data are already provided as '.jpg' files (frame-level).

* The generated ground-truth saliency maps with the corresponding text descriptions can be found in the Zenodo link (here)(zenodo link)

If you are unable to download any of them, contant us at ioankont@iti.gr. 

## B. TSalV360 approach

This section implements TSalV360, a method for text-driven saliency detection in 360-degree videos. It includes the full TSalV360 architecture along with the setup required to train and evaluate the model.

### Dataset Preparation

Download the TSV360 dataset—which includes the generated ground-truth saliency maps, text descriptions, and corresponding ERP frames—by following the instructions in section A. TSV360 Dataset.
After extracting the frames from both the VR-EyeTracking and Sports-360 datasets, place all subfolders (named by video, each containing its respective frames) into a single folder.

### Training stage
To train a model, edit train.yml file [here](configs/train.yml) by updating the paths to the `path_text_saliency_maps` (path inside the TSV360_gt folder downloaded from Zenodo) and the folder containing the extracted frames for each video `path_to_erp_frames` (created in the above Dataset Preparation subsection), then run the following command:

```
python train.py
```

### Inference stage

To use your own trained model—or the provided [pretrained model](https://drive.google.com/file/d/1oMyNRPtgtDMHkCpttPXaSyGj45CG8HS-/view?usp=sharing)—pass the model checkpoint, the ERP-format 360-degree video path, and a corresponding text description as input parameters to the following command:

```
python inference.py --model_path="path_to_model" --video_path="path_to_video" --text_input="text_description"
```
## Licence

This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Citation
If you find our work, code or trained models useful in your work, please cite the following publication:

I. Kontostathis, E. Apostolidis, V. Mezaris, "TSalV360: A Method and Dataset for Text-driven Saliency Detection in 360-Degrees Videos", IEEE Int. Conf. on Content-Based Multimedia Indexing (CBMI 2025), Special Session on Multimedia Indexing for XR Dublin, Dublin, Ireland, Oct. 2025.

BibTex:
````
@misc{tsalv360,
  title={TSalV360: A Method and Dataset for Text-driven Saliency Detection in 360-Degrees Videos},
  author={Kontostathis, Ioannis
    and Apostolidis, Evlampios
    and Mezaris, Vasileios},
  note={Under review at CBMI 2025, MmIXR Special Session},
  year={2025}}
}
````
## Acknowledgements
This work was supported by the EU's Horizon Europe programme under grant aggreement 101070109 TransMIXR.



