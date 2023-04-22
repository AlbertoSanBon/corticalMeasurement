# Systematic measuring cortical thickness in tibiae for bio-mechanical analysis

This repository is associated to the article published by Alberto Sánchez-Bonaste, Luis F. S. Merchante, Carlos Gónzalez-Bravo and Alberto Carnicero

# Abstract:

Measuring the thickness of cortical bone tissue is crucial for diagnosing bone diseases and monitoring treatment progress. One can perform this measurement visually from CT images or using semi-automatic algorithms from Houndsfield values. This article proposes a mechanism capable of measuring thickness over the entire bone surface, aligning and orienting all images in the same direction to reduce human intervention. The objective is to batch process large numbers of patients' CAT images to obtain thickness profiles of their cortical tissue for various applications.

To achieve this, classical morphological and segmentation techniques are used to extract the area of interest, filter and interpolate the bones, and detect their contour and Signed Distance Functions to measure cortical thickness. The set of bones is aligned by detecting their longitudinal direction, and the orientation is determined by computing their principal component of the center of mass slice.

Measuring cortical thickness would enable accurate traumatological surgeries and the study of structural properties. Obtaining thickness profiles of a vast number of patients can open the door to various studies aimed at identifying patterns between bone thickness and patients' medical, social, or demographic variables.


<p align="center">
  <img src="docs/images/paper.png" alt=""/>
</p>


# Data:

In the DATA folder, there are two sets of CT images used in the article to evaluate the performance of the code provided in this repository. The rest of the images cannot be shared due to confidentiality reasons. If any researcher is interested in replicating the article's results with the same dataset, they can proceed after signing the appropriate confidentiality agreements with the authors of the paper.

# Parameters description:

All the configuration is centralized in the file.ini in the "config" folder. That file is organized in sections:

#### dicom
This section configures all the paths:

- **data_path_dicom**. Folder with the CT images sets
- **output_path**. Folder to store STL files
- **resources_path**. Temporary folder to assist Visual Logging

#### pre-process

- **spacing**. Milimeters per slice and milimeters per pixel. Used to sample all the datasets at the same resolution. Default value: [0.5,0.25,0.25]
- **threshold**. Segmentation parameter. Set to 50 to extract the whole leg. To extract isolated bones, use higher values around 210, but it is more instable. It can be modified but first execution is recommended to be left to 50.
- **extract**. List of segmented elements IDs to extract. If empty, it extracts the largest element. In our case, the tibiae. 
- **size**. Main erosion and dilation kernel size. With spacing [0.5, 0.25, 0.25] it is set to 60. When the X and Y sampling rate is different, kernel size is updated authomatically. It can be provided but first execution is recommended not to be toched. If a different value is provided it will override the automated kernel size.
- **kernel_preerosion**. Smoothing and hole filling operators. It can be modified but first execution is recommended to be left to [1,1]
- **kernel_firstdilation**. Smoothing and hole filling operators. It can be modified but first execution is recommended to be left to [7,7]
- **kernel_firsterosion**. Smoothing and hole filling operators. It can be modified but first execution is recommended to be left to [6,6]

#### post-process

- **threshold_between_min**. Lower bound of the HU units to be filtered. It can be modified but first execution is recommended to be left to 250 for cortical detection
- **threshold_between_max**. Upper bound of the HU units to be filtered. It rarely needs to be modified from 2000
- **convert_stl**. Boolean variable that indicates if STL needs to be generated or not.

#### thickness

- **num_views_thickness**. If 1D thickness permiter profiles are desired, this variables set the number of those profiles. The main profile is captured in the Center of Mass, and the rest are extracted equidistantly from it.

#### all dicom

- **reference_bone**. It the CT bones are desired to be aligned and oriented against a reference bone, this variable sets the PATH to its STL file

#### reference vectors

- **orientation_vector**. This parameter is filled out after the execution of the script referencBone.py. Only two coordinates (X and Y) are required. The rest of the bones will orient against this reference. User doesn't need to provide a value
- **alignment_vector** . This parameter is filled out after the execution of the script referencBone.py. Three spacial coordinates are required (X,Y,Z). The rest of the bones will align against this reference. For instance [0,0,1] to be aligned against the Z axis. User doesn't need to provide a value

#### retake (apply corrections)

- **legX**. The X of this parameter is the ID assiged by the application and can be retrived from the log file and it indentifies the bone to be re-executed to apply some corrections. It indicates two possible corrections. Its value is a tuple of two binary values meaning no correction (0) or correction required (1). The first element of the tuple is for changing the leg side (if the reference bone is from the left side and the bone to be computed is from the right side). The second element is to fix the PCA main variance direction (check paper for more details). For example, if the bone 4 needs to be applyed the side correction but not the PCA correction, then this parameter would look like this: 

	leg4 = 1,0


# Api Reference

A description of the implemented methods can be found in README file from "docs" folder. 

# User guide:

A user guide can be found in USERGuide file from "docs" folder. 

# Replication:

To replicate the results provided in the article, run those steps:

1. Clone repository
2. Use this configuration file updating the paths according to the cloning path:

		[dicom]
		data_path_dicom = C:/corticalMeasurement/data/
		output_path = C:/corticalMeasurement/output/
		resources_path = C:/corticalMeasurement/resources/

		[pre-process]
		spacing = [0.5,0.25,0.25]
		threshold = 50
		extract = []
		size = 60
		kernel_preerosion = [1,1]
		kernel_firstdilation = [7,7]
		kernel_firsterosion = [6,6]

		[post-process]
		threshold_between_min = 250
		threshold_between_max = 2000
		convert_stl = True

		[thickness]
		num_views_thickness = 9

		[all dicom]
		reference_bone = C:/corticalMeasurement/data/TAC A

		[reference vectors]
		orientation_vector = 
		alignment_vector = 
		
		[retake]
		

4. Run PYTHON code:

The folder structure that we should have is the following.

<p align="center">
  <img src="docs/images/Structure.png" alt=""/>
</p>

To run PYTHON code without errors, we must first install the libraries stored in the requirements.txt

<p align="center">
  <img src="docs/images/InstRequirements.png" alt=""/>
</p>

Run the first script --> generateSTLs.py

<p align="center">
  <img src="docs/images/script_st.png" alt=""/>
</p>

Before continuing, establish in file.ini the path to the stl of the reference_bone.

Example: reference_bone = D:/corticalMeasurement/output/leg1.stl

Run the second script --> referenceBone.py

<p align="center">
  <img src="docs/images/script_nd.png" alt=""/>
</p>

Run the third script --> correctionsThickness.py

<p align="center">
  <img src="docs/images/script_rd.png" alt=""/>
</p>

After execution, check the log file correctionsThickness.html and add to the retake section of the file.ini, if needed, the necessary values for the retake.

Example: leg2=1.0

Run the fourth script --> correctionsThickness.py

<p align="center">
  <img src="docs/images/script_th.png" alt=""/>
</p>

6. Review LOG file


# Dependences:
See requirements.txt for tested library versions

### Visual logging dependences:
pip install visual-logging
pip install vg
