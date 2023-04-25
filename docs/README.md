# User Guide

**DISCLAIMER**. The code provided in this repository is only intended to share the results of the article it refers to. The code is not optimised for efficient use. Multiple blocks have been implemented in a simple way for ease of debugging, but not for runtime optimisation.

### 1. Clone repository:

<p align="center">
  <img src="images/step1.png" alt=""/>
</p>

### 2. Understand repo folders:

The repository is structured in 6 folders:

* config - contains configuration file
* data - contains two datasets to test the code
* docs - contains this guide and the API reference
* libs - contain the library required for the code to run
* logs - folder that will store the application traces
* src - folde that contains the python scripts

<p align="center">
  <img src="images/step2.png" alt=""/>
</p>

### 3. Setup configuration file:

Before starting, write in config/file.ini the following parameters. Visit the entry [README](../README.md) file for details of each parameter:

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

**IMPORTANT**. The spacing identifies the resampling of the input images. If after a first iteration a new execution needs to be done to improve results of some bones, spacing should not be changed. 

### 4. Understanding the Scripts:

#### generateSTls.py.
Generate STL models from DICOM files. After the execution is finished, we will have in the directory set in output_path the STL models named as legX.stl.


#### referenceBone.py.
Generates reference vectors (orientation_vector and alignment_vector) for the corrections of other bones. 
When the execution ends, the orientation and alignment reference vectors for future corrections appear in the file.ini. A log file (html) will be generated with the results of thickness for the reference bone.

#### correctionsThickness.py.
Performs orientation and alignment corrections and generates thickness profiles. When the execution ends, a log file (html) will be generated. In this file, the corrections and thickness profiles are represented by bone.
This log file needs to be reviewed for every bone to check if the leg side and the PCA direction have been properly detected. To assist with this task, the corrected version of each bone is rendered in 3D with the reference bone in the log file.
If the side of the leg was not properly identified, a new iteration must be run changing the change_leg parameter to If the PCA vector was chosen in the opposite direction, a new iteration must be run setting the parameter correct_direction_manually to 1.

Parameters change_leg and correct_direction_manually are set in section [retake] in the file.ini with the following structure: 

	legX = change_leg, correct_direction_manually

Example: if bone 4 belongs to the opposite leg to the reference one but the PCA component is correct, the file.ini setting will be: 

	leg4 = 1,0

#### cT_Retake.py.
Performs re-orientation and re-alignment corrections and re-generates thickness profiles in selected bones.


# Recommendations

* Make first run of the whole process at low resolution, for example:

	spacing = [1,1,1]
	
  This setting will dramatically improve the execution time. It will also allow identifying issues about the leg side or the PCA direction in a low resolution and fast execution.
 
 * Use the recommended settings of the file.ini configuration file. Even if the renderization of the bones in the logfile shows artefacts, the thickness measuring code only computes the thickness for the largest element of the slice. This element is frequently the bone section. Check the colored rendered version of the bone to identify if the thickness values belong to the desired element.
 
 


# API Reference

The document linked [here](API-Reference.md) describe the implemented methods, and their parameters.



