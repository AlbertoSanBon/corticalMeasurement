Before starting, write in config/file.ini the following parameters:

	- data_path_dicom --> path to the directory with DICOM files.
	- reference_bone --> path to the reference bone.
	- output_path  --> path to the directory in which STL files will be save.
	- resources_path  --> path to the directory in which resources (png) will be save.
	- spacing --> resolution of the slices. Recommended: [0.5, 0.25, 0.25].
	- threshold --> threshold for bounding box. Recommended: 50.
	- extract --> choose between bones. Recommended: []
	- size --> Recommended: 60 for spacing [0.5, 0.25, 0.25]. 30 for spacing [0.5, 0.5, 0.5]. 15 for spacing [0.5, 0.75, 0.75]
	- kernel_preerosion --> kernel preerosion size. Recommended [1,1].
	- kernel_firstdilation --> dilation kernel size. Recommended [7,7].
	- kernel_firsterosion --> erosion kernel size. Recommended [6,6].
	- threshold_between_min --> lower limit of the range of UH in the STL. Recommended: 250.
	- threshold_between_max --> higher limit of the range of UH in the STL. Recommended: 2000.
	- convert_stl --> boolean to make or not the STL file. Recommended: True.
	- reference_bone --> path to the reference STL.
	- num_views_thickness --> number of thicknesses.

*Do not modify the spacing value in all the process*

1. Script: generateSTls.py --> Generate STL models from DICOM files.
	After the execution is finished, we will have in the directory set in output_path the STL models named as legX.stl.
	Establish the reference STL model in file.ini.

2. Script: referenceBone.py --> Generates reference vectors for future corrections and reference bone thickness profiles. 
	When the execution ends, the orientation and alignment reference vectors that will be used in future corrections appear in the file.ini.
	In addition, a log file (html) will be generated with the results of thickness for the reference bone.

3. Script: correctionsThickness.py --> Performs orientation and alignment corrections and generates thickness profiles.
	When the execution ends, a log file (html) will be generated. In this file the corrections and thickness profiles are represented by bone.

To continue, you have to review this log file bone by bone. Only if the bone in the orientation correction looks like it belongs to the opposite leg to the reference one, the change_leg parameter must be set to 1 or/and if the PCA component is towards the back of the bone, the parameter Correct_direction_manually must be set to 1.

These values must be written in file file.ini in section [retake] with the following structure: legX = value_change_leg, value_Correct_direction_manually.

Example --> if the bone 4 belongs to the opposite leg to the reference one but the PCA component is okey, the structure will be: leg4 = 1,0

4. Script: cT_Retake.py --> Performs re-orientation and re-alignment corrections and re-generates thickness profiles in selected bones.
