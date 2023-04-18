
# libraries


## load_scan
```python
load_scan(path, files)
```

Introduction
------------
Load all DICOM images from a folder and return a list with their metadata for manipulation

Parameters
----------
path : string
    Path to the directory with the dicom images.
files : list
    List with the images.

Returns
-------
slices : list.
    List with the metadata of the images.


## get_pixels_hu
```python
get_pixels_hu(scans)
```

Introduction
------------
Create a Numpy matrix of Hounsfield Units. The voxel values in the images are raw. get_pixels_hu converts raw values into Hounsfield Units (HU).
The transformation is linear. Therefore, so long as you have a slope and an intercept, you can rescale a voxel value to HU.
Both the rescale intercept and rescale slope are stored in the DICOM header at the time of image acquisition.

Parameters
----------
scans : list.
    List with the metadata of the images. The return of load_scan function.

Returns
-------
array of int16.
    Numpy matrix of Hounsfield Units (HU).


## sample_stack
```python
sample_stack(stack,
             rows=11,
             cols=10,
             start_with=1,
             show_every=7,
             color=False,
             cmap=None)
```

Introduction
------------
Show a subplot of images

Parameters
----------
stack : array
    Numpy array of dimensions (num, resolution_x, resolution_y).
rows : int, optional
    Number of rows of the subplot. The default is 11.
cols : int, optional
    Number of columns of the subplot. The default is 10.
start_with : int, optional
    Image with which the subplot starts. The default is 1.
show_every : int, optional
    The subplot shows one image every this number. The default is 7.
color : boolean, optional
    If is True, shows de color of the labels in the image. The default is False.
cmap : string, optional
    The default is None.

Returns
-------
None.


## resample
```python
resample(image, scan, new_spacing=[1, 1, 1])
```

Introduction
------------
Using the metadata from the DICOM we can figure out the size of each voxel as the slice thickness.
In order to display the CT in 3D isometric, and also to compare between different scans, it would be useful to ensure that each slice is resampled spacing[0]xspacing[1]xspacing[2] mm pixels and slices.

Parameters
----------
image : array of int16
    Numpy matrix with the Hounsfield Units.
scan : list
    List with the metadata of the images. The return of load_scan function.
new_spacing : array, optional
    spacing that will be applied. The default is [1,1,1].

Returns
-------
image : array
    Numpy matrix generated after doing the resample.
new_spacing : array
    The spacing applied.


## find_bounding_box
```python
find_bounding_box(img,
                  hu=True,
                  threshold=200,
                  display=True,
                  sizex=5,
                  sizey=5,
                  title=True,
                  linewidth=1)
```

Introduction
------------
Auto-detect the boundaries surrounding a volume of interest.

Parameters
----------
img : array
    One image of the numpy matrix generated after doing the resample.
hu : boolean, optional
    True if input image is RGB. The default is True.
threshold : int, optional
    Threshold parameter for the bounding box detection. The default is 200.
display : boolean, optional
    Plot or not the bounding box in the image. The default is True.

Returns
-------
(x,y) : tuple
    Coordinate x and y of the bounding box.
(x+w, y+h): tuple
    Coordinates X and Y of the bounding boxes.
w : float
    Width of the bounding box.
h : float
    Height of the bounding box.


## find_bounding_box_sample_stack
```python
find_bounding_box_sample_stack(img,
                               hu=True,
                               show_box=True,
                               threshold=200,
                               rows=10,
                               cols=10,
                               start_with=1,
                               show_every=3,
                               areamin=None)
```

Introduction
------------
Show a subplot of images with the bounding box detection.

Parameters
----------
img : array
    Numpy array to show their bounding box, of dimensions (num, resolution_x, resolution_y).
hu : boolean, optional
    True if input image is RGB. The default is True.
show_box : boolean, optional
    If it is True, plot full ressolution with bounding box. The default is True.
    If it is False, plot only content of bounidng box.
threshold : int, optional
    Threshold parameter of the bounding box detection. The default is 200.
rows : int, optional
    Number of rows of the subplot. The default is 10.
cols : int, optional
    Number of columns of the subplot. The default is 10.
start_with : int, optional
    Image with which the subplot starts. The default is 1.
show_every : int, optional
    The subplot shows one image every this number. The default is 3.

Returns
-------
None.


## make_bonesmask
```python
make_bonesmask(img,
               kernel_preErosion,
               kernel_firstDilation,
               kernel_firstErosion,
               hu=True,
               threshold=200,
               display=False,
               extract=[],
               size=60,
               areamin=None)
```

Introduction
------------
Obtain the mask, the labels of the mask and the image after applying the mask.

Parameters
----------
img : array
    Image which will be processed.
kernel_preErosion : list
    Size of the kernel to delete some artefacts.
kernel_firstDilation : list
    Size of the kernel in the first dilation.
kernel_firstErosion : list
    Size of the kernel in the first erosion.
hu : True, optional
    True if input image is RGB. The default is True.
threshold : int, optional
    Threshold parameter of the bounding box detection. The default is 200.
display : boolean, optional
    If it is True, plot the result of applying the make_bonesmask function in one image. The default is False.
extract : list, optional
    Select the label of the bone to be extracted.
    It only makes sense if you use the threshold > 200 because then the code extracts several bones.
    If we use threshold < 50 then the code ectracts large organs like the leg, and there are usualy not several labels of interest.
    The default is [].
size : int, optional
    Erosion and dilation kernel size.
    With the spacing [1,1,1] is usually size = 16.
    With spacing [0.5, 0.5, 0.5] is usually size = 30.
    With spacing [0.5, 0.25, 0.25] is usually size = 60.
    But beware, this depends on the original size of the DICOM image.
    The default is 16.

Returns
-------
mask : array
    Mask of the input image generated.
mask*img : array
    Result of applying the mask in the input img.
mask*img_org : array
    Result of applying the mask in a copy of the input img.
new_label_norm : array
    Labels in which the mask generated is divided.


## CreateTissueFromArray
```python
CreateTissueFromArray(imageData,
                      ThrIn,
                      ThrOut,
                      color='skeleton',
                      isoValue=127.5)
```

Introduction
------------
This function set all values between ThrIn, ThrOut to 255 for the correct 3D visualization.
Applies to imageData a gaussian filter and a smoother to generate the figure as perfect as possible.

Parameters
----------
imageData : vtk
    .
ThrIn : int
    Lower limit of the range of values.
ThrOut : int
    Upper limit of the range of values.
color : string, optional
    Color of the tissue map. The default is "skeleton".
isoValue : float, optional
    Value of the surface at a HU level compressed between 0 and 255. The default is 127.5.

Returns
-------
actor : vtk
    Figure of the 3D visualization.
stripper : vtk
    .


## CreateTissueMap
```python
CreateTissueMap()
```

Introduction
------------
Create a Tissue map

Returns
-------
tissueMap : dict
    Tissue map created.


## CreateLut
```python
CreateLut()
```

Introduction
------------
Assign a color to a tissue

Returns
-------
colorLut : vtk
    Color asigned to a tissue.


## compute_thickness_image
```python
compute_thickness_image(image, contourid=0, grow=False)
```

Introduction
------------
Calculate the thickness of the contour selected with the contourid

Parameters
----------
image : array
    Image of which we want to compute the thickness.
contourid : int, optional
    Is the index of the contour. If it is '-1', corresponds to the contour with largest area . The default is 0.
grow : boolean, optional
    If it is True, increase the size of contour. Only for visual purpose.
    The default is False.

Returns
-------
contourid : int
    Return the index of the contour with largest area.
contours : tuple
    Contour coordinates.
external.T
    Binary images with contour.
width_matrix.T
    Thickness values.


## convertTo1D
```python
convertTo1D(array_coordinates,
            array_thickness,
            countour_index=0,
            reference_x=100,
            verbose=True)
```

Introduction
------------
Convert the value of thickness of the slice into a 1D measure.

Parameters
----------
array_coordinates : list
    List with contour coordinates returned by compute_thickness_image.
array_thickness : list
    List with thickness values on the form of an image returned by compute_thickness_image.
countour_index : list, optional
    Select the contour id:
        For only one bone in the dicom, contourID will be surely:0
        For two bones in the dicom, contourID could be :0 for the smallest and 2 for the largest
        If countour_index is a list, then is a list with the contourID  used for each slide (obtained runing compute_thickness_image when contourid=-1). This list have the contour ID or -1 if no contours where detected
reference_x : int, optional
    Point in the slides to take the reference for the 2D->1D conversion.
    The default is 100.

Returns
-------
array_thickness_1d : dict
    Measures of thickness of the contour selected of each slice.


## show_cuts_position
```python
show_cuts_position(cortesG, num_views, G, poly_data, bounds, spacing)
```

Introduction
------------
Represent in the 3D model the height of the slices of which the thickness has been shown.

Parameters
----------
cortesG : list
    Array with slices whose thickness has been represented.
num_views : int
    Number of cuts.
G : tuple
    Center of mass of the bone to set the center of mass cut.
poly_data : vtkPolyData
    Poly data of the stl generated.
bounds : tuple
    Bounds of the poly data to set origin and points of the cuts.

Returns
-------
None.



## show_cuts_position_restored
```python
show_cuts_position_restored(cortesG, num_views, G, np_scalars, bounds,
                            spacing)
```

Introduction
------------
Represent in the 3D model the height of the slices of which the thickness has been shown.

Parameters
----------
cortesG : list
    Array with slices whose thickness has been represented.
num_views : int
    Number of cuts.
G : tuple
    Center of mass of the bone to set the center of mass cut.
poly_data : vtkPolyData
    Poly data of the stl generated.
bounds : tuple
    Bounds of the poly data to set origin and points of the cuts.

Returns
-------
None.



## show_cuts
```python
show_cuts(array_thickness, cortesG, num_views, spacing, origin)
```

Introdcution
------------
Represent the sections of the bone that generate the cuts

Parameters
----------
array_thickness : list
    List with thickness values on the form of an image returned by compute_thickness_image.
cortesG : list
    Array with slices whose thickness has been represented.
num_views : int
    Number of cuts.
spacing : array
    Spacing to be set in the vtkImageData.
origin : array
    Origin to be set in the vtkImageData.

Returns
-------
None.



## rotation_matrix_from_vectors
```python
rotation_matrix_from_vectors(vec1, vec2)
```

Introduction
------------
Find the rotation matrix that aligns vec1 to vec2

Parameters
----------
vec1 : array
    A 3d "source" vector.
vec2 : array
    A 3d "destination" vector.

Returns
-------
mat
    A transform matrix (3x3) which when applied to vec1, aligns it with vec2.



## orientation_slice
```python
orientation_slice(image, represent=False)
```

Introduction
------------
Detect orientation by using PCA and maximum variance directions.

Parameters
----------
image : array
    Slice that corresponds to the bone center of mass.
represent : boolean, optional
    If it is True, represent the slice with the main vector of orientation.
    The default is False.

Returns
-------
vector : array
    Main vector that corresponds to the orientation of the bone.



## getOrientationMainVector
```python
getOrientationMainVector(pts,
                         img,
                         arrowsize=3,
                         textsize=0.3,
                         textcoord=None)
```

Introduction
------------
Obtain the main orientation vector with the principal component.

Parameters
----------
pts : int
    Index of the contour with max area.
img : array
    Slice in question.
arrowsize : float
    Size of the array.
    The default is 3.
textsize : float
    Size of the text.
    The default is 0.3.
textcoord : array
    Coordinates of the text.
    The default is None.

Returns
-------
array
    Main vector with the principal component.



## drawAxis
```python
drawAxis(img, p_, q_, color, scale)
```

Introduction
------------
Draw the arrows of the main vector in the image.

Parameters
----------
img : array
    Slice on which to draw the arrow.
p_ : tuple
    Center of the image.
q_ : Tuple
    Point of the vector.
color : tuple
    Color to show the vector.
scale : int
    Scale of the draw.

Returns
-------
None.



## getClosestPointInRadius
```python
getClosestPointInRadius(kDTree,
                        thickness,
                        thickness_values,
                        ccenter,
                        radius=0.5)
```

Introduction
------------
Function to retreive all the closest points in radius. And between them, the closest with the thickness different from zero.
Note that:
    If there are no points because the radius is to small, it returns None.
    If there are several points but all with zero thickness, it returns the closest.
    If there are several points with thicknesses values, it returns the closest.
    The distance used is the euclidean between coordinates, but the distance returned by FindClosestPoint is the squared euclidean distance

Parameters
----------
kDTree : vtk object
    vtkKdTree object of vtkmodules.vtkCommonDataModel module.
thickness : vtkImageData
    Coordinates of closest point in thickness array.
thickness_values : vtkFloatArray
    Thinkness value of closest point in thickness array.
ccenter : list
    Center of the cell (triangle).
radius : float, optional
    Radius to locate the nearby voxels of the points. The default is 0.5.

Returns
-------
ids : int
    .
distances[index] : float
    distance value.
thicknesses[index] : float
    thick value to color the 3D model.



## color3DModelWithThickness
```python
color3DModelWithThickness(array_thickness, spacing_n, origin, imgstenc)
```

Introduction
------------
Represent a 3D model colored based on the thickness value.

Parameters
----------
array_thickness : list
    Color images with thickness.
spacing_n : list
    Spacing of the whiteImage.
origin : list
    Origin of the whiteImage.
imgstenc : vtkImageStencil
    .

Returns
-------
None.



## getArea
```python
getArea(side)
```

Introduction
------------
Return the area of a square

Parameters
----------
side : array
    Numpy array 4x3 with the coordinates of vertex.

Returns
-------
scalar : float
    Area of a square.



## WriteImage
```python
WriteImage(fileName, renWin, rgba=True)
```

Introduction
------------
Write the render window view to an image file.

Parameters
----------
fileName : string
    The file name, if no extension then PNG is assumed.
renWin : vtk Render window
    The render window.
rgba : TYPE, optional
     Used to set the buffer type.
     The default is True.

Raises
------
RuntimeError
    What is needed.

Returns
-------
None.



## sort_contours2
```python
sort_contours2(cnts, method='left-to-right')
```

Introduction
------------
Sort contours left to right

Parameters
----------
cnts : tuple
    contours of the slice.
method : string, optional
    Method to sort the contours.
    The default is "left-to-right".

Returns
-------
cnts : tuple
    Contours sorted left to right.
boundingBoxes : TYPE
    DESCRIPTION.


