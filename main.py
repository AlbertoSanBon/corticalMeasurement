# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:03:34 2022

@author: albsb
"""
import json
import vtk
import numpy as np
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import logging
from logging import FileHandler
from vlogging import VisualRecord
import cv2
import math

from config.libraries import load_scan, get_pixels_hu, sample_stack, resample, find_bounding_box, find_bounding_box_sample_stack, make_bonesmask, CreateTissueFromArray, CreateTissueMap, CreateLut, compute_thickness_image, convertTo1D, show_cuts_position, show_cuts, getClosestPointInRadius, color3DModelWithThickness

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0


####################
# READ CONFIG FILE #
####################

# instantiate
config = ConfigParser()

# parse existing file
config.read('config/file.ini')

# read values from a section
data_path = config.get('dicom', 'data_path')
output_path = config.get('dicom', 'output_path')

# read values from a section
spacing = config.get('pre-process', 'spacing')
spacing = json.loads(spacing)
threshold = config.getint('pre-process', 'threshold')
extract = config.get('pre-process', 'extract')
extract = json.loads(extract)
size = config.getint('pre-process', 'size')
kernel_preErosion = config.get('pre-process', 'kernel_preErosion')
kernel_preErosion = json.loads(kernel_preErosion)
kernel_firstDilation = config.get('pre-process', 'kernel_firstDilation')
kernel_firstDilation = json.loads(kernel_firstDilation)
kernel_firstErosion = config.get('pre-process', 'kernel_firstErosion')
kernel_firstErosion = json.loads(kernel_firstErosion)

# read values from a section
threshold_between_min = config.getint('post-process', 'threshold_between_min')
threshold_between_max = config.getint('post-process', 'threshold_between_max')
convert_stl = config.getboolean('post-process', 'convert_stl')

# read values from a section
num_views_thickness = config.getint('thickness', 'num_views_thickness')


##########################
# INITIALIZE LOGGER FILE #
##########################

logger = logging.getLogger("demo")
fh = FileHandler('logs.html', mode="w")

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
fh.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(fh)


###############
# PRE-PROCESS #
###############

# List DICOM files
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Print out the first 5 file names to verify we're in the right folder.
print("Total of %d DICOM images.\nFirst 5 filenames:" % len(onlyfiles))
logger.debug(VisualRecord("Total of %d DICOM images." % len(onlyfiles)))

print('\n'.join(onlyfiles[:5]))

# Load all DICOM from a folder into a list for manipulation
slices=load_scan(data_path,onlyfiles)

# Show metadata
print(slices[0])

# Create a numpy matrix of Hounsfield Units (HU)
imgs = get_pixels_hu(slices)
imgs.shape

# Save images
id=0
np.save(output_path + "fullimages_new%d.npy" % (id), imgs)

# Load images
file_used=output_path+"fullimages_new%d.npy" % (id)
imgs_to_process = np.load(file_used)

# Show & log imgs_to_process
sample_stack(imgs_to_process, rows=5, cols=7, start_with=1, show_every=7)
cv_sample_stack = cv2.imread("Sample_stack.png")
resized = cv2.resize(cv_sample_stack, (1020,1020), interpolation = cv2.INTER_AREA)
logger.debug(VisualRecord("Images to process", resized, fmt="png"))

# Plot & log histogram to see Hounsfield Units (HU)
plt.figure()
plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()
plt.savefig('Histogram.png')
cv_histogram = cv2.imread('Histogram.png')
logger.debug(VisualRecord("Histogram", cv_histogram, fmt="png"))

# Print Slice Thickness & Pixel Spacing
print ("Slice Thickness: %f" % slices[0].SliceThickness)
logger.debug(VisualRecord("Slice Thickness: %f" % slices[0].SliceThickness))
print ("Pixel Spacing (row, col): (%f, %f) " % (slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]))
logger.debug(VisualRecord("Pixel Spacing (row, col): (%f, %f) " % (slices[0].PixelSpacing[0], slices[0].PixelSpacing[1])))

# Print dimensions of one image & bone length
x,y=imgs_to_process[0].shape
print("Each image is ({}mm x {}mm)".format(x*slices[0].PixelSpacing[0],y*slices[0].PixelSpacing[1]))
logger.debug(VisualRecord("Each image is ({}mm x {}mm)".format(x*slices[0].PixelSpacing[0],y*slices[0].PixelSpacing[1])))
print("Bone length {}mm".format(imgs_to_process.shape[0]*slices[0].SliceThickness))
logger.debug(VisualRecord("Bone length {}mm".format(imgs_to_process.shape[0]*slices[0].SliceThickness)))

# Print shape before resampling
print("Shape before resampling\t", imgs_to_process.shape)
logger.debug(VisualRecord("Shape before resampling: (%d, %d, %d)" %(imgs_to_process.shape[0], imgs_to_process.shape[1], imgs_to_process.shape[2])))

# Resample
imgs_after_resamp, spacing = resample(imgs_to_process, slices, spacing)

# Print shape after resamplings
print("Shape after resampling\t", imgs_after_resamp.shape)
logger.debug(VisualRecord("Shape after resampling: (%d, %d, %d)" %(imgs_after_resamp.shape[0], imgs_after_resamp.shape[1], imgs_after_resamp.shape[2])))

# Print dimensions of one image after resampling
x,y=imgs_after_resamp[0].shape
print("Each image is ({}mm x {}mm)".format(x*spacing[1],y*spacing[2]))
logger.debug(VisualRecord("Each image is ({}mm x {}mm)".format(x*spacing[1],y*spacing[2])))

# Show bounding box detection of images
find_bounding_box_sample_stack(imgs_after_resamp, hu=True, show_box=True, threshold=110, rows=7, cols=7, start_with=1, show_every=7)

# Arrays in which the results of make_bonesmask function will be stored
masked_bones= []
masked_bones_hu= []
masks =[]
labels =[]

# Obtain masked bones, masks and labels of the masks
counter=0
for img in tqdm(imgs_after_resamp):
    mascara,imagen_norm,imagen_hu,etiquetas=make_bonesmask(img, kernel_preErosion, kernel_firstDilation, kernel_firstErosion, hu=True, threshold=threshold, display=False, extract=extract, size=size)
    masked_bones.append(imagen_norm)
    masked_bones_hu.append(imagen_hu)
    masks.append(mascara)
    labels.append(etiquetas)

#mascara,imagen_norm,imagen_hu,etiquetas=make_bonesmask(imgs_after_resamp[25], [1,1], [27,27], [2,2], hu=True, threshold=125, display=True, extract=extract, size=size)

# Show & log masked bones
sample_stack(masked_bones, rows=7, cols=7, start_with=1, show_every=7)
cv_sample_stack = cv2.imread("Sample_stack.png")
resized = cv2.resize(cv_sample_stack, (1020,1020), interpolation = cv2.INTER_AREA)
logger.debug(VisualRecord("Masked bones", resized, fmt="png"))

# Show & log masks
sample_stack(masks, rows=7, cols=7, start_with=1, show_every=7)
cv_sample_stack = cv2.imread("Sample_stack.png")
resized = cv2.resize(cv_sample_stack, (1020,1020), interpolation = cv2.INTER_AREA)
logger.debug(VisualRecord("Masks", resized, fmt="png"))

# Show & log labels of the mask
sample_stack(labels, rows=7, cols=7, start_with=1, show_every=7, color=True)
cv_sample_stack = cv2.imread("Sample_stack.png")
resized = cv2.resize(cv_sample_stack, (1020,1020), interpolation = cv2.INTER_AREA)
logger.debug(VisualRecord("Labels", resized, fmt="png"))

# Convert images into a numpy array
imgs_after_mask=np.array(masked_bones_hu)

# Save pre-processed images
np.save(output_path + "leg_new.npy", imgs_after_mask)


################
# POST-PROCESS #
################

# Select the file to use
file_used=output_path+"leg_new.npy"

# Load file and obtain the shape
imgs_to_process = np.load(file_used).astype(np.float64) 
imgs = imgs_to_process
data_shape = imgs.shape
print(data_shape)

# Create vtk model and set dimensions,spacing and origin for the 3D visualization
imdata = vtk.vtkImageData()
depthArray = numpy_support.numpy_to_vtk(num_array=imgs.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
imdata.SetDimensions((data_shape[2], data_shape[1], data_shape[0])) 
imdata.SetSpacing([spacing[2], spacing[1], spacing[0]])
imdata.SetOrigin([0,0,0])
_=imdata.GetPointData().SetScalars(depthArray)

selectTissue = vtk.vtkImageThreshold()
selectTissue.ThresholdBetween(threshold_between_min,threshold_between_max) # Select which HU represent
selectTissue.ReplaceInOn()    # Determines whether to replace the pixel in range with InValue
selectTissue.SetInValue(255)  # set all values between ThrIn,ThrOut to 255
selectTissue.ReplaceOutOn()
selectTissue.SetOutValue(0)   # set all values otside ThrIn,ThrOut to 0
selectTissue.SetInputData(imdata)
selectTissue.Update()

#Apply a gaussian filter to smooth the 3D
gaussianRadius = 5
gaussianStandardDeviation = 1.0
gaussian = vtk.vtkImageGaussianSmooth()
gaussian.SetStandardDeviations(gaussianStandardDeviation, gaussianStandardDeviation, gaussianStandardDeviation)
gaussian.SetRadiusFactors(gaussianRadius, gaussianRadius, gaussianRadius)
gaussian.SetInputData(selectTissue.GetOutput())
gaussian.Update()
    
# Create a surface
surface = vtk.vtkMarchingCubes()
surface.SetInputData(gaussian.GetOutput())
surface.ComputeNormalsOn()
surface.SetValue(0, 127.5)  # define surface 0 as a isosurface at HU level 127.5

# Create a renderer and apply backgtound color
renderer = vtk.vtkRenderer()
colors = vtk.vtkNamedColors()
renderer.SetBackground(colors.GetColor3d('DarkSlateGray'))

# Create the render window
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetWindowName('MarchingCubes')

# Create an interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Create a mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(surface.GetOutputPort())
mapper.ScalarVisibilityOff()

# Create an actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(colors.GetColor3d('MistyRose'))

# Apply the actor to the renderer
renderer.AddActor(actor)

# Show the 3D model
render_window.Render()
#interactor.Start()

# Convert to STL file
if convert_stl:
    renderer = vtk.vtkRenderer()
    actor, stripper = CreateTissueFromArray(imdata,threshold_between_min,threshold_between_max,"skeleton")
        
    #Save to STL in the output path
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_path+"leg_new.stl")
    stripper.Update()
    writer.SetInputData(stripper.GetOutput())
    writer.Write()


#################
# GET THICKNESS #
#################

#Select the STL generated
reader = vtk.vtkSTLReader()
reader.SetFileName(output_path+"leg_new.stl")
reader.Update()

# Extract Poly data
poly_data = reader.GetOutput()
bounds = poly_data.GetBounds()

# Show and Log dimensions of the poly data
print("Max dimension in X: ", bounds[1] - bounds[0])
logger.debug(VisualRecord("Max dimension in X: %d" %(bounds[1] - bounds[0])))
print("Max dimension in Y: ", bounds[3] - bounds[2])
logger.debug(VisualRecord("Max dimension in Y: %d" %(bounds[3] - bounds[2])))
print("Max dimension in Z: ", bounds[5] - bounds[4])
logger.debug(VisualRecord("Max dimension in Z: %d" %(bounds[5] - bounds[4])))

# Calculate the center of mass
centerOfMass = vtk.vtkCenterOfMass()
centerOfMass.SetInputData(poly_data)
centerOfMass.SetUseScalarsAsWeights(False)
centerOfMass.Update()
G = centerOfMass.GetCenter()
print("Center of Mass:", G )

# Obtain whiteImage
whiteImage=vtk.vtkImageData()
spacing_n=[spacing[2], spacing[1], spacing[0]]
whiteImage.SetSpacing(spacing_n[0],spacing_n[1],spacing_n[2])

# Set dimensions to whiteImage
dim=[]
dim.append(int(math.ceil((bounds[1] - bounds[0]) /spacing_n[0])))
dim.append(int(math.ceil((bounds[3] - bounds[2]) /spacing_n[1])))
dim.append(int(math.ceil((bounds[5] - bounds[4]) /spacing_n[2])))
whiteImage.SetDimensions(dim);
whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
print("Voxels: ",dim)

# Set origin to whiteImage
origin=[]
origin.append(bounds[0] + spacing_n[0] / 2)
origin.append(bounds[2] + spacing_n[1] / 2)
origin.append(bounds[4] + spacing_n[2] / 2)
whiteImage.SetOrigin(origin)
print("Origin: ",origin)
whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

# fill the image with foreground voxels:
inval = 255
outval = 0
numberOfPoints = whiteImage.GetNumberOfPoints();
for i in tqdm(range(0, numberOfPoints)):
    whiteImage.GetPointData().GetScalars().SetTuple1(i, inval)
    
# polygonal data --> image stencil:
pol2stenc = vtk.vtkPolyDataToImageStencil()
pol2stenc.SetInputData(reader.GetOutput())
pol2stenc.SetOutputOrigin(origin)
pol2stenc.SetOutputSpacing(spacing_n)
pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
pol2stenc.Update()

# cut the corresponding white image and set the background:
imgstenc = vtk.vtkImageStencil()
imgstenc.SetInputData(whiteImage)
imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
imgstenc.ReverseStencilOff()
imgstenc.SetBackgroundValue(outval)
imgstenc.Update()

# Obtain the sacalars of data
scalars = imgstenc.GetOutput().GetPointData().GetScalars()
np_scalars = vtk_to_numpy(scalars)     
np_scalars = np_scalars.reshape(dim[2], dim[1], dim[0]) 
np_scalars = np_scalars.transpose(0,2,1)
print("Shape: ",np_scalars.shape)

# Process all the slices and extract and store:
thickness_spacing = spacing[2]
array_thickness=[]      # color images with thickness
array_contours=[]       # binary images with contour 
array_coordinates=[]    # the coordinates of the contours 
array_contourid=[]      # index of the contour with largest area
for i in tqdm(range(len(np_scalars))):
    contourid, coordinates, contour,thick=compute_thickness_image(np_scalars[i,:,:],contourid=-1,grow=True)
    
    array_thickness.append(thick*thickness_spacing)
    array_contours.append(contour)
    array_coordinates.append(coordinates)
    array_contourid.append(contourid)
    
# Show the content of array_thickness
sample_stack(array_thickness, rows=7, cols=7, start_with=1, show_every=7, color=True, cmap="magma")
cv_sample_stack = cv2.imread("Sample_stack.png")
resized = cv2.resize(cv_sample_stack, (1020,1020), interpolation = cv2.INTER_AREA)
logger.debug(VisualRecord("Array thickness", resized, fmt="png"))

# Show size of slices and the length of the array of contours
dimz,dimx,dimy = np_scalars.shape
print("Size of slices: ",dimz, dimx, dimy)
print("Lenght of Array of contours: ",len(array_coordinates))

# Extraxt 1D profiles of thickness

#First select the contourid which is repeated the most
counter = {}
for i in range(min(array_contourid), max(array_contourid)+1):
    counter[i] = array_contourid.count(i)
    
m = max(counter.values())
for i in range(min(array_contourid), max(array_contourid)+1):
    if counter[i] == m: 
        contourindex = i
        
# With this we look for the point of the X axis that more contours of the slices intersect
lengths=[]
for referencex in tqdm(range(dimx)):
    array_thickness_1d,noreferences = convertTo1D(array_coordinates,array_thickness,countour_index=contourindex,reference_x = referencex, verbose=False)
    lengths.append(noreferences)
referencex=list(range(dimx))[lengths.index(min(lengths))]
print("The reference X that maximices the number of valid contours is: ",referencex)

# Extraxt 1D profiles of thickness
array_thickness_1d,_ = convertTo1D(array_coordinates,array_thickness,countour_index=array_contourid,reference_x = referencex)

# Show number of slices 
print("Number of slices: ",len(array_thickness_1d.keys()))
logger.debug(VisualRecord("Number of slices: %d" %len(array_thickness_1d.keys())))

# Plot the graphs with the thickness. (1/4 mm each axis)
cortesG = []                                        # Array with slices whose thickness is going to be represented
num_viewsG=num_views_thickness                      # Number of views in the plot
keysG=[k for k,v in array_thickness_1d.items()]     # Items of array_thickness
totalG=len(keysG)                                   # Length of keysG 
deltaG=totalG//num_viewsG                           # Distancia between each cut
rowsG=num_viewsG//4+1                               # Size of the rows
fig=plt.figure(figsize=(18,rowsG*4))

referenciaG = int(G[2]/0.5)                         # Reference to start doing cuts
inicio = referenciaG - num_viewsG//2*deltaG         # State the first cut at the init of the bone
# Starting in the inicio, we increase the distanceG
for i in range (1,num_viewsG+1): # Is num_views +1 because the first iteration can not be 0
    plt.subplot(rowsG,num_viewsG//3,i)              # Number of columns in the plot
    
    try: 
        # If we found in delta position a slice with thickness, plot it
        if array_thickness_1d[keysG[inicio + deltaG*(i-1)]]: 
            plt.plot(array_thickness_1d[keysG[inicio + deltaG*(i-1)]])
            plt.title("Slice: "+str(keysG[inicio + deltaG*(i-1)]))
            cortesG.append(keysG[inicio + deltaG*(i-1)])
        # Else, we look for the closest slice with thickness    
        else:
            for j in range (1, len(keysG)):
                if array_thickness_1d[keysG[inicio + deltaG*(i-1) + j]]:
                            plt.plot(array_thickness_1d[keysG[inicio + deltaG*(i-1) + j]])
                            plt.title("Slice: "+str(keysG[inicio + deltaG*(i-1) + j]))
                            cortesG.append(keysG[inicio + deltaG*(i-1) + j])
                            break
    except: print("Slice not found")
    
    # Put the labels of the plot (1/4 mm because the spacing is 0.25 in x,y axis)
    plt.xlabel("1/4 mm")
    plt.xlim(0,500)
    plt.ylabel("1/4 mm")
    plt.ylim(0,50)
fig.suptitle('1D Thickness Contours')
plt.show()
plt.savefig("Thickness.png")
cv_thickness = cv2.imread("Thickness.png")
resized = cv2.resize(cv_thickness, (1020,1020), interpolation = cv2.INTER_AREA)
logger.debug(VisualRecord("Thickness", resized, fmt="png"))

# Show the cuts generated in 2D
show_cuts(array_thickness, cortesG, num_viewsG, spacing_n, origin)
cv_cuts = cv2.imread("Cuts.png")
resized = cv2.resize(cv_cuts, (1020,1020), interpolation = cv2.INTER_AREA)
logger.debug(VisualRecord("Thickness", resized, fmt="png"))

# Show the position of the cuts in a 3D model
show_cuts_position(cortesG, num_viewsG, G, poly_data, bounds, spacing_n)
# CLOSE THE WINDOW GENERATED TO CONTINUE

# Represent a 3D model colored based on the thickness value
color3DModelWithThickness(array_thickness, spacing_n, origin, imgstenc)
