import json
import vtk
import numpy as np
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from logging import FileHandler
from vlogging import VisualRecord
import math
import os
import pickle

import cv2 as cv

import sys
sys.path.append("../libs")
from libraries import load_scan, get_pixels_hu, sample_stack, resample, find_bounding_box, find_bounding_box_sample_stack, make_bonesmask, CreateTissueFromArray, CreateTissueMap, CreateLut, compute_thickness_image, convertTo1D, show_cuts, show_cuts_position, rotation_matrix_from_vectors, orientation_slice, getOrientationMainVector, drawAxis, getClosestPointInRadius, color3DModelWithThickness, getArea, show_cuts_position_restored, WriteImage

import configparser
    
####################
# READ CONFIG FILE #
####################

# instantiate
config = configparser.RawConfigParser()

# parse existing file
config.read('../config/file.ini')

# read values from a section
resources_path = config.get('dicom', 'resources_path')

# read values from a section
spacing_n = config.get('pre-process', 'spacing')
spacing_n = json.loads(spacing_n)

# read values from a section
num_views_thickness = config.getint('thickness', 'num_views_thickness')

# read values from a section
reference_bone = config.get('all dicom', 'reference_bone')

# create a section to write reference vectors if does not exists
if 'reference vectors' not in config:
    config.add_section('reference vectors')
    
# read values from a section
output_path = config.get('dicom', 'output_path')
if not os.path.exists(output_path+"thickness"):
    os.makedirs(output_path+"thickness")
if not os.path.exists(output_path+"profiles"):
    os.makedirs(output_path+"profiles")
resources_path = config.get('dicom', 'resources_path')

##########################
# INITIALIZE LOGGER FILE #
##########################

logger = logging.getLogger("demo")
fh = FileHandler('../logs/referenceBone.html', mode="w")

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
fh.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(fh)


############################
# OBTAIN REFERENCE VECTORS #
############################
logger.debug(VisualRecord(">>> REFERENCE BONE IS:  %s" %(reference_bone)))

reader = vtk.vtkSTLReader()
reader.SetFileName(reference_bone)

spacing = [spacing_n[2], spacing_n[1], spacing_n[0]] #[0.25, 0.25, 0.5]
thickness_spacing = spacing_n[2]                     # 0.25
reader.Update()

#Extract Poly data
poly_data = reader.GetOutput()
bounds = poly_data.GetBounds()

print("Max dimension in X: ", bounds[1] - bounds[0])
print("Max dimension in Y: ", bounds[3] - bounds[2])
print("Max dimension in Z: ", bounds[5] - bounds[4])

# Calculate Center Of Mass
centerOfMass = vtk.vtkCenterOfMass()
centerOfMass.SetInputData(poly_data)
centerOfMass.SetUseScalarsAsWeights(False)
centerOfMass.Update()
G = centerOfMass.GetCenter()

whiteImage=vtk.vtkImageData()
whiteImage.SetSpacing(spacing[0],spacing[1],spacing[2])

dim=[]
dim.append(int(math.ceil((bounds[1] - bounds[0]) /spacing[0])))
dim.append(int(math.ceil((bounds[3] - bounds[2]) /spacing[1])))
dim.append(int(math.ceil((bounds[5] - bounds[4]) /spacing[2])))

whiteImage.SetDimensions(dim);
whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
print("Voxels: ",dim)

origin=[]
origin.append(bounds[0] + spacing[0] / 2)
origin.append(bounds[2] + spacing[1] / 2)
origin.append(bounds[4] + spacing[2] / 2)
whiteImage.SetOrigin(origin)
print("Origin: ",origin)
whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

# fill the image with foreground voxels
inval = 255
outval = 0

ones=inval*np.ones(whiteImage.GetDimensions())
ones=np.array(ones,'uint8')
print(type(ones[0,0,0]))
print(ones.shape)
print(ones.flatten().shape)

vtk_data_array = numpy_to_vtk(ones.flatten())
whiteImage.GetPointData().SetScalars(vtk_data_array)
print(vtk_to_numpy(whiteImage.GetPointData().GetArray(0)).shape)
print(whiteImage.GetPointData().GetArray(0).GetValueRange())

# polygonal data --> image stencil:
pol2stenc = vtk.vtkPolyDataToImageStencil()
pol2stenc.SetInputData(reader.GetOutput())
pol2stenc.SetOutputOrigin(origin)
pol2stenc.SetOutputSpacing(spacing)
pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
pol2stenc.Update()

# cut the corresponding white image and set the background:
imgstenc = vtk.vtkImageStencil()
imgstenc.SetInputData(whiteImage)
imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
imgstenc.ReverseStencilOff()
imgstenc.SetBackgroundValue(outval)
imgstenc.Update()

scalars = imgstenc.GetOutput().GetPointData().GetScalars()
np_scalars = vtk_to_numpy(scalars)     
np_scalars = np_scalars.reshape(dim[2], dim[1], dim[0]) 
np_scalars = np_scalars.transpose(0,2,1)
print("Shape: ",np_scalars.shape)

sample_stack(np_scalars, rows=10, cols=10, start_with=1, show_every=7, color=False)

array_thickness=[]
array_contours=[]
array_coordinates=[]
array_contourid=[]
for i in tqdm(range(len(np_scalars))):
    contourid,coordinates, contour,thick=compute_thickness_image(np_scalars[i,:,:],contourid=-1,grow=False)
    array_thickness.append(thick*thickness_spacing)
    array_contours.append(contour)
    array_coordinates.append(coordinates)
    array_contourid.append(contourid)
    
sample_stack(array_thickness, rows=10, cols=10, start_with=1, show_every=7, color=True, cmap="magma")

print("Max dimension in Z: ", bounds[5] - bounds[4])
print("minz", bounds[4])
print("max", bounds[5])
print("Center of Mass", G)
print("Spacing", spacing)

#Obtain the slice of Center Of Mass
absolute_distance_to_COM=G[2]+bounds[4]
print("Absolute distance COM: ",absolute_distance_to_COM)
sliceCOM=round(absolute_distance_to_COM/spacing[2])
print("slice COM: ",sliceCOM)
plt.imshow(array_thickness[sliceCOM],cmap="magma")

#Save thickness to disk
fname=reference_bone.split("\\")[-1].split(".")[0]
array=np.array(array_thickness)
np.savez_compressed(output_path+"thickness\\"+fname, array)
logger.debug(VisualRecord(">>> THICKNESS saved in:  %s" %(output_path+"thickness\\"+fname)))

####################################
# REFERENCE VECTOR FOR ORIENTATION #
####################################
image=np_scalars[sliceCOM,:,:]

image_rgb = cv.cvtColor(image, cv.COLOR_GRAY2BGR )
gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
_, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

areas=[]
for i, c in enumerate(contours):
    areas.append(cv.contourArea(c))
index=areas.index(max(areas))
    
# Find the orientation of each shape
vector1pca=getOrientationMainVector(contours[index], image_rgb, arrowsize=5)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(image_rgb)
#plt.show()
plt.savefig(resources_path+"RefOrientationV.png")
cv_refOr = cv.imread(resources_path+"RefOrientationV.png")
resized = cv.resize(cv_refOr, (350,350), interpolation = cv.INTER_AREA)
logger.debug(VisualRecord("Ref Orientation Vector", resized, fmt="png"))

vector1pca_tuple = tuple([float(i) for i in vector1pca])
print("main ORIENTING REFERENCE: ", vector1pca_tuple)

# Add reference vector in our config file
config.set('reference vectors', 'orientation_vector', [vector1pca_tuple[0], vector1pca_tuple[1]])

# Writing our configuration file to 'config/file.ini'
with open('../config/file.ini', 'w') as configfile:
    config.write(configfile)
    

##################################
# REFERENCE VECTOR FOR ALIGNMENT #
##################################
BoundingBox = vtk.vtkOBBTree()
BoundingBox.SetDataSet(poly_data)
BoundingBox.SetMaxLevel(1)
BoundingBox.BuildLocator()
BoundingBoxPolyData = vtk.vtkPolyData()
BoundingBox.GenerateRepresentation(0, BoundingBoxPolyData)

BoundingBoxNormals = vtk.vtkPolyDataNormals()
BoundingBoxNormals.ComputeCellNormalsOn()
BoundingBoxNormals.SetInputData(BoundingBoxPolyData)
BoundingBoxNormals.Update()
BoundingBoxNormalsData = BoundingBoxNormals.GetOutput().GetCellData().GetNormals()
array=vtk_to_numpy(BoundingBoxNormalsData)

#print(BoundingBoxNormalsData)

BoundingBoxPolyData.GetCellData()
#print("Numero de celdas: ",BoundingBoxPolyData.GetNumberOfCells())
#print("Numero de puntos de la celda 0: ",BoundingBoxPolyData.GetCell(0).GetNumberOfPoints())
areas=[]
normals=[]
for i in range(6):
    side=vtk_to_numpy(BoundingBoxPolyData.GetCell(i).GetPoints().GetData())
    area=getArea(side)
    areas.append(area)
    normal=BoundingBoxNormals.GetOutput().GetCellData().GetNormals().GetTuple(i)
    normals.append(normal)
    print("cell ",i," area ",area," normal: ",normal)
    
vector1normal=normals[areas.index(min(areas))]

print("main ALIGNING REFERENCE: ", vector1normal)

# Add reference vector in our config file
config.set('reference vectors', 'alignment_vector', [vector1normal[0], vector1normal[1], vector1normal[2]])

# Writing our configuration file to 'config/file.ini'
with open('../config/file.ini', 'w') as configfile:
    config.write(configfile)
    
    
####################################
# EXTRACT 1D PROFILES OF THICKNESS #
####################################
# Show size of slices and the length of the array of contours
dimz,dimx,dimy = np_scalars.shape
print("Size of slices: ",dimz, dimx, dimy)
print("Lenght of Array of contours: ",len(array_coordinates))
     
# With this we look for the point of the X axis that more contours of the slices intersect
lengths=[]
for referencex in tqdm(range(dimx)):
    array_thickness_1d,noreferences = convertTo1D(array_coordinates,array_thickness,countour_index=array_contourid,reference_x = referencex, verbose=False)
    lengths.append(noreferences)
referencex=list(range(dimx))[lengths.index(min(lengths))]
print("The reference X that maximices the number of valid contours is: ",referencex)
logger.debug(VisualRecord(">>> REFERENCE for the 1D profiles conversion was:  %s" %(referencex)))

# Extraxt 1D profiles of thickness
array_thickness_1d,_ = convertTo1D(array_coordinates,array_thickness,countour_index=array_contourid,reference_x = referencex)

# Show number of slices 
print("Number of slices: ",len(array_thickness_1d.keys()))

# Plot the graphs with the thickness. 
cortes = []                                        # Array with slices whose thickness is going to be represented
profiles = {}                                       # Dict to be saved with slice number and 1D thickness
num_views=num_views_thickness                       # Number of views in the plot
keys=[k for k,v in array_thickness_1d.items() if v!=None]   # Items of array_thickness
total=len(keys)
delta=total//num_views                              # Distance between each cut
rows=num_views//4+1
fig=plt.figure(figsize=(18,rows*4))
for i in range (1,num_views+1):
    plt.subplot(rows,4,i)
    if array_thickness_1d[keys[delta*i-1]]: 
        plt.plot(array_thickness_1d[keys[delta*i-1]])
        x1,x2,y1,y2 = plt.axis()  
        plt.axis((x1,x2,0,10))
        plt.ylabel("Thickness [mm]")
        cortes.append(keys[delta*i-1])
        profiles[keys[delta*i-1]]=array_thickness_1d[keys[delta*i-1]]
    plt.title("Slice: "+str(keys[delta*i-1]))
fig.suptitle('1D Thickness Contours')
#plt.show()
plt.savefig(resources_path+"Thickness.png")
cv_thickness = cv.imread(resources_path+"Thickness.png")
resized = cv.resize(cv_thickness, (500,500), interpolation = cv.INTER_AREA)
logger.debug(VisualRecord("1D Thickness Contours", resized, fmt="png"))

#Save thickness dictionary profiles to disk
fname=reference_bone.split("\\")[-1].split(".")[0]
with open(output_path+"profiles\\"+fname+".pkl", 'wb') as handle:
     pickle.dump(profiles, handle, protocol=pickle.HIGHEST_PROTOCOL)
logger.debug(VisualRecord(">>> PROFILES DICTIONARY saved in:  %s" %(output_path+"profiles\\"+fname+".pkl")))


#############################
# SHOW CUTS GENERATED IN 2D #
#############################
show_cuts(array_thickness, cortes, num_views, spacing, origin)
plt.savefig(resources_path+"cuts.png")
cv_cuts = cv.imread(resources_path+"cuts.png")
resized = cv.resize(cv_cuts, (500,500), interpolation = cv.INTER_AREA)
logger.debug(VisualRecord("2D Thickness Contours", resized, fmt="png"))


#############################
# SHOW CUTS GENERATED IN 3D #
#############################
show_cuts_position(cortes, num_views, G, poly_data, bounds, spacing)
cv_cuts_p = cv.imread("cuts_p.png")
resized = cv.resize(cv_cuts_p, (350,350), interpolation = cv.INTER_AREA)
logger.debug(VisualRecord("3D chosen profiles", resized, fmt="png"))


#########################
# REPRESENT COLOR IN 3D #
#########################
#Convert the array_thickness to a vtkImageData
array=np.array(array_thickness)
array_tmp=array.transpose(1, 2, 0)
print("ORG: ",array.shape)
print("ORG_TRSP: ",array_tmp.shape)

# Convert numpy array to VTK array (vtkFloatArray)
vtk_data_array = numpy_support.numpy_to_vtk(
    num_array=array_tmp.transpose(2, 1, 0).ravel(),  # ndarray contains the fitting result from the points. It is a 3D array
    deep=True,
    array_type=vtk.VTK_FLOAT)

# Convert the VTK array to vtkImageData
img_vtk = vtk.vtkImageData()
img_vtk.SetDimensions(array_tmp.shape)
img_vtk.SetSpacing(spacing)
img_vtk.SetOrigin(origin)
img_vtk.GetPointData().SetScalars(vtk_data_array)

print("VTK: ",img_vtk.GetDimensions())

x,y,num=img_vtk.GetDimensions()
print(img_vtk.GetDimensions())
ox,oy,oz=img_vtk.GetOrigin()
print(img_vtk.GetOrigin())

thickness=img_vtk

surface = vtk.vtkMarchingCubes()
surface.SetInputData(imgstenc.GetOutput())
surface.ComputeNormalsOn()
surface.SetValue(0, 127.5)

surface.Update()

probe = vtk.vtkProbeFilter()
probe.SetInputData(surface.GetOutput())
probe.SetSourceData(thickness)
probe.Update()

probe.GetOutput()
rng = thickness.GetScalarRange()
fMin = rng[0]
fMax = rng[1]
print("RANGE:", rng[0],rng[1])

# Make the lookup table.
lut = vtk.vtkLookupTable()
lut.SetTableRange(fMin, fMax)
lut.SetSaturationRange(1, 1)
lut.SetHueRange(0, 0.6)
lut.SetValueRange(0, 5)
lut.Build()

normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(probe.GetOutputPort())

mapper = vtk.vtkPolyDataMapper()
mapper.ScalarVisibilityOn()
mapper.SetLookupTable(lut)
mapper.SetInputConnection(normals.GetOutputPort())
mapper.SetScalarRange(fMin, fMax)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a renderer, render window, and interactor
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Add the actors to the scene
renderer.AddActor(actor)
renderer.SetBackground(1, 1, 1) 
renderer.ResetCamera()
renderer.ResetCameraClippingRange()
camera = renderer.GetActiveCamera()
camera.Elevation(230)
camera.Azimuth(135)
camera.Roll(55)
renderer.SetActiveCamera(camera)
# Add the actors to the scene

# Render and interact
renderWindow.SetSize(480, 480)
renderWindow.Render()
filename = resources_path+'colors3D.png'
WriteImage(filename, renderWindow, rgba=False)
cv_colors3D = cv.imread(resources_path+"colors3D.png")
resized = cv.resize(cv_colors3D, (350,350), interpolation = cv.INTER_AREA)
logger.debug(VisualRecord("3D Bone thickness in colors", resized, fmt="png"))
#renderWindowInteractor.Start()
