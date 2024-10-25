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
from os import listdir
import vg
import cv2 as cv
from scipy.ndimage import rotate
import os
import pickle

import sys
sys.path.append("../libs")
from libraries import load_scan, get_pixels_hu, sample_stack, resample, find_bounding_box, find_bounding_box_sample_stack, make_bonesmask, CreateTissueFromArray, CreateTissueMap, CreateLut, compute_thickness_image, convertTo1D, show_cuts, show_cuts_position, rotation_matrix_from_vectors, orientation_slice, getOrientationMainVector, drawAxis, getClosestPointInRadius, color3DModelWithThickness, getArea, show_cuts_position_restored, WriteImage, sort_contours2

import configparser

#import variable image to compare in alignment
#from referenceBone import image

####################
# READ CONFIG FILE #
####################

# instantiate
config = configparser.RawConfigParser()

# parse existing file
config.read('../config/file.ini')

# read values from a section
output_path = config.get('dicom', 'output_path')
if not os.path.exists(output_path+"thickness"):
    os.makedirs(output_path+"thickness")
if not os.path.exists(output_path+"profiles"):
    os.makedirs(output_path+"profiles")
resources_path = config.get('dicom', 'resources_path')

# read values from a section
spacing_n = config.get('pre-process', 'spacing')
spacing_n = json.loads(spacing_n)

# read values from a section
num_views_thickness = config.getint('thickness', 'num_views_thickness')

# read values from a section
reference_bone = config.get('all dicom', 'reference_bone')

# read values from a section
vector1pca = config.get('reference vectors', 'orientation_vector')
vector1pca = json.loads(vector1pca)
vector1normal = config.get('reference vectors', 'alignment_vector')
vector1normal = json.loads(vector1normal)


##########################
# INITIALIZE LOGGER FILE #
##########################

logger = logging.getLogger("demo2")
fh = FileHandler('../logs/corrections&Thickness.html', mode="w")

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
fh.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(fh)


# Set spacing
spacing2 = [spacing_n[2], spacing_n[1], spacing_n[0]] #[0.25, 0.25, 0.5]
thickness_spacing2 = spacing_n[2]                     # 0.25


##################
# REFERENCE BONE #
##################

logger.debug(VisualRecord(">>> REFERENCE BONE IS:  %s" %(reference_bone)))

# Obtain the poly data of the reference bone
reader = vtk.vtkSTLReader()
reader.SetFileName(reference_bone)              
reader.Update()

#Extract Poly data
poly_data = reader.GetOutput()
bounds = poly_data.GetBounds()

# Calculate Center Of Mass
centerOfMass = vtk.vtkCenterOfMass()
centerOfMass.SetInputData(poly_data)
centerOfMass.SetUseScalarsAsWeights(False)
centerOfMass.Update()
G = centerOfMass.GetCenter()

whiteImage=vtk.vtkImageData()
whiteImage.SetSpacing(spacing2[0],spacing2[1],spacing2[2])

dim=[]
dim.append(int(math.ceil((bounds[1] - bounds[0]) /spacing2[0])))
dim.append(int(math.ceil((bounds[3] - bounds[2]) /spacing2[1])))
dim.append(int(math.ceil((bounds[5] - bounds[4]) /spacing2[2])))

whiteImage.SetDimensions(dim);
whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)

origin=[]
origin.append(bounds[0] + spacing2[0] / 2)
origin.append(bounds[2] + spacing2[1] / 2)
origin.append(bounds[4] + spacing2[2] / 2)
whiteImage.SetOrigin(origin)
whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

# fill the image with foreground voxels
inval = 255
outval = 0

ones=inval*np.ones(whiteImage.GetDimensions())
ones=np.array(ones,'uint8')

vtk_data_array = numpy_to_vtk(ones.flatten())
whiteImage.GetPointData().SetScalars(vtk_data_array)

# polygonal data --> image stencil:
pol2stenc = vtk.vtkPolyDataToImageStencil()
pol2stenc.SetInputData(reader.GetOutput())
pol2stenc.SetOutputOrigin(origin)
pol2stenc.SetOutputSpacing(spacing2)
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

#Obtain the slice of Center Of Mass
absolute_distance_to_COM=G[2]+bounds[4]
sliceCOM=round(absolute_distance_to_COM/spacing2[2])

image=np_scalars[sliceCOM,:,:]


##################
# CORRECT BONE 2 #
##################

# List ALL STLs 
output_path_stl = output_path+"\\stl\\"
onlyfiles_stl = [f for f in listdir(output_path_stl)] 


# Perform correction for all bones except for reference bone
for bone in onlyfiles_stl:
    if bone not in reference_bone:
        logger.debug(VisualRecord(">>> CORRECTIONS FOR BONE:  %s" %(bone)))
        
        reader2 = vtk.vtkSTLReader()
        reader2.SetFileName(output_path_stl+bone) 
        reader2.Update()
        
        #Extract Poly data
        poly_data2 = reader2.GetOutput()
        bounds2 = poly_data2.GetBounds()
        
        print("Max dimension in X: ", bounds2[1] - bounds2[0])
        print("Max dimension in Y: ", bounds2[3] - bounds2[2])
        print("Max dimension in Z: ", bounds2[5] - bounds2[4])
        
        whiteImage2=vtk.vtkImageData()
        whiteImage2.SetSpacing(spacing2[0],spacing2[1],spacing2[2])
        
        dim2=[]
        dim2.append(int(math.ceil((bounds2[1] - bounds2[0]) /spacing2[0])))
        dim2.append(int(math.ceil((bounds2[3] - bounds2[2]) /spacing2[1])))
        dim2.append(int(math.ceil((bounds2[5] - bounds2[4]) /spacing2[2])))
        
        whiteImage2.SetDimensions(dim2);
        whiteImage2.SetExtent(0, dim2[0] - 1, 0, dim2[1] - 1, 0, dim2[2] - 1)
        print("Voxels: ",dim2)
        
        origin2=[]
        origin2.append(bounds2[0] + spacing2[0] / 2)
        origin2.append(bounds2[2] + spacing2[1] / 2)
        origin2.append(bounds2[4] + spacing2[2] / 2)
        whiteImage2.SetOrigin(origin2)
        print("Origin: ",origin2)
        whiteImage2.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
        # fill the image with foreground voxels
        inval = 255
        outval = 0
        
        ones=inval*np.ones(whiteImage2.GetDimensions())
        ones=np.array(ones,'uint8')
        vtk_data_array = numpy_to_vtk(ones.flatten())
        whiteImage2.GetPointData().SetScalars(vtk_data_array)

        # polygonal data --> image stencil:
        pol2stenc2 = vtk.vtkPolyDataToImageStencil()
        pol2stenc2.SetInputData(reader2.GetOutput())
        pol2stenc2.SetOutputOrigin(origin2)
        pol2stenc2.SetOutputSpacing(spacing2)
        pol2stenc2.SetOutputWholeExtent(whiteImage2.GetExtent())
        pol2stenc2.Update()
        
        # cut the corresponding white image and set the background:
        imgstenc2 = vtk.vtkImageStencil()
        imgstenc2.SetInputData(whiteImage2)
        imgstenc2.SetStencilConnection(pol2stenc2.GetOutputPort())
        imgstenc2.ReverseStencilOff()
        imgstenc2.SetBackgroundValue(0)
        imgstenc2.Update()
        
        scalars2 = imgstenc2.GetOutput().GetPointData().GetScalars()
        np_scalars2 = vtk_to_numpy(scalars2)     
        np_scalars2 = np_scalars2.reshape(dim2[2], dim2[1], dim2[0]) 
        np_scalars2 = np_scalars2.transpose(0,2,1)
        
        
        #FIRST STEP: 
            # Calculate the Normal of bone 2 (poly_data2) 
            # and correct it in alignment with respect to bone 1
        BoundingBox2 = vtk.vtkOBBTree()
        BoundingBox2.SetDataSet(poly_data2)
        BoundingBox2.SetMaxLevel(1)
        BoundingBox2.BuildLocator()
        BoundingBoxPolyData2 = vtk.vtkPolyData()
        BoundingBox2.GenerateRepresentation(0, BoundingBoxPolyData2)
        
        BoundingBoxNormals2 = vtk.vtkPolyDataNormals()
        BoundingBoxNormals2.ComputeCellNormalsOn()
        BoundingBoxNormals2.SetInputData(BoundingBoxPolyData2)
        BoundingBoxNormals2.Update()
        BoundingBoxNormalsData2 = BoundingBoxNormals2.GetOutput().GetCellData().GetNormals()
        array2=vtk_to_numpy(BoundingBoxNormalsData2)
        
        print(BoundingBoxNormalsData2)
        
        areas2=[]
        normals2=[]
        for i in range(6):
            side2=vtk_to_numpy(BoundingBoxPolyData2.GetCell(i).GetPoints().GetData())
            area2=getArea(side2)
            areas2.append(area2)
            normal2=BoundingBoxNormals2.GetOutput().GetCellData().GetNormals().GetTuple(i)
            normals2.append(normal2)
            print("cell ",i," area ",area2," normal: ",normal2)
            
        vector2normal=normals2[areas2.index(min(areas2))]
        print("SECOND BONE ALIGNING REFERENCE: ", vector2normal)
        
        print("main ALIGNING REFERENCE: ", vector1normal)
        
        #Perform the correction in alignment
        angle=np.arccos(np.dot(vector1normal, vector2normal))
        print("Angle: ",np.rad2deg(angle))
        if np.rad2deg(angle) > 90:
            vector2normal=[-i for i in vector2normal]
            angle=np.arccos(np.dot(vector1normal, vector2normal))
            print("New Angle: ",np.rad2deg(angle))
        
        #Plot normals
        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(0, 0, 0, vector1normal[0], vector1normal[1], vector1normal[2],  normalize = True)
        ax.quiver(0, 0, 0, vector2normal[0], vector2normal[1], vector2normal[2],  normalize = True)
        ax.axes.set_xlim3d(left=-1, right=1) 
        ax.axes.set_ylim3d(bottom=-1, top=1) 
        ax.axes.set_zlim3d(bottom=-1, top=1) 
        #plt.show()
        
        #Create the rotation matrix
        from_vector=vector2normal
        to_vector=vector1normal #referencia
        RotationMatrix=rotation_matrix_from_vectors(from_vector,to_vector)
        
        matrix = vtk.vtkMatrix4x4()
        for i in range(0,3):
            for j in range(0,3):
                matrix.SetElement(i,j, RotationMatrix[i,j])
        matrix.SetElement(0,3,0)
        matrix.SetElement(1,3,0)
        matrix.SetElement(2,3,0)
        matrix.SetElement(3,0,0)
        matrix.SetElement(3,1,0)
        matrix.SetElement(3,2,0)
        matrix.SetElement(3,3,1)
        print(matrix)
        
        #Do a copy to save the data
        poly_data2_copy=vtk.vtkPolyData()
        poly_data2_copy.DeepCopy(poly_data2)
        
        #Apply the Rotation Matrix to do the correction
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(poly_data2_copy)
        transformFilter.Update()
        poly_data2_aligned=transformFilter.GetOutput()
        
        # Combine the two bones
        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(poly_data)           #bone1
        appendFilter.AddInputData(poly_data2_aligned)   #bone2 aligned
        appendFilter.Update()
        
        colors = vtk.vtkNamedColors()
        
        # Create a mapper and actor
        mapper3 = vtk.vtkPolyDataMapper()
        mapper3.SetInputData(appendFilter.GetOutput())
        mapper3.SetColorModeToDirectScalars()
        
        # Create a renderer, render window, and interactor
        renderer3 = vtk.vtkRenderer()
        renderer3.SetBackground(colors.GetColor3d('White'))
        
        actor3 = vtk.vtkActor()
        actor3.SetMapper(mapper3)
        actor3.GetProperty().SetColor(colors.GetColor3d("wheat"))
        renderer3.AddActor(actor3)
        
        renderWindow3 = vtk.vtkRenderWindow()
        renderWindow3.AddRenderer(renderer3)
        
        renderWindowInteractor3 = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor3.SetRenderWindow(renderWindow3)
        
        renderer3.ResetCamera()
        renderer3.ResetCameraClippingRange()
        camera = renderer3.GetActiveCamera()
        camera.Elevation(45)
        camera.Azimuth(90)
        camera.Roll(-45)
        renderer3.SetActiveCamera(camera)
        #renderWindow3.SetSize(480, 480)
        #renderer.SetBackground(.3, .2, .1) 
        
        # Render and interact
        renderWindow3.Render()
        filename = resources_path+'correctAlignment.png'
        WriteImage(filename, renderWindow3, rgba=False)
        cv_correctAlignment = cv.imread(resources_path+"correctAlignment.png")
        resized = cv.resize(cv_correctAlignment, (350,350), interpolation = cv.INTER_AREA)
        logger.debug(VisualRecord("Correction in alignment", resized, fmt="png"))
        #renderWindowInteractor3.Start()
        
        
        #SECOND STEP: 
            # On the poly_data2_aligned ALIGNED with REFERENCE 1 
            # repeat the steps to generate the STENCIL and export to NUMPY
        bounds2aligned = poly_data2_aligned.GetBounds()
        
        print("Max dimension in X: ", bounds2aligned[1] - bounds2aligned[0])
        print("Max dimension in Y: ", bounds2aligned[3] - bounds2aligned[2])
        print("Max dimension in Z: ", bounds2aligned[5] - bounds2aligned[4])
        
        centerOfMass2aligned = vtk.vtkCenterOfMass()
        centerOfMass2aligned.SetInputData(poly_data2_aligned)
        centerOfMass2aligned.SetUseScalarsAsWeights(False)
        centerOfMass2aligned.Update()
        G2 = centerOfMass2aligned.GetCenter()
        
        whiteImage2=vtk.vtkImageData()
        whiteImage2.SetSpacing(spacing2[0],spacing2[1],spacing2[2])
        
        dim2=[]
        dim2.append(int(math.ceil((bounds2aligned[1] - bounds2aligned[0]) /spacing2[0])))
        dim2.append(int(math.ceil((bounds2aligned[3] - bounds2aligned[2]) /spacing2[1])))
        dim2.append(int(math.ceil((bounds2aligned[5] - bounds2aligned[4]) /spacing2[2])))
        
        whiteImage2.SetDimensions(dim2);
        whiteImage2.SetExtent(0, dim2[0] - 1, 0, dim2[1] - 1, 0, dim2[2] - 1)
        print("Voxels: ",dim2)
        
        origin2aligned=[]
        origin2aligned.append(bounds2aligned[0] + spacing2[0] / 2)
        origin2aligned.append(bounds2aligned[2] + spacing2[1] / 2)
        origin2aligned.append(bounds2aligned[4] + spacing2[2] / 2)
        whiteImage2.SetOrigin(origin2aligned)
        print("Origin: ",origin2aligned)
        whiteImage2.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
        inval = 255
        outval = 0
        
        ones=inval*np.ones(whiteImage2.GetDimensions())
        ones=np.array(ones,'uint8')
        
        vtk_data_array = numpy_to_vtk(ones.flatten())
        whiteImage2.GetPointData().SetScalars(vtk_data_array)
        
        # polygonal data --> image stencil:
        pol2stenc2aligned = vtk.vtkPolyDataToImageStencil()
        pol2stenc2aligned.SetInputData(poly_data2_aligned)
        pol2stenc2aligned.SetOutputOrigin(origin2aligned)
        pol2stenc2aligned.SetOutputSpacing(spacing2)
        pol2stenc2aligned.SetOutputWholeExtent(whiteImage2.GetExtent())
        pol2stenc2aligned.Update()
        
        # cut the corresponding white image and set the background:
        imgstenc2aligned = vtk.vtkImageStencil()
        imgstenc2aligned.SetInputData(whiteImage2)
        imgstenc2aligned.SetStencilConnection(pol2stenc2aligned.GetOutputPort())
        imgstenc2aligned.ReverseStencilOff()
        imgstenc2aligned.SetBackgroundValue(0)
        imgstenc2aligned.Update()
        
        scalars2aligned = imgstenc2aligned.GetOutput().GetPointData().GetScalars()
        np_scalars2aligned = vtk_to_numpy(scalars2aligned)     
        np_scalars2aligned = np_scalars2aligned.reshape(dim2[2], dim2[1], dim2[0]) 
        np_scalars2aligned = np_scalars2aligned.transpose(0,2,1)
        print("Shape: ",np_scalars2aligned.shape)
        
        #sample_stack(np_scalars2aligned, rows=10, cols=10, start_with=1, show_every=6, color=False)

        
        #THIRD STEP: 
            # On the NUMPY array of the STENCIL of the ALIGNED PolyData np_scalars2aligned, 
            # take the PCA from its Center of Mass and calculate the orientation vectors
        print("Max dimension in Z: ", bounds2aligned[5] - bounds2aligned[4])
        print("minz", bounds2aligned[4])
        print("max", bounds2aligned[5])
        print("Center of Mass", G2)
        print("Spacing", spacing2)
        
        absolute_distance_to_COM2=G2[2]+bounds2aligned[4]
        print("Absolute distance COM: ",absolute_distance_to_COM2)
        sliceCOM2=round(absolute_distance_to_COM2/spacing2[2])
        print("slice COM: ",sliceCOM2)
        
        #AUTO DETECT change_leg
        image2=np_scalars2aligned[sliceCOM2,:,:]
        
        image_rgb2 = cv.cvtColor(image2, cv.COLOR_GRAY2BGR )
        gray2 = cv.cvtColor(image_rgb2, cv.COLOR_BGR2GRAY)
        _, bw2 = cv.threshold(gray2, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        contours2, _ = cv.findContours(bw2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        
        areas2=[]
        for i, c in enumerate(contours2):
            areas2.append(cv.contourArea(c))
        index2=areas2.index(max(areas2))
            
        # Find the orientation of each shape
        vector2pca=getOrientationMainVector(contours2[index2], image_rgb2, arrowsize=4)
        
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(image2)
        plt.subplot(1,2,2)
        plt.imshow(image_rgb2)
        #plt.show()
        plt.savefig(resources_path+"beforeChangeLeg.png")
        cv_beforeChangeLeg = cv.imread(resources_path+"beforeChangeLeg.png")
        resized = cv.resize(cv_beforeChangeLeg, (450,350), interpolation = cv.INTER_AREA)
        logger.debug(VisualRecord("Before Change Leg", resized, fmt="png"))
        
        unit_vector_1 = vector1pca / np.linalg.norm(vector1pca)
        unit_vector_2 = vector2pca / np.linalg.norm(vector2pca)
        unit_vector_1=np.append(unit_vector_1,0)
        unit_vector_2=np.append(unit_vector_2,0)
        angle=vg.signed_angle(unit_vector_1, unit_vector_2, look=vg.basis.z)
        print(angle)
        
        # First "demo" rotation
        np_scalars2aligned_oriented_demo = rotate(np_scalars2aligned, angle=angle, axes=(1, 2), reshape=False)
        
        image_restored=np_scalars2aligned_oriented_demo[sliceCOM2,:,:]
        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(image)
        plt.title("Leg 1")
        plt.subplot(1,3,2)
        plt.imshow(image2) #image that is not rotated
        plt.title("Leg 2")
        plt.subplot(1,3,3)
        plt.imshow(image_restored) # image rotated
        plt.title("Leg 2 Corrected")
        #plt.show()
        plt.savefig(resources_path+"demo.png")
        cv_demo = cv.imread(resources_path+"demo.png")
        resized = cv.resize(cv_demo, (450,350), interpolation = cv.INTER_AREA)
        logger.debug(VisualRecord("Correction in Orientation", resized, fmt="png"))
        
        array_thickness2=[]
        array_contours2=[]
        array_coordinates2=[]
        array_contourid2=[]
        for i in tqdm(range(len(np_scalars2))):
            contourid2,coordinates2, contour2,thick2=compute_thickness_image(np_scalars2aligned_oriented_demo[i,:,:],contourid=-1,grow=False)
            array_thickness2.append(thick2*thickness_spacing2)
            array_contours2.append(contour2)
            array_coordinates2.append(coordinates2)
            array_contourid2.append(contourid2) # array with the index of the contour with the largest area
        
        # Flip the "demo" image_restored (image rotated) and obtain the contours
        img = np.flip(image_restored,0)
        image_rgb = cv.cvtColor(img, cv.COLOR_GRAY2BGR )
        gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
        _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        coordinates_2, _ = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                
        # Sort the "demo" contours of the image rotated and flip
        left_right,_ = sort_contours2(coordinates_2) #Devuelve en posicion 0 el contorno situado más a la izquierda
                                                        # y en la siguiente posición el contorno situado más a la derecha
        # Obtain the number of the contour more to the left (left_right[0])
        for i in range(0, len(coordinates_2)):
            try: 
                c = sum(coordinates_2[i]-left_right[0])
                if c.sum() == 0:
                    ind = i
            except: pass
    
        # If the number of the contour more to the left of the flip and rotated image is the same as the image2 (not rotated), changle_leg = True
        if array_contourid2[sliceCOM2] == ind:
            change_leg = True
        else:
            change_leg = False
        
        #TRUE IF COMPARING RIGHT AND LEFT LEG
        #change_leg=True  # Poner a True si la referencia y el hueso 2 son de lados diferentes (pierna izquierda y pierna derecha)
        if change_leg:
            np_scalars2aligned=np.flip(np_scalars2aligned,1)
        
        image2=np_scalars2aligned[sliceCOM2,:,:]
        
        image_rgb2 = cv.cvtColor(image2, cv.COLOR_GRAY2BGR )
        gray2 = cv.cvtColor(image_rgb2, cv.COLOR_BGR2GRAY)
        _, bw2 = cv.threshold(gray2, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        contours2, _ = cv.findContours(bw2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        
        areas2=[]
        for i, c in enumerate(contours2):
            areas2.append(cv.contourArea(c))
        index2=areas2.index(max(areas2))
            
        # Find the orientation of each shape
        vector2pca=getOrientationMainVector(contours2[index2], image_rgb2, arrowsize=4)
        
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(image2)
        plt.subplot(1,2,2)
        plt.imshow(image_rgb2)
        #plt.show()
        plt.savefig(resources_path+"OrientationV.png")
        cv_OrV = cv.imread(resources_path+"OrientationV.png")
        resized = cv.resize(cv_OrV, (450,350), interpolation = cv.INTER_AREA)
        logger.debug(VisualRecord("Orientation Vector", resized, fmt="png"))
        
        #TRUE IF MAIN VECTOR IS IN OTHER DIRECTION
        Correct_direction_manually=False # sólo poner a True si la referencia de PCA apunta hacia la parte mala de la tibia en lugar de la parte buena
        print("SECOND BONE ORIENTING REFERENCE: ", vector2pca)
        if Correct_direction_manually:
            vector2pca=[-coord for coord in vector2pca]
            print("SECOND BONE ORIENTING REFERENCE CORRECTED: ", vector2pca) 
        
        unit_vector_1 = vector1pca / np.linalg.norm(vector1pca)
        unit_vector_2 = vector2pca / np.linalg.norm(vector2pca)
        unit_vector_1=np.append(unit_vector_1,0)
        unit_vector_2=np.append(unit_vector_2,0)
        angle=vg.signed_angle(unit_vector_1, unit_vector_2, look=vg.basis.z)
        print(angle)
        
        np_scalars2aligned_oriented = rotate(np_scalars2aligned, angle=angle, axes=(1, 2), reshape=False)
        
        image_restored=np_scalars2aligned_oriented[sliceCOM2,:,:]
        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(image)
        plt.title("Leg 1")
        plt.subplot(1,3,2)
        plt.imshow(image2)
        plt.title("Leg 2")
        plt.subplot(1,3,3)
        plt.imshow(image_restored)
        plt.title("Leg 2 Corrected")
        #plt.show()
        plt.savefig(resources_path+"cOr.png")
        cv_cOr = cv.imread(resources_path+"cOr.png")
        resized = cv.resize(cv_cOr, (450,350), interpolation = cv.INTER_AREA)
        logger.debug(VisualRecord("Correction in Orientation", resized, fmt="png"))
        
        
        #FOURTH STEP: 
                # With these vectors go to the ALIGNED POLYDATA (poly_data2_aligned) and ORIENT IT
        #Create a backup
        poly_data2_aligned_back=vtk.vtkPolyData()
        poly_data2_aligned_back.DeepCopy(poly_data2_aligned)
            
        #But before correcting the Aligned PolyData, rotating it in that direction
        #we must take into account if we had to create the mirror image
        #in the case that bone 2 and the reference bone are from different legs
        if change_leg:
            transform = vtk.vtkTransform()
            transform.Scale(-1,1,1)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(poly_data2_aligned_back)
            tf.SetTransform(transform);
            tf.Update()
            poly_data2_aligned_mirrored=tf.GetOutput()
            reverse=vtk.vtkReverseSense()
            reverse.SetInputConnection(tf.GetOutputPort())
            reverse.ReverseNormalsOn();
            reverse.Update()
            print(reverse) 
            
        if change_leg:
            # Combine the two boundingboxes
            appendFilter = vtk.vtkAppendPolyData()
            appendFilter.AddInputData(poly_data2_aligned_back) #Same bone one leg
            appendFilter.AddInputData(reverse.GetOutput()) #Same bone other leg (Flipped with change_leg True)
            appendFilter.Update()
            
        if change_leg:
            poly_data2_aligned=reverse.GetOutput()
            
        vector1pcaVTK=list(vector1pca)+[0.0]
        vector2pcaVTK=list(vector2pca)+[0.0]
        vec1=vector2pcaVTK
        vec2=vector1pcaVTK
        
        #Notice as now, if the angle is too big (probably bad PCA detection at 180º)
        #and the variable Correct_direction_manually is not set to True,
        #then we correct it automatically.
        #This is done to correct in the first iteration the maximum possible errors when 
        #the PCA points in the wrong direction (backward of the bone).
        unit_vector_1 = vector1pcaVTK / np.linalg.norm(vector1pcaVTK)
        unit_vector_2 = vector2pcaVTK / np.linalg.norm(vector2pcaVTK)
        angle=vg.signed_angle(unit_vector_1, unit_vector_2, look=vg.basis.z)
        print("Angle: ",angle)
        if abs(angle) > 90 and Correct_direction_manually==False:
            vector2pcaVTK=[-i for i in vector2pcaVTK]
            unit_vector_1 = vector1pcaVTK / np.linalg.norm(vector1pcaVTK)
            unit_vector_2 = vector2pcaVTK / np.linalg.norm(vector2pcaVTK)
            angle=vg.signed_angle(unit_vector_1, unit_vector_2, look=vg.basis.z)
            print("New Angle: ",angle)
        
        #Perform the correction in orientation
        from_vector=vector1pcaVTK #referencia
        to_vector=vector2pcaVTK
        RotationMatrix=rotation_matrix_from_vectors(from_vector,to_vector)
        matrix = vtk.vtkMatrix4x4()
        for i in range(0,3):
            for j in range(0,3):
                matrix.SetElement(i,j, RotationMatrix[i,j])
        matrix.SetElement(0,3,0)
        matrix.SetElement(1,3,0)
        matrix.SetElement(2,3,0)
        matrix.SetElement(3,0,0)
        matrix.SetElement(3,1,0)
        matrix.SetElement(3,2,0)
        matrix.SetElement(3,3,1)
        print(matrix)
        
        #Correct the ALIGNED POLYDATA (poly_data2_aligned) using the vectors calculated with PCA
        poly_data2_copy=vtk.vtkPolyData()
        poly_data2_copy.DeepCopy(poly_data2_aligned)
        
        #Correction
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(poly_data2_copy)
        transformFilter.Update()
        poly_data2_aligned_oriented=transformFilter.GetOutput()
        
        #Be careful, in some bones the oriented one, when it is very large,
        #can misalign bone 2 with the reference...
        #Let's make a second alignment:
        BoundingBox2 = vtk.vtkOBBTree()
        BoundingBox2.SetDataSet(poly_data2_aligned_oriented)
        BoundingBox2.SetMaxLevel(1)
        BoundingBox2.BuildLocator()
        BoundingBoxPolyData2 = vtk.vtkPolyData()
        BoundingBox2.GenerateRepresentation(0, BoundingBoxPolyData2)
        
        BoundingBoxNormals2 = vtk.vtkPolyDataNormals()
        BoundingBoxNormals2.ComputeCellNormalsOn()
        BoundingBoxNormals2.SetInputData(BoundingBoxPolyData2)
        BoundingBoxNormals2.Update()
        BoundingBoxNormalsData2 = BoundingBoxNormals2.GetOutput().GetCellData().GetNormals()
        array2=vtk_to_numpy(BoundingBoxNormalsData2)
        
        print(BoundingBoxNormalsData2)
        
        areas2=[]
        normals2=[]
        for i in range(6):
            side2=vtk_to_numpy(BoundingBoxPolyData2.GetCell(i).GetPoints().GetData())
            area2=getArea(side2)
            areas2.append(area2)
            normal2=BoundingBoxNormals2.GetOutput().GetCellData().GetNormals().GetTuple(i)
            normals2.append(normal2)
            print("cell ",i," area ",area2," normal: ",normal2)
        
        vector2normal=normals2[areas2.index(min(areas2))]
        print("SECOND BONE ALIGNING REFERENCE: ", vector2normal)
        print("MAIN ALIGNING REFERENCE: ", vector1normal)
        
        angle=np.arccos(np.dot(vector1normal, vector2normal))
        print("Angle: ",np.rad2deg(angle))
        if np.rad2deg(angle) > 90:
            vector2normal=[-i for i in vector2normal]
            angle=np.arccos(np.dot(vector1normal, vector2normal))
            print("New Angle: ",np.rad2deg(angle))
            
        from_vector=vector2normal
        to_vector=vector1normal 
        RotationMatrix=rotation_matrix_from_vectors(from_vector,to_vector)
        RotationMatrix
        matrix = vtk.vtkMatrix4x4()
        for i in range(0,3):
            for j in range(0,3):
                matrix.SetElement(i,j, RotationMatrix[i,j])
        matrix.SetElement(0,3,0)
        matrix.SetElement(1,3,0)
        matrix.SetElement(2,3,0)
        matrix.SetElement(3,0,0)
        matrix.SetElement(3,1,0)
        matrix.SetElement(3,2,0)
        matrix.SetElement(3,3,1)
        print(matrix)
        
        poly_data2_copy=vtk.vtkPolyData()
        poly_data2_copy.DeepCopy(poly_data2_aligned_oriented)
        
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(poly_data2_copy)
        transformFilter.Update()
        poly_data2_aligned_oriented_2=transformFilter.GetOutput()
        
        # Combine the two boundingboxes
        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(poly_data2_aligned_oriented_2) #Bone 2nd alignement
        appendFilter.AddInputData(poly_data) #Bone1
        appendFilter.Update()
        
        colors = vtk.vtkNamedColors()
        
        # Create a mapper and actor
        mapper3 = vtk.vtkPolyDataMapper()
        mapper3.SetInputData(appendFilter.GetOutput())
        mapper3.SetColorModeToDirectScalars()
        
        # Create a renderer, render window, and interactor
        renderer3 = vtk.vtkRenderer()
        renderer3.SetBackground(colors.GetColor3d('White'))
        
        actor3 = vtk.vtkActor()
        actor3.SetMapper(mapper3)
        actor3.GetProperty().SetColor(colors.GetColor3d("wheat"))
        renderer3.AddActor(actor3)
        
        renderWindow3 = vtk.vtkRenderWindow()
        renderWindow3.AddRenderer(renderer3)
        
        renderWindowInteractor3 = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor3.SetRenderWindow(renderWindow3)
        
        renderer3.ResetCamera()
        renderer3.ResetCameraClippingRange()
        camera = renderer3.GetActiveCamera()
        camera.Elevation(45)
        camera.Azimuth(90)
        camera.Roll(-45)
        renderer3.SetActiveCamera(camera)
        #renderWindow3.SetSize(480, 480)
        #renderer.SetBackground(.3, .2, .1) 
        
        # Render and interact
        renderWindow3.Render()
        filename = resources_path+'aligOrien.png'
        WriteImage(filename, renderWindow3, rgba=False)
        cv_aligOrien = cv.imread(resources_path+"aligOrien.png")
        resized = cv.resize(cv_aligOrien, (350,350), interpolation = cv.INTER_AREA)
        logger.debug(VisualRecord("Bone 2 aligned and oriented respect reference bone", resized, fmt="png"))
        #renderWindowInteractor3.Start()
        
        # Now we have bone 2 aligned and oriented respect reference bone
        
        #FIFTH STEP:
            # With the POLYDATA ALIGNED AND ORIENTED (poly_data2_aligned_oriented_2),
            # regenerate the STENCIL export to NUMPY and calculate thickness
        bounds2aligned_oriented = poly_data2_aligned_oriented_2.GetBounds()

        whiteImage2=vtk.vtkImageData()
        whiteImage2.SetSpacing(spacing2[0],spacing2[1],spacing2[2])
        
        dim2=[]
        dim2.append(int(math.ceil((bounds2aligned_oriented[1] - bounds2aligned_oriented[0]) /spacing2[0])))
        dim2.append(int(math.ceil((bounds2aligned_oriented[3] - bounds2aligned_oriented[2]) /spacing2[1])))
        dim2.append(int(math.ceil((bounds2aligned_oriented[5] - bounds2aligned_oriented[4]) /spacing2[2])))
        
        whiteImage2.SetDimensions(dim2);
        whiteImage2.SetExtent(0, dim2[0] - 1, 0, dim2[1] - 1, 0, dim2[2] - 1)
        print("Voxels: ",dim2)
        
        origin2aligned_oriented=[]
        origin2aligned_oriented.append(bounds2aligned_oriented[0] + spacing2[0] / 2)
        origin2aligned_oriented.append(bounds2aligned_oriented[2] + spacing2[1] / 2)
        origin2aligned_oriented.append(bounds2aligned_oriented[4] + spacing2[2] / 2)
        whiteImage2.SetOrigin(origin2aligned_oriented)
        print("Origin: ",origin2aligned_oriented)
        
        whiteImage2.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
        inval = 255
        outval = 0
        
        ones=inval*np.ones(whiteImage2.GetDimensions())
        ones=np.array(ones,'uint8')
        vtk_data_array = numpy_to_vtk(ones.flatten())
        whiteImage2.GetPointData().SetScalars(vtk_data_array)
        
        # polygonal data --> image stencil:
        pol2stenc2aligned_oriented = vtk.vtkPolyDataToImageStencil()
        pol2stenc2aligned_oriented.SetInputData(poly_data2_aligned_oriented_2)
        pol2stenc2aligned_oriented.SetOutputOrigin(origin2aligned_oriented)
        pol2stenc2aligned_oriented.SetOutputSpacing(spacing2)
        pol2stenc2aligned_oriented.SetOutputWholeExtent(whiteImage2.GetExtent())
        pol2stenc2aligned_oriented.Update()
        # cut the corresponding white image and set the background:
        imgstenc2aligned_oriented = vtk.vtkImageStencil()
        imgstenc2aligned_oriented.SetInputData(whiteImage2)
        imgstenc2aligned_oriented.SetStencilConnection(pol2stenc2aligned_oriented.GetOutputPort())
        imgstenc2aligned_oriented.ReverseStencilOff()
        imgstenc2aligned_oriented.SetBackgroundValue(0)
        imgstenc2aligned_oriented.Update()
        
        scalars2aligned_oriented = imgstenc2aligned_oriented.GetOutput().GetPointData().GetScalars()
        np_scalars2_aligned_oriented = vtk_to_numpy(scalars2aligned_oriented)     
        np_scalars2_aligned_oriented = np_scalars2_aligned_oriented.reshape(dim2[2], dim2[1], dim2[0]) 
        np_scalars2_aligned_oriented = np_scalars2_aligned_oriented.transpose(0,2,1)
        print("Shape: ",np_scalars2_aligned_oriented.shape)
        
        sample_stack(np_scalars2_aligned_oriented, rows=10, cols=10, start_with=1, show_every=6, color=False)
        
        array_thickness2=[]
        array_contours2=[]
        array_coordinates2=[]
        array_contourid2=[]
        for i in tqdm(range(len(np_scalars2))):
            contourid2,coordinates2, contour2,thick2=compute_thickness_image(np_scalars2_aligned_oriented[i,:,:],contourid=-1,grow=False)
            array_thickness2.append(thick2*thickness_spacing2)
            array_contours2.append(contour2)
            array_coordinates2.append(coordinates2)
            array_contourid2.append(contourid2)
            
        centerOfMass2 = vtk.vtkCenterOfMass()
        centerOfMass2.SetInputData(poly_data2_aligned_oriented_2)
        centerOfMass2.SetUseScalarsAsWeights(False)
        centerOfMass2.Update()
        G2_aligned_oriented = centerOfMass2.GetCenter()
        print(G2_aligned_oriented)
        
        print("Max dimension in Z: ", bounds2aligned_oriented[5] - bounds2aligned_oriented[4])
        print("minz", bounds2aligned_oriented[4])
        print("max", bounds2aligned_oriented[5])
        print("Center of Mass", G2_aligned_oriented)
        print("Spacing", spacing2)
        
        absolute_distance_to_COM2_aligned_oriented=G2_aligned_oriented[2]+bounds2aligned_oriented[4]
        print("Absolute distance COM: ",absolute_distance_to_COM2_aligned_oriented)
        sliceCOM2aligned_oriented=round(absolute_distance_to_COM2_aligned_oriented/spacing2[2])
        print("slice COM: ",sliceCOM2aligned_oriented)
        
        image2aligned_oriented=np_scalars2_aligned_oriented[sliceCOM2aligned_oriented,:,:]
        
        image_rgb2aligned_oriented = cv.cvtColor(image2aligned_oriented, cv.COLOR_GRAY2BGR )
        gray2aligned_oriented = cv.cvtColor(image_rgb2aligned_oriented, cv.COLOR_BGR2GRAY)
        _, bw2aligned_oriented = cv.threshold(gray2aligned_oriented, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        contours2aligned_oriented, _ = cv.findContours(bw2aligned_oriented, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        
        areas2=[]
        for i, c in enumerate(contours2aligned_oriented):
            areas2.append(cv.contourArea(c))
        index2=areas2.index(max(areas2))
         
        # Find the orientation of each shape
        vector2pca=getOrientationMainVector(contours2aligned_oriented[index2], image_rgb2aligned_oriented, arrowsize=4)
        
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(image2aligned_oriented)
        plt.subplot(1,2,2)
        plt.imshow(image_rgb2aligned_oriented)
        #plt.show()
        plt.savefig(resources_path+"pcaF.png")
        cv_pcaF = cv.imread(resources_path+"pcaF.png")
        resized = cv.resize(cv_pcaF, (450,350), interpolation = cv.INTER_AREA)
        logger.debug(VisualRecord("PCA vector of final bone corrected", resized, fmt="png"))
        
        image_final=np_scalars2_aligned_oriented[sliceCOM2aligned_oriented,:,:]
        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(image)  # COM bone1
        plt.title("Leg 1")
        plt.subplot(1,3,2)
        plt.imshow(image2) # COM bone2 after aligning
        plt.title("Leg 2")
        plt.subplot(1,3,3)
        plt.imshow(image_final) # COM bone2 after aligning and orienting
        plt.title("Leg 2 Oriented")
        #plt.show()
        plt.savefig(resources_path+"cOrF.png")
        cv_cOrF = cv.imread(resources_path+"cOrF.png")
        resized = cv.resize(cv_cOrF, (450,350), interpolation = cv.INTER_AREA)
        logger.debug(VisualRecord("#1 bone1, #2 bone2 after aligning, #3 bone2 after aligning and orienting", resized, fmt="png"))
        
        
        #STEP SIX: 
            # Plot3D
        array2=np.array(array_thickness2)
        np.savez_compressed(output_path+"thickness\\"+bone.split(".")[0], array2)
        logger.debug(VisualRecord(">>> THICKNESS saved in:  %s" %(output_path+"thickness\\"+bone.split(".")[0])))
        
        array_tmp2=array2.transpose(1, 2, 0)
        
        
        # Convert numpy array to VTK array (vtkFloatArray)
        vtk_data_array2 = numpy_support.numpy_to_vtk(
            num_array=array_tmp2.transpose(2, 1, 0).ravel(),  # ndarray contains the fitting result from the points. It is a 3D array
            deep=True,
            array_type=vtk.VTK_FLOAT)
        
        # Convert the VTK array to vtkImageData
        img_vtk2 = vtk.vtkImageData()
        img_vtk2.SetDimensions(array_tmp2.shape)
        img_vtk2.SetSpacing(spacing2)
        img_vtk2.SetOrigin(origin2aligned_oriented)
        img_vtk2.GetPointData().SetScalars(vtk_data_array2)
        
        print("VTK: ",img_vtk2.GetDimensions())
        
        surface2 = vtk.vtkMarchingCubes()
        surface2.SetInputData(imgstenc2aligned_oriented.GetOutput())
        surface2.ComputeNormalsOn()
        surface2.SetValue(0, 127.5)
        surface2.Update()
        
        probe = vtk.vtkProbeFilter()
        probe.SetInputData(surface2.GetOutput())
        probe.SetSourceData(img_vtk2)
        probe.Update()
        
        probe.GetOutput()
        rng = img_vtk2.GetScalarRange()
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
        filename = resources_path+'corrected3D.png'
        WriteImage(filename, renderWindow, rgba=False)
        cv_corrected3D = cv.imread(resources_path+"corrected3D.png")
        resized = cv.resize(cv_corrected3D, (350,350), interpolation = cv.INTER_AREA)
        logger.debug(VisualRecord("Corrected Bone Colored 3D", resized, fmt="png"))
        #renderWindowInteractor.Start()
        
        #STEP SEVEN:
            # Calculate 1D profiles of thickness
        dimz,dimx,dimy = np_scalars2_aligned_oriented.shape

        # With this we look for the point of the X axis that more contours of the slices intersect
        lengths=[]
        for referencex in tqdm(range(dimx)):
            array_thickness_1d,noreferences = convertTo1D(array_coordinates2,array_thickness2,countour_index=array_contourid2,reference_x = referencex, verbose=False)
            lengths.append(noreferences)
        referencex=list(range(dimx))[lengths.index(min(lengths))]
        print("The reference X that maximices the number of valid contours is: ",referencex)
        logger.debug(VisualRecord(">>> REFERENCE for the 1D profiles conversion was:  %s" %(referencex)))

        
        # Extraxt 1D profiles of thickness
        array_thickness_1d,_ = convertTo1D(array_coordinates2,array_thickness2,countour_index=array_contourid2,reference_x = referencex)

        # Plot the graphs with the thickness. 
        cortes = []                                         # Array with slices whose thickness is going to be represented
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
        with open(output_path+"profiles\\"+bone.split(".")[0]+".pkl", 'wb') as handle:
             pickle.dump(profiles, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(VisualRecord(">>> PROFILES DICTIONARY saved in:  %s" %(output_path+"profiles\\"+bone.split(".")[0]+".pkl")))
        
        # Show the cuts generated in 2D
        show_cuts(array_thickness2, cortes, num_views, spacing2, origin2aligned_oriented)
        plt.savefig(resources_path+"cuts.png")
        cv_cuts = cv.imread(resources_path+"cuts.png")
        resized = cv.resize(cv_cuts, (500,500), interpolation = cv.INTER_AREA)
        logger.debug(VisualRecord("2D Thickness Contours", resized, fmt="png"))


        # Show the position of the cuts in a 3D model
        show_cuts_position(cortes, num_views, G2_aligned_oriented, poly_data2_aligned_oriented_2, bounds2aligned_oriented, spacing2)
        cv_cuts_p = cv.imread("cuts_p.png")
        resized = cv.resize(cv_cuts_p, (350,350), interpolation = cv.INTER_AREA)
        logger.debug(VisualRecord("3D chosen profiles", resized, fmt="png"))

    else:
        
        print("Reference bone")
