import json
import vtk
import numpy as np
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import logging
from logging import FileHandler
from vlogging import VisualRecord
import cv2
import math
from scipy.ndimage import rotate
import os
import datetime
# import vg

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
output_path = config.get('dicom', 'output_path')
if not os.path.exists(output_path+"stl"):
    os.makedirs(output_path+"stl")

if not os.path.exists(output_path+"side"):
    os.makedirs(output_path+"side")

resources_path = config.get('dicom', 'resources_path')
data_path_dicom = config.get('dicom', 'data_path_dicom')

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

# read last file properly processed
continue_from = config.get('pre-process', 'start_from')

##########################
# INITIALIZE LOGGER FILE #
##########################

ct=datetime.datetime.now()
sufix=ct.strftime("%Y-%m-%d_%H-%M-%S")

logger = logging.getLogger("generateSTLs")
if len(continue_from)>0:
    fh = FileHandler(f'../logs/generateSTLs_cont_{sufix}.html', mode="w")
else:
    fh = FileHandler(f'../logs/generateSTLs_{sufix}.html', mode="w")    

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
fh.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(fh)


#####################
# GENERATE ALL STLs #
#####################

# List ALL DICOM TACs
data_path_dicom = data_path_dicom

onlyfiles_dicom = [f for f in listdir(data_path_dicom)] 

# Control of number of bones
id=0

# Examine each TAC
for i in onlyfiles_dicom[id:]:
    # Increase the value to identify the bone
    id+=1

    if len(continue_from)>0:
        logger.debug(VisualRecord("JUMPED BONE:  %s" %(i)))
        if i==continue_from:
            # Since we reach the last properly processed folder, start proccessing the next one
            logger.debug(VisualRecord("THIS IS A CONTINUED EXECUTION"))
            continue_from=""
        continue
    else:
        # Run this block with normal execution or when a continued execution is detected and the last
        # processed folder is reached

        logger.debug(VisualRecord("NEW BONE:  %s" %(i)))       
        logger.debug(VisualRecord("BONE ID: %s" %id ))
        
        # Path of slices
        data_path = data_path_dicom+i
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    
        # Print out the first 5 file names to verify we're in the right folder.
        print("Total of %d DICOM images.\nFirst 5 filenames:" % len(onlyfiles))
        logger.debug(VisualRecord("Total of %d DICOM images of bone id %d." % (len(onlyfiles), id)))
            
        # Load all DICOM from a folder into a list for manipulation
        slices=load_scan(data_path,onlyfiles)
        
        # Show metadata
        print(slices[0])

        # Show leg side
        ds=slices[0]
        tag = (0x0032, 0x1060) #procedure tag
        requested_procedure_description = ds.get(tag, None)
        text=requested_procedure_description.value
        if requested_procedure_description is not None:
            print("{0:35}- {1:20}".format(i,text))
            logger.debug(VisualRecord("{0:35}- {1:20}".format(i,text)))
        else:
            print("El tag (0032,1060) no está presente en el dataset.")
            logger.debug(VisualRecord("El tag (0032,1060) no está presente en el dataset"))
        
        if "right" in text.lower():
            legside="right"
        else:
            legside="left"
        
        # Create a numpy matrix of Hounsfield Units (HU)
        imgs = get_pixels_hu(slices)
            
        # Load images
        imgs_to_process = imgs
        n_of_images=imgs_to_process.shape[0]
        step=n_of_images//25
        
        # Show & log imgs_to_process

        sample_stack(imgs_to_process, rows=5, cols=5, start_with=1, show_every=step)
        plt.savefig(resources_path+"Sample_stack.png")
        cv_sample_stack = cv2.imread(resources_path+"Sample_stack.png")
        resized = cv2.resize(cv_sample_stack, (1020,1020), interpolation = cv2.INTER_AREA)
        logger.debug(VisualRecord("Images to process", resized, fmt="png"))
        
        # Plot & log histogram to see Hounsfield Units (HU)
        plt.figure()
        plt.hist(imgs_to_process.flatten(), bins=50, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        #plt.show()
        plt.savefig(resources_path+'Histogram.png')
        cv_histogram = cv2.imread(resources_path+'Histogram.png')
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
        
        # Find bounding box of a slice
        # img = imgs_after_resamp[260]
        # (x,y),(X,Y),W,H=find_bounding_box(img, hu=True,threshold=50, display=True, sizex=8, sizey=8, linewidth=2,title=False)

        # Show bounding box detection of images
        f=find_bounding_box_sample_stack(imgs_after_resamp, hu=True, show_box=True, threshold=threshold, rows=5, cols=5, start_with=1, show_every=imgs_after_resamp.shape[0]//25)
        # Show bounding box detection of images
        canvas = FigureCanvas(f)  # f es tu plt.figure
        canvas.draw()  # Renderiza la figura
        # Extrae los datos de la imagen como un array RGBA
        image_array = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        width, height = canvas.get_width_height()
        image_array = image_array.reshape((height, width, 4))  # Reorganiza en formato (alto, ancho, canales)
        # Convierte la imagen a formato compatible con OpenCV (BGR)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(image_bgr, (1020,1020), interpolation = cv2.INTER_AREA)
        logger.debug(VisualRecord("Bounded Box IMAGE", resized, fmt="png"))
        
        # Calculate size for the spacing used. Reference 0.5 --> size=30
        auto_size = int(0.5*30/spacing[1])
        if size <= auto_size-5 or size >= auto_size+5:
            size=auto_size
        
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

        
        # Show & log masked bones
        sample_stack(masked_bones, rows=5, cols=5, start_with=1, show_every=len(masked_bones)//25)
        plt.savefig(resources_path+"Sample_stack.png")
        cv_sample_stack = cv2.imread(resources_path+"Sample_stack.png")
        resized = cv2.resize(cv_sample_stack, (1020,1020), interpolation = cv2.INTER_AREA)
        logger.debug(VisualRecord("Masked bone", resized, fmt="png"))
        
        # Show & log masks
        sample_stack(masks, rows=5, cols=5, start_with=1, show_every=len(masks)//25)
        plt.savefig(resources_path+"Sample_stack.png")
        cv_sample_stack = cv2.imread(resources_path+"Sample_stack.png")
        resized = cv2.resize(cv_sample_stack, (1020,1020), interpolation = cv2.INTER_AREA)
        logger.debug(VisualRecord("Masks", resized, fmt="png"))
        
        # Show & log labels of the mask
        sample_stack(labels, rows=5, cols=5, start_with=1, show_every=len(labels)//25, color=True)
        plt.savefig(resources_path+"Sample_stack.png")
        cv_sample_stack = cv2.imread(resources_path+"Sample_stack.png")
        resized = cv2.resize(cv_sample_stack, (1020,1020), interpolation = cv2.INTER_AREA)
        logger.debug(VisualRecord("Labels", resized, fmt="png"))
        
        # Convert images into a numpy array
        imgs_after_mask=np.array(masked_bones_hu)
        
    
        ################
        # POST-PROCESS #
        ################
            
        # Load file and obtain the shape
        imgs_to_process = imgs_after_mask.astype(np.float64) 
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

        renderer.ResetCamera()
        renderer.ResetCameraClippingRange()
        camera = renderer.GetActiveCamera()
        camera.Elevation(45)
        camera.Azimuth(90)
        camera.Roll(-45)
        renderer.SetActiveCamera(camera)
        
        # Show the 3D model
        render_window.Render()

        # Capture the rendered scene to an image
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.Update()

        # Save the image to a PNG file
        filename = resources_path+'bone3D.png'
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()
        
        #WriteImage(filename, render_window, rgba=False)
        cv_bone3D = cv2.imread(filename)
        resized = cv2.resize(cv_bone3D, (350,350), interpolation = cv2.INTER_AREA)
        logger.debug(VisualRecord("Render Bone 3D: "+i, resized, fmt="png"))
        #interactor.Start()
        
        # Convert to STL file
        if convert_stl:
            renderer = vtk.vtkRenderer()
            actor, stripper = CreateTissueFromArray(imdata,threshold_between_min,threshold_between_max,"skeleton")
            print("Saving STL")
            
            #Save to STL in the output path
            writer = vtk.vtkSTLWriter()
            #linux
            writer.SetFileName(output_path+"stl/"+"%s.stl" %(i))
            #windows
            #writer.SetFileName(output_path+"\\stl\\"+"%s.stl" %(i))
            stripper.Update()
            writer.SetInputData(stripper.GetOutput())
            writer.Write()  

            #save side into proper path
            #linux
            side_path=output_path+"side/"+f"{i}.txt"
            #windows
            #side_path=output_path+"\\side\\"+f"{i}.txt"  
            with open(side_path,"w") as f:
                f.write(legside)        
