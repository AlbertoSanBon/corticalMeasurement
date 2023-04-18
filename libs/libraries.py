#C:\Users\albsb\OneDrive\Documentos\ICAI 4 TELECO\ARTICULO CIENTIFICO\config>python -m pydoc -w libraries

import time
import os
import vtk
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib as mpl
import scipy.ndimage
from skimage import morphology
from skimage.morphology import skeletonize
from skimage import measure
from sklearn.cluster import KMeans
from tqdm import tqdm
import cv2
from skimage.util import invert
from skimage.filters import threshold_otsu
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
from math import atan2, cos, sin, sqrt, pi
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
import logging
from logging import FileHandler
from vlogging import VisualRecord


def load_scan(path, files):
    """
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
    """
    # Will load all DICOM images from a folder into a list for manipulation
    t=time.time()
    slices = [pydicom.read_file(path + '/' + s) for s in files]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    print("Elapsed time: {} sec.".format(time.time()-t))    
    return slices

def get_pixels_hu(scans):
    """
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
    """
    # Requiere instalar pylibjpeg y python-gdcm
    
    t=time.time()
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    print("Elapsed time: {} sec.".format(time.time()-t))    
    return np.array(image, dtype=np.int16)

def sample_stack(stack, rows=11, cols=10, start_with=1, show_every=7, color=False, cmap=None):
    """
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
    """
    try:
        t=time.time()
        fig,ax = plt.subplots(rows,cols,figsize=[17,17])
        count=0
        for i in tqdm(range(rows)):
            for j in range(cols):
                ind = start_with + count*show_every
                ax[i,j].set_title('slice %d' % ind)
                if color:
                    if cmap:
                        ax[i,j].imshow(stack[ind],cmap=cmap)
                    else:
                        ax[i,j].imshow(stack[ind])
                else:
                    ax[i,j].imshow(stack[ind],cmap='gray')
                ax[i,j].axis('off')
                count+=1
        plt.show()
        #plt.savefig("Sample_stack.png")
        print("Elapsed time: {} sec.".format(time.time()-t)) 
    except:
        t=time.time()
        fig,ax = plt.subplots(rows,cols,figsize=[17,17])
        count=0
        rows=5
        cols=5
        show_every=5
        for i in tqdm(range(rows)):
            for j in range(cols):
                ind = start_with + count*show_every
                ax[i,j].set_title('slice %d' % ind)
                if color:
                    if cmap:
                        ax[i,j].imshow(stack[ind],cmap=cmap)
                    else:
                        ax[i,j].imshow(stack[ind])
                else:
                    ax[i,j].imshow(stack[ind],cmap='gray')
                ax[i,j].axis('off')
                count+=1
        plt.show()
        #plt.savefig("Sample_stack.png")
        print("Elapsed time: {} sec.".format(time.time()-t)) 
def resample(image, scan, new_spacing=[1,1,1]):
    """
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
    """
    t=time.time()
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    print("Elapsed time: {} sec.".format(time.time()-t)) 
    return image, new_spacing

def find_bounding_box(img, hu=True,threshold=200, display=True, sizex=5, sizey=5, title=True, linewidth=1):
    """
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
    """
    if hu==False:
        #then RGB
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        #if HU image
        im_normalized=(img - img.min()) / (img.max() - img.min())
        im_normalized_scaled=(im_normalized*255)
        gray = np.array(im_normalized_scaled, dtype=np.uint8)
    thresh = cv2.inRange(gray, threshold, 255)
    # find contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # get rectangle
    if len(contours)==0:
        return (0,0),(0,0),0, 0
    x,y,w,h = cv2.boundingRect(contours[-1])
    middle = img[y:y+h,x:x+w]
    
    if display:
        fig, ax = plt.subplots(1, 2, figsize=[sizex, sizey])
        if title: fig.suptitle("Bounding box detection", fontsize=16)
        ax[0].imshow(img, cmap='gray')
        rect = patches.Rectangle((x, y), w, h, linewidth=linewidth, edgecolor='g', facecolor='none')
        ax[0].add_patch(rect)
        ax[1].imshow(middle, cmap='gray')
        plt.show()
    
    return (x,y),(x+w,y+h),w, h

def find_bounding_box_sample_stack(img, hu=True, show_box=True, threshold=200, rows=10, cols=10, start_with=1, show_every=3, areamin=None):
    """
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
    """
    t=time.time()
    fig,ax = plt.subplots(rows,cols,figsize=[17,17])
    count=0
    for i in range(rows):
        for j in range(cols):
            ind = start_with + count*show_every
            ax[i,j].set_title('slice %d' % ind)
            (x,y),(X,Y),W,H =find_bounding_box(img[ind], hu=hu,threshold=threshold, display=False)
            area_bbox = abs(x-X)*abs(y-Y)
            thres=threshold
            if areamin!=None:
                while area_bbox<areamin:
                    thres=thres-1
                    (x,y),(X,Y),W,H =find_bounding_box(img[ind], hu=hu,threshold=thres, display=False)
                    area_bbox = abs(x-X)*abs(y-Y)
            if show_box:
                ax[i,j].imshow(img[ind], cmap='gray')
                rect = patches.Rectangle((x, y), W, H, linewidth=1, edgecolor='g', facecolor='none')
                ax[i,j].add_patch(rect)
            else:
                middle = img[ind][y:Y,x:X]
                ax[i,j].imshow(middle, cmap='gray')
            ax[i,j].axis('off')
            count+=1
    plt.show()
    print("Elapsed time: {} sec.".format(time.time()-t))  
    
def make_bonesmask(img, kernel_preErosion, kernel_firstDilation, kernel_firstErosion, hu=True, threshold=200, display=False, extract=[], size=60, areamin=None):
    """
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
    """
    img_org=img.copy()
    row_size= img.shape[0]
    col_size = img.shape[1]  
    
    # Find bounding box
    (x,y),(X,Y),W,H =find_bounding_box(img,hu,threshold, display=False)
    area_bbox = abs(x-X)*abs(y-Y)
    thres=threshold
    if areamin!=None:
        while area_bbox<areamin and thres>1:
            thres=thres-1
            (x,y),(X,Y),W,H =find_bounding_box(img, hu,thres, display=False)
            area_bbox = abs(x-X)*abs(y-Y)
    
    if thres==1:
        print("Bounding box not found")
        mask = np.ndarray([row_size,col_size],dtype=np.int8)
        mask[:] = 0
        return mask, mask*img, mask*img_org, mask
    
    #Standardize the pixel values
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std

    # Find the average pixel value near in the bounding box
    # to renormalize washed out images
    middle = img[y:Y,x:X] 
    #print(middle.shape)

    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    img[img==max]=mean
    img[img==min]=mean
    
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    if middle.shape[0]==1 or middle.shape[1]==1:
        mask = np.ndarray([row_size,col_size],dtype=np.int8)
        mask[:] = 0
        return mask, mask*img, mask*img_org, mask
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,0.0,1.0)  # threshold the image
                                                  # Sólo tiene valores 0 y 1
                                                  # Invierto los valores para que el hueso sean los 1s

    # THRESHOLD IMAGE PROCESSING
    # ==========================
    
    # First a light erosion to remove artifacts
    # Then a powerful dilation to try to fill the masks. But if it is
    # too powerful, can bind bones
    # Finally, erode to return the mask to its original size.
    # These images only have values 0 and 1.
    initial_erosion = morphology.erosion(thresh_img,np.ones(kernel_preErosion)) # Primera erosión para eliminar artefactos
    dilation = morphology.dilation(initial_erosion,np.ones(kernel_firstDilation)) # Segunda dilatación para eliminar huecos
    erosion = morphology.erosion(dilation,np.ones(kernel_firstErosion))  # Tercera erosión para recuperar el tamaño normal

    # LABEL GENERATION
    # ================
    
    # The above processing works with most slices, but other may 
    # different scenarios occur:
    #    - That many labels are generated because there are atifacts that have not been eliminated
    #    - Gaps left inside the masks because the dilation wasn't powerful enough 
    labels = measure.label(erosion,background=0)  # Different labels are displayed in different colors
                                                  # Be careful, we say that background is 0 because here what 
                                                  # the bone is white
    
    # LABEL PROCESSING
    # ================
    
    # The first step is to keep the two most important labels that will correspond to the two largest areas, the two bones
    # The second step is to dilate the bones separately to plug their holes and re-erode them to return them to its original size
    # The third step is to generate the joint mask or a bone mask to be able to separate them
    
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)

    # Selection of the labels that correspond to the 2 largest areas
    if len(regions)>=2:
        areas=[i.area for i in regions]
        areas.sort(reverse=True)
        larger_areas=areas[:2]
        good_labels=[i.label for i in regions if i.area in larger_areas]
    elif len(regions)==1:
        good_labels=[regions[0].label]
    else:
        good_labels=[]

    # Generation of the new LABELS image keeping only the 2 largest areas and renaming them to label 1 and label 2
    new_label = np.ndarray([row_size,col_size],dtype=np.int8)
    new_label[:] = 0
    for l in good_labels:
        tmp=np.where(labels==l,l,0)
        dilation2 = morphology.dilation(tmp,np.ones([size,size]))  
        erosion2 = morphology.erosion(dilation2,np.ones([size,size]))
        new_label = new_label + erosion2

    transdict=dict(zip(good_labels,range(1,len(good_labels)+1)))
    new_label_norm = np.zeros_like(new_label)
    for key,val in transdict.items():
        new_label_norm[new_label==key] = val

    # Generation of the new definitive MASK
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0
    
    if extract==[]:
        for l in good_labels:
            mask = mask + np.where(new_label==l,1,0)
    else:
        for l in extract:
            mask = mask + np.where(new_label_norm==l,1,0)
    
    if (display):
        fig, ax = plt.subplots(2, 4, figsize=[15, 6])
        fig.suptitle("Segmentation", fontsize=16)
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[0, 2].set_title("First Erosion and Dilation")
        ax[0, 2].imshow(erosion, cmap='gray')
        ax[0, 2].axis('off')
        ax[0, 3].set_title("After Main Erosion and Dilation")
        ax[0, 3].imshow(new_label, cmap='gray')
        ax[0, 3].axis('off')
        ax[1, 0].set_title("Color Labels Before processing")
        ax[1, 0].imshow(labels)
        for prop in regions:
            B = prop.bbox
            rect = patches.Rectangle((B[1],B[0]), B[3]-B[1],B[2]-B[0], linewidth=1, edgecolor='g', facecolor='none')
            ax[1, 0].add_patch(rect)
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels After processing")
        ax[1, 1].imshow(new_label_norm)
        ax[1, 1].axis('off')
        ax[1, 2].set_title("Final Mask")
        ax[1, 2].imshow(mask, cmap='gray')
        ax[1, 2].axis('off')
        ax[1, 3].set_title("Apply Mask on Original")
        ax[1, 3].imshow(mask*img, cmap='gray')
        ax[1, 3].axis('off')
        
        plt.show()
    return mask, mask*img, mask*img_org, new_label_norm

def CreateTissueFromArray(imageData, ThrIn, ThrOut, color = "skeleton", isoValue = 127.5):
    """
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
    """
    tissueMap=CreateTissueMap()
    colorLut=CreateLut()
    selectTissue = vtk.vtkImageThreshold()
    selectTissue.ThresholdBetween(ThrIn,ThrOut)
    selectTissue.ReplaceInOn()    # Determines whether to replace the pixel in range with InValue
    selectTissue.SetInValue(255)  # set all values between ThrIn,ThrOut to 255
    selectTissue.ReplaceOutOn()
    selectTissue.SetOutValue(0)   # set all values otside ThrIn,ThrOut to 0
    selectTissue.SetInputData(imageData)
    selectTissue.Update()
    
    gaussianRadius = 5
    gaussianStandardDeviation = 1.0
    gaussian = vtk.vtkImageGaussianSmooth()
    gaussian.SetStandardDeviations(gaussianStandardDeviation, gaussianStandardDeviation, gaussianStandardDeviation)
    gaussian.SetRadiusFactors(gaussianRadius, gaussianRadius, gaussianRadius)
    gaussian.SetInputData(selectTissue.GetOutput())
    gaussian.Update()

    #isoValue = 127.5
    surface = vtk.vtkMarchingCubes()
    surface.SetInputData(gaussian.GetOutput())
    surface.ComputeScalarsOff()
    surface.ComputeGradientsOff()
    surface.ComputeNormalsOff()
    surface.SetValue(0, isoValue) # define surface 0 as a isosurface at HU level (compressed from 0 to 255) isoValue

    smoothingIterations = 5
    passBand = 0.001
    featureAngle = 60.0
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(surface.GetOutputPort())
    smoother.SetNumberOfIterations(smoothingIterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(featureAngle)
    smoother.SetPassBand(passBand)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.SetFeatureAngle(featureAngle)

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(normals.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor( colorLut.GetTableValue(tissueMap[color])[:3])
    actor.GetProperty().SetSpecular(.5)
    actor.GetProperty().SetSpecularPower(10)
    
    return actor, stripper

def CreateTissueMap():
    """
    Introduction
    ------------
    Create a Tissue map
    
    Returns
    -------
    tissueMap : dict
        Tissue map created.
    """
    tissueMap = dict()
    tissueMap["blood"] = 1
    tissueMap["brain"] = 2
    tissueMap["duodenum"] = 3
    tissueMap["eyeRetina"] = 4
    tissueMap["eyeWhite"] = 5
    tissueMap["heart"] = 6
    tissueMap["ileum"] = 7
    tissueMap["kidney"] = 8
    tissueMap["intestine"] = 9
    tissueMap["liver"] = 10
    tissueMap["lung"] = 11
    tissueMap["nerve"] = 12
    tissueMap["skeleton"] = 13
    tissueMap["spleen"] = 14
    tissueMap["stomach"] = 15

    return tissueMap

def CreateLut():
    """
    Introduction
    ------------
    Assign a color to a tissue

    Returns
    -------
    colorLut : vtk
        Color asigned to a tissue.
    """
    colors = vtk.vtkNamedColors()

    colorLut = vtk.vtkLookupTable()
    colorLut.SetNumberOfColors(17)
    colorLut.SetTableRange(0, 16)
    colorLut.Build()

    colorLut.SetTableValue(0, 0, 0, 0, 0)
    colorLut.SetTableValue(1, colors.GetColor4d("salmon"))  # blood
    colorLut.SetTableValue(2, colors.GetColor4d("beige"))  # brain
    colorLut.SetTableValue(3, colors.GetColor4d("orange"))  # duodenum
    colorLut.SetTableValue(4, colors.GetColor4d("misty_rose"))  # eye_retina
    colorLut.SetTableValue(5, colors.GetColor4d("white"))  # eye_white
    colorLut.SetTableValue(6, colors.GetColor4d("tomato"))  # heart
    colorLut.SetTableValue(7, colors.GetColor4d("raspberry"))  # ileum
    colorLut.SetTableValue(8, colors.GetColor4d("banana"))  # kidney
    colorLut.SetTableValue(9, colors.GetColor4d("peru"))  # l_intestine
    colorLut.SetTableValue(10, colors.GetColor4d("pink"))  # liver
    colorLut.SetTableValue(11, colors.GetColor4d("powder_blue"))  # lung
    colorLut.SetTableValue(12, colors.GetColor4d("carrot"))  # nerve
    colorLut.SetTableValue(13, colors.GetColor4d("wheat"))  # skeleton
    colorLut.SetTableValue(14, colors.GetColor4d("violet"))  # spleen
    colorLut.SetTableValue(15, colors.GetColor4d("plum"))  # stomach

    return colorLut

def compute_thickness_image(image, contourid=0, grow=False):
    """
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
    """
    thresh = threshold_otsu(image.T)
    binary = image.T > thresh
    skeleton = skeletonize(binary)
    im=binary.astype(np.uint8)*255
    zeros=(binary.astype(np.uint8)*0).copy()
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours)!=0 and contourid==-1:
        areas=[cv2.contourArea(c) for c in contours]
        contourid=areas.index(max(areas))
    if len(contours)!=0 and contourid+1<=len(contours):
        external=cv2.drawContours(zeros, contours, contourid, 255,1)      
        #cv2.drawContours(zeros, contours, contourid,255,1) # drawContours modifica zeros
        #external=zeros
    else:
        external=zeros
    skeleton=skeleton.astype(np.uint8)*255
    skeleton_inv=invert(skeleton)
    sdf=cv2.distanceTransform(skeleton_inv, cv2.DIST_L2, 3)
    mask=external>0
    width_matrix=sdf*mask*2   
    if grow:
        width_matrix = morphology.dilation(width_matrix,np.ones([7,7]))
    return contourid, contours, external.T ,width_matrix.T 

def convertTo1D(array_coordinates,array_thickness,countour_index=0,reference_x = 100, verbose=True ):
    """
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
    """
    noreferences=0
    array_thickness_1d={}
    contourid = countour_index
    for i in range(len(array_coordinates)):
        contour = array_coordinates[i]
        thick   = array_thickness[i]
        if type(countour_index)==list:
            contourid = countour_index[i]     
        thick1d = []
        if contourid!= -1 and len(contour)>=contourid+1:
            c=contour[contourid].squeeze()
            if len(c.shape)==2:
                # Check if contour has 2D
                items,dim=c.shape
                if items>1 and dim==2:
                    # Check if contour has enough ponints
                    X,Y=[],[]
                    for x,y in c: 
                        X.append(x)
                        Y.append(y)
                    try:
                        X.index(reference_x)
                    except:
                        if verbose: print("no reference_x in contour ",i)
                        noreferences=noreferences+1
                        array_thickness_1d[i]=None
                        continue
                    # find ocurrences of reference_x
                    indices = [pos for pos, x in enumerate(X) if x == reference_x]
                    if len(indices)==1:
                        # only one ocurrence of reference_x
                        start=indices[0]
                    elif len(indices)>1:
                        # many points with same reference_x, take the one with lowest Y
                        YY=[Y[k] for k in indices]
                        minYindex=YY.index(min(YY))  
                        start=indices[minYindex]
                    else:
                        print("should not enter here ever ",i)
                    for j in range(len(X)):
                        coord_index=(start+j)%len(X)
                        thick1d.append(thick[X[coord_index],Y[coord_index]])

                    array_thickness_1d[i]=thick1d
                else:
                    if verbose: print("wrong contour in ",i," strange shape. Should be (number_of_points,2) and is: ",c.shape)
                    array_thickness_1d[i]=None
            else:
                if verbose: print("wrong contour in ",i," strange shape. Should be (number_of_points,2) and is: ",c.shape)
                array_thickness_1d[i]=None
        else:
            if verbose: print("no contour ",contourid," in slice ",i)
            array_thickness_1d[i]=None
    
    return array_thickness_1d,noreferences

def show_cuts_position(cortesG, num_views, G, poly_data, bounds, spacing):
    """
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

    """
    # And we paint the plane selecting SetOrigin which is a vertex and SetPoint1 and SetPoint2 which are the two vertices closest to the Origin.    
    # plane = vtk.vtkPlaneSource()
    # plane.SetNormal(0,0,1)
    # plane.SetOrigin(G[0]-(bounds[1] - bounds[0])/2, G[1]-(bounds[3] - bounds[2])/2, G[2])
    # plane.SetPoint1(G[0]+(bounds[1] - bounds[0])/2 , G[1]-(bounds[3] - bounds[2])/2, G[2])
    # plane.SetPoint2(G[0]-(bounds[1] - bounds[0])/2 , G[1]+(bounds[3] - bounds[2])/2, G[2])
    # plane.Update()
    
    appendFilter = vtk.vtkAppendPolyData()
    # appendFilter.AddInputData(plane.GetOutput())
    # appendFilter.Update()
    
    for i in range(num_views):
        # Cut in the slices of the cuts vector
        plane1 = vtk.vtkPlaneSource()
        plane1.SetNormal(0,0,1)
        plane1.SetOrigin(G[0]-(bounds[1] - bounds[0])/2, G[1]-(bounds[3] - bounds[2])/2, cortesG[i-1]*spacing[2])
        plane1.SetPoint1(G[0]+(bounds[1] - bounds[0])/2 , G[1]-(bounds[3] - bounds[2])/2, cortesG[i-1]*spacing[2])
        plane1.SetPoint2(G[0]-(bounds[1] - bounds[0])/2 , G[1]+(bounds[3] - bounds[2])/2, cortesG[i-1]*spacing[2])
        plane1.Update()
        
        # Combine the meshes        
        appendFilter.AddInputData(plane1.GetOutput())
        appendFilter.Update()
        
    appendFilter.AddInputData(poly_data)
    appendFilter.Update()
    
    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(appendFilter.GetOutputPort())
    mapper.SetColorModeToDirectScalars()
    
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
    colors = vtk.vtkNamedColors()
    renderer.SetBackground(colors.GetColor3d('DarkSlateGray'))
    
    # Render and interact
    renderWindow.Render()
    filename = 'cuts_p.png'
    WriteImage(filename, renderWindow, rgba=False)
    #renderWindowInteractor.Start()

def show_cuts_position_restored(cortesG, num_views, G, np_scalars, bounds, spacing):
    """
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

    """
    
    file_used = np_scalars
    imgs_to_process = file_used.astype(np.float64) 
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
    
    # And we paint the plane selecting SetOrigin which is a vertex and SetPoint1 and SetPoint2 which are the two vertices closest to the Origin.    
    plane = vtk.vtkPlaneSource()
    plane.SetNormal(0,0,1)
    plane.SetOrigin(G[0]-(bounds[1] - bounds[0])/2, G[1]-(bounds[3] - bounds[2])/2, G[2])
    plane.SetPoint1(G[0]+(bounds[1] - bounds[0])/2 , G[1]-(bounds[3] - bounds[2])/2, G[2])
    plane.SetPoint2(G[0]-(bounds[1] - bounds[0])/2 , G[1]+(bounds[3] - bounds[2])/2, G[2])
    plane.Update()
    
    appendFilter = vtk.vtkAppendPolyData()
    appendFilter.AddInputData(plane.GetOutput())
    appendFilter.Update()
    
    for i in range(num_views):
        # Cut in the slices of the cuts vector
        plane1 = vtk.vtkPlaneSource()
        plane1.SetNormal(0,0,1)
        plane1.SetOrigin(G[0]-(bounds[1] - bounds[0])/2, G[1]-(bounds[3] - bounds[2])/2, cortesG[i-1]*spacing[2])
        plane1.SetPoint1(G[0]+(bounds[1] - bounds[0])/2 , G[1]-(bounds[3] - bounds[2])/2, cortesG[i-1]*spacing[2])
        plane1.SetPoint2(G[0]-(bounds[1] - bounds[0])/2 , G[1]+(bounds[3] - bounds[2])/2, cortesG[i-1]*spacing[2])
        plane1.Update()
        
        # Combine the meshes        
        appendFilter.AddInputData(plane1.GetOutput())
        appendFilter.Update()
        
    appendFilter.AddInputData(imdata)
    appendFilter.Update()
    
    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(appendFilter.GetOutputPort())
    mapper.SetColorModeToDirectScalars()
    
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
    colors = vtk.vtkNamedColors()
    renderer.SetBackground(colors.GetColor3d('DarkSlateGray'))
    
    # Render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()


def show_cuts(array_thickness, cortesG, num_views, spacing, origin):
    """
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
    
    """
    #Convertir el array de espesores thickness en un vtkImageData
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
    
    # Obtain dimensions & origin of vtkImagedata
    x,y,num=img_vtk.GetDimensions()
    print(img_vtk.GetDimensions())
    ox,oy,oz=img_vtk.GetOrigin()
    print(img_vtk.GetOrigin())
    
    rows=num_views//4+1
    fig=plt.figure(figsize=(18,rows*4))
    for i in range (1,num_views+1):
        try:
            plt.subplot(rows,4,i)        
            plane = vtk.vtkPlane()
            plane.SetOrigin(ox,oy,cortesG[i-1]*spacing[2])
            plane.SetNormal(0,0,1)
            
            cutter = vtk.vtkCutter()
            cutter.SetCutFunction(plane)
            cutter.SetInputData(img_vtk)
            cutter.Update()
            
            poly = cutter.GetOutput()
            array1=vtk_to_numpy(poly.GetPointData().GetScalars()).reshape(y,x,-1)
            print(array1.shape)
            
            #plt.subplot(num_views//4+1, num_views//3, i)
            plt.imshow(array1)
            plt.title("slice "+ str(cortesG[i-1]))
    
        except: print("Slice not found")
        
    plt.show()
    #plt.savefig("Cuts.png")

def rotation_matrix_from_vectors(vec1, vec2):
    """
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

    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

def orientation_slice(image, represent=False):
    """
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

    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR )
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #Contornos es una lista de Python de todos los contornos de la imagen. Cada contorno individual es una matriz Numpy de coordenadas (x, y) de puntos límite del objeto.
    
    areas=[]
    for i, c in enumerate(contours):
        areas.append(cv2.contourArea(c))
    index=areas.index(max(areas))
        
    # Draw each contour only for visualisation purposes
    cv2.drawContours(image_rgb, contours, index, (0,0,255), 2)
    
    # Find the orientation of each shape
    vector=getOrientationMainVector(contours[index], image_rgb, arrowsize=5)

    if represent == True:
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(image_rgb)
        plt.show()
        plt.savefig("Main_vector.png")
    
    return vector

def getOrientationMainVector(pts, img, arrowsize=3, textsize=0.3, textcoord=None):
    """
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

    """
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
        
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]
    
    # Draw the principal components
    #cv.circle(img, cntr, 3, (255, 255, 0), 1)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    drawAxis(img, cntr, p1, (255, 255, 0), arrowsize)
    
    # Label with the rotation angle
    label = "Vector: ({:.4f},{:.4f})".format(eigenvectors[0,0],eigenvectors[0,1]) 
    #textbox = cv.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    if textcoord==None:
        cv2.putText(img, label, (cntr[0]-30, cntr[1]-100), cv2.FONT_HERSHEY_SIMPLEX, textsize, (255, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, label, (textcoord[0], textcoord[1]), cv2.FONT_HERSHEY_SIMPLEX, textsize, (255, 255, 0), 1, cv2.LINE_AA)
        
    return eigenvectors[0,:]

def drawAxis(img, p_, q_, color, scale):
    """
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

    """
    p = list(p_)
    q = list(q_)
    
    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)
    
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)
    
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)
    
def getClosestPointInRadius(kDTree, thickness, thickness_values, ccenter, radius=0.5):
    """
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

    """
    point=ccenter
    closestPointDist=vtk.mutable(0)
    results=vtk.vtkIdList()
    kDTree.FindPointsWithinRadius(radius, point, results)
    
    if results.GetNumberOfIds()==0:
        # There is no points at this distance
        return None, None, None
    else:
        # Compute all distances and all thicknesses
        ids=[]
        distances=[]
        thicknesses=[]
    
        for i in range(results.GetNumberOfIds()):
            idd=results.GetId(i)
            ids.append(idd)
            coordinates=thickness.GetPoint(idd)
            distance=np.linalg.norm(np.array(coordinates)-np.array(point))      
            distances.append(distance)
            thick=thickness_values.GetComponent(idd, 0)
            thicknesses.append(thick)
            #print("Id: %d Distance: %f Thickness: %f" %(idd, distance, thick))
        
        nonzeros=np.nonzero(thicknesses)
        distances=np.array(distances)
        if len(nonzeros[0])==0:
            # if all thicknesses are 0, return the closest point
            index=np.argmin(distances)
        else:
            # if there are thicknesses different from 0, return the closest between them
            index=np.argmin(distances[nonzeros[0]])
            index=nonzeros[0][index]

        return ids[index], distances[index], thicknesses[index]
    
def color3DModelWithThickness(array_thickness, spacing_n, origin, imgstenc):
    """
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

    """
    
    # Convert the thickness array to a vtkImageData
    array=np.array(array_thickness)
    array_tmp=array.transpose(1, 2, 0)
    
    # Convert numpy array to VTK array (vtkFloatArray)
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=array_tmp.transpose(2, 1, 0).ravel(),  # ndarray contains the fitting result from the points. It is a 3D array
        deep=True,
        array_type=vtk.VTK_FLOAT)
    
    # Convert the VTK array to vtkImageData
    img_vtk = vtk.vtkImageData()
    img_vtk.SetDimensions(array_tmp.shape)
    img_vtk.SetSpacing(spacing_n)
    img_vtk.SetOrigin(origin)
    img_vtk.GetPointData().SetScalars(vtk_data_array)

    # Create the surface
    surface = vtk.vtkMarchingCubes()
    surface.SetInputData(imgstenc.GetOutput())
    surface.ComputeNormalsOn()
    surface.SetValue(0, 127.5)
    
    surface.Update()
    surface.GetOutput().GetNumberOfCells()
    
    # Assign a color according to the closest voxel
    thickness=img_vtk # Coordinates of closest point in thickness array
    points=vtk.vtkPoints()
        
        # For each ccenter (cell center) find the closest thickness(vtkImageData) value to those coordinates
    for i in tqdm(range(thickness.GetNumberOfPoints())):
        points.InsertNextPoint(thickness.GetPoint(i))
       
    points.GetNumberOfPoints()
    
    kDTree=vtk.vtkKdTree()
    kDTree.BuildLocatorFromPoints(points)
    
    # Thinkness value of closest point in thickness array
    thickness_values=thickness.GetPointData().GetScalars()
    
    # We are going to assign the maximum value of the "safe" area of the tibia because 
    # there are areas of the instep where the algorithm measures more
    minimo=array[200:500,:,:].min()
    maximo=array[200:500,:,:].max()
    
    norm = mpl.colors.Normalize(vmin=minimo, vmax=maximo)
    cmap = cm.magma
    
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    color = vtk.vtkUnsignedCharArray()
    color.SetName("Colors")
    color.SetNumberOfComponents(3)
    color.SetNumberOfTuples( surface.GetOutput().GetNumberOfCells() )
    
    cellcenter=vtk.vtkCellCenters()
    cellcenter.SetInputData(surface.GetOutput())
    cellcenter.VertexCellsOn()
    cellcenter.Update()
    
    assert cellcenter.GetOutput().GetNumberOfPoints()==surface.GetOutput().GetNumberOfCells(), "dimensions do not match"

    radius=0.35 # below this radius there are points that do not have any nearby voxels
                # is chosen by trial and error starting at 0.25 and going up from 0.05 to 0.05
                
    for i in tqdm(range(surface.GetOutput().GetNumberOfCells())):
        ccenter=[0,0,0]
        cellcenter.GetOutput().GetPoint(i, ccenter)
        # starting from the center of the triangle cell, I have to find the thickness point
        # closest to that value
        _, dist, thick=getClosestPointInRadius(kDTree, thickness, thickness_values, ccenter, radius=radius)
        if thick==None:
            print("Radius too small")
            break
        color_tup=(np.array(m.to_rgba(thick)[0:3])*255).astype(int)
        color.SetTuple(i, color_tup)
        
    assert color.GetNumberOfTuples()==surface.GetOutput().GetNumberOfCells(), "dimensions do not match"
    
    surface.Update()
    surface.GetOutput().GetCellData().SetScalars(color)
    
    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(surface.GetOutputPort())
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarModeToUseCellData()
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1) 
    
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    
    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()
    camera = renderer.GetActiveCamera()
    camera.Elevation(45)
    camera.Azimuth(90)
    camera.Roll(-45)
    renderer.SetActiveCamera(camera)
    # Add the actors to the scene
    
    # Render and interact
    renderWindow.SetSize(480, 480)
    renderWindow.Render()
    renderWindowInteractor.Start()
    
def getArea (side):
    """
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

    """
    
    return np.linalg.norm(side[0,:]-side[1,:])*np.linalg.norm(side[1,:]-side[2,:])

def WriteImage(fileName, renWin, rgba=True):
    """
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

    """
    
    if fileName:
        # Select the writer to use.
        path, ext = os.path.splitext(fileName)
        ext = ext.lower()
        if not ext:
            ext = '.png'
            fileName = fileName + ext
        else:
            writer = vtkPNGWriter()

        windowto_image_filter = vtkWindowToImageFilter()
        windowto_image_filter.SetInput(renWin)
        windowto_image_filter.SetScale(1)  # image quality
        if rgba:
            windowto_image_filter.SetInputBufferTypeToRGBA()
        else:
            windowto_image_filter.SetInputBufferTypeToRGB()
            # Read from the front buffer.
            windowto_image_filter.ReadFrontBufferOff()
            #windowto_image_filter.Update()

        writer.SetFileName(fileName)
        writer.SetInputConnection(windowto_image_filter.GetOutputPort())
        writer.Write()
    else:
        raise RuntimeError('Need a filename.')
        
def sort_contours2(cnts, method="left-to-right"):
    """
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

    """
	# initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)