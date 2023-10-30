import numpy as np
import cv2 as cv 
import tifffile as tif
import math 
import mayavi.mlab as mlb
from tvtk.api import tvtk
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
import matplotlib.pyplot as plt

def coordinates(tiff):
    return (tiff.geotiff_metadata.get("ModelTiepoint")[4], tiff.geotiff_metadata.get("ModelTiepoint")[3])


def haversine_distance(coo1, coo2):
    R = 6371.0
    
    lat1, lon1, lat2, lon2 = map(math.radians, [coo1[0], coo1[1], coo2[0], coo2[1]])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

#interpolo per reimpire i dati mancanti
def fill_nan(data):
    dim = data.shape
    data = np.reshape(data, (data.shape[0]*data.shape[1]))
    first = True

    for i in range(data.shape[0]):
        if not np.isnan(data[i]):
            if first:
                a = data[i]
                i_a = i
            else:
                b = data[i]
                arr = data[i_a+1:i]
                for j in range(arr.shape[0]):
                    dst = j/arr.shape[0]
                    arr[j] = a*(1-dst)+b*(dst)
                data[i_a+1:i] = arr
                first = True
        else:
            first = False
            continue
        
    data = np.reshape(data, dim)

    return data


def resample(img1, img2, res1, res2, thresh): 

    if(res2<thresh):
        res3 = thresh
        img2, _ = resample(img2, np.nan, res2, res3, thresh) #se res di img troppo piccola (<thresh m/px), resample a 10 m/px
        res2 = res3
    
    fac = res2/res1
    u = int(img1.shape[1]/fac)
    v = int(img1.shape[0]/fac)

    return (cv.resize(img1, dsize=[u, v], fx=fac, fy=fac, interpolation=cv.INTER_LINEAR), img2)

#croppo il dem alle dimensioni dell'immagine
def crop(zos, img, demcoo, imgcoo, imgres):

    #calcolo distanza in x, y in metri

    EW_dst = haversine_distance(dem_coo, (demcoo[0], imgcoo[1]))*1000
    NS_dst = haversine_distance(dem_coo, (imgcoo[0], demcoo[1]))*1000

    #calcolo la distanza in pixel dei centri
    if EW_dst-np.floor(EW_dst) < 0.5:
        EW_dst = np.floor(EW_dst)
    else:
        EW_dst = np.ceil(EW_dst)

    if NS_dst-np.floor(NS_dst) < 0.5:
        NS_dst = np.floor(NS_dst)
    else:
        NS_dst = np.ceil(NS_dst)

    
    x_dst = int(EW_dst/imgres)
    y_dst = int(NS_dst/imgres)

    u0 = zos.shape[1]//2 + x_dst - (img.shape[1]//2)
    u1 = zos.shape[1]//2 + x_dst + (img.shape[1]//2)

    v0 = zos.shape[0]//2 + y_dst - (img.shape[0]//2)
    v1 = zos.shape[0]//2 + y_dst + (img.shape[0]//2)
    
    dem = zos[v0:v1, u0:u1]
    
    return dem


def desquare(zhole, DEMcorners, demres):
    u = int(haversine_distance(DEMcorners[0], DEMcorners[1])*1000/demres)
    v = int(haversine_distance(DEMcorners[0], DEMcorners[2])*1000/demres)

    return cv.resize(zhole, [u, v])


# dem_path = "data/n32_e103_1arc_v3.tif"
# img_path = "data/land_texture.jpg"

# dem_res = 30
# img_res = 10
# thresh = 5

# dem_coo = (32.5, 103.5)
# img_coo = (32.340540, 103.539226)

# DEM_corners = ([33, 103], #NW
#                [33, 104], #NE
#                [32, 103], #SW
#                [32, 104]) #SE

dem_path = "data/n43_w111_1arc_v3.tif"
img_path = "data/land_textureT.tif"

dem_res = 30
img_res = 0.6
thresh = 5

dem_coo = (43.5 , -110.5)
img_coo = (43.4690027 , -110.4059861)

DEM_corners = ([44, -111], #NW
               [44, -110], #NE
               [43, -111], #SW
               [43, -110]) #SE

z_raw = cv.imread(dem_path, cv.IMREAD_UNCHANGED)
img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

z_hole = np.where(z_raw<0.5, np.nan, z_raw)

z_dsq = desquare(z_hole, DEM_corners, dem_res)

z_os, img = resample(z_dsq, img, dem_res, img_res, thresh)

if img_res < thresh:
    img_res = thresh

z_crp = crop(z_os, img, dem_coo, img_coo, img_res)

z = fill_nan(z_crp)

x = img.shape[1]
y = img.shape[0]

# PLOTTING
grid = vtk.vtkImageData()
grid.SetDimensions(y, x, 1)

tex = np.reshape(img, (x*y, 3), order="F")

vtkarr = numpy_to_vtk(tex)
vtkarr.SetName('Image')

grid.GetPointData().AddArray(vtkarr)
grid.GetPointData().SetActiveScalars('Image')

vtex = vtk.vtkTexture()
vtex.SetInputDataObject(grid)
vtex.Update()

surf = mlb.surf(x, y, z, warp_scale=1/img_res)

surf.actor.mapper.interpolate_scalars_before_mapping = False
surf.actor.actor.mapper.scalar_visibility = 0
surf.actor.enable_texture = True
surf.actor.tcoord_generator_mode = 'plane'    
surf.actor.actor.texture = vtex

mlb.show()
