import numpy as np
import cv2 as cv 
import tifffile as tif
import xml.etree.ElementTree as ET
import math 
import mayavi.mlab as mlb
from tvtk.api import tvtk
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
import matplotlib.pyplot as plt

def geotiff_getCoordinates(tiff, res, x, y):
    x = x*res
    y = y*res
    r = 6371000
    
    NW_lat = tiff.geotiff_metadata.get("ModelTiepoint")[4]
    NW_lon = tiff.geotiff_metadata.get("ModelTiepoint")[3]

    lat_rad = math.radians(NW_lat)
    lon_rad = math.radians(NW_lon)

    center_lat = math.degrees(lat_rad - y/(r*2))
    center_lon = math.degrees(2/(math.cos(lat_rad))*math.sin(x/(4*r))+lon_rad)

    SE_lat = math.degrees(lat_rad - y/(r))
    SE_lon = math.degrees(2/(math.cos(lat_rad))*math.sin(x/(2*r))+lon_rad)
    
    return ((center_lat, center_lon), (NW_lat, NW_lon), (NW_lat, SE_lon), (SE_lat, NW_lon), (SE_lat, SE_lon))


def XML_getResolution(path, isDEM=False):
    tree = ET.parse(path)
    root = tree.getroot()
    
    if isDEM:    
        for el in root.iter():
            res_el = el.findall("keywords/theme/grouping/themekey")
            if len(res_el):
                res = float(res_el[1].text)   
    
    else:
         for el in root.iter():
            res_el = el.find("{http://www.isotc211.org/2005/gmd}resolution/{http://www.isotc211.org/2005/gco}Length")
            if res_el != None:
                    res = float(res_el.text)
    
    assert res != None, "Couldn't parse resolution in XML file."

    return res


def XML_getCoordinates(path):

    tree = ET.parse(path)
    root = tree.getroot()

    W = E = S = N = None

    for el in root.iter():

        W_el = el.find("{http://www.isotc211.org/2005/gmd}westBoundLongitude/{http://www.isotc211.org/2005/gco}Decimal")
        E_el = el.find("{http://www.isotc211.org/2005/gmd}eastBoundLongitude/{http://www.isotc211.org/2005/gco}Decimal")
        S_el = el.find("{http://www.isotc211.org/2005/gmd}southBoundLatitude/{http://www.isotc211.org/2005/gco}Decimal") 
        N_el = el.find("{http://www.isotc211.org/2005/gmd}northBoundLatitude/{http://www.isotc211.org/2005/gco}Decimal")

        if W_el != None:
                W = float(W_el.text)

        if E_el != None:
                E = float(E_el.text)
                
        if S_el != None:
                S = float(S_el.text)

        if N_el != None:
                N = float(N_el.text)
    
    assert W != None or E != None or S != None or N != None  , "Couldn't parse coordinates in XML file."

    C = [np.mean([N, S]), np.mean([W, E])]

    return (C, [N, W], [N, E], [S, W], [S, E])


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
        res3 = res_thresh
        img2, _ = resample(img2, np.nan, res2, res3, thresh) #se res di img troppo piccola (<res_thresh m/px), resample a 10 m/px
        res2 = res3
    
    fac = res2/res1
    u = int(img1.shape[1]/fac)
    v = int(img1.shape[0]/fac)

    return (cv.resize(img1, dsize=[u, v], fx=fac, fy=fac, interpolation=cv.INTER_LINEAR), img2)

#croppo il dem alle dimensioni dell'immagine
def crop(zos, img, demcoo, imgcoo, res):

    #calcolo distanza in x, y in metri
    EW_dst = haversine_distance(demcoo, (demcoo[0], imgcoo[1]))*1000
    NS_dst = haversine_distance(demcoo, (imgcoo[0], demcoo[1]))*1000

    #calcolo la distanza in pixel dei centri
    if EW_dst-np.floor(EW_dst) < 0.5:
        EW_dst = np.floor(EW_dst)
    else:
        EW_dst = np.ceil(EW_dst)

    if NS_dst-np.floor(NS_dst) < 0.5:
        NS_dst = np.floor(NS_dst)
    else:
        NS_dst = np.ceil(NS_dst)

    x_dst = int(EW_dst/res)
    y_dst = int(NS_dst/res)

    # definizione della finestra da croppare
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


dem_path = "C:/Users/giovi/Dropbox (Politecnico Di Torino Studenti)/Polito/Primo anno/Secondo semestre_/Image processing and computer vision/Progetto/DEM_visualization/DEM_visualization/data/n43_w111_1arc_v3.tif"
img_path = "C:/Users/giovi/Dropbox (Politecnico Di Torino Studenti)/Polito/Primo anno/Secondo semestre_/Image processing and computer vision/Progetto/DEM_visualization/DEM_visualization/data/land_textureT.tif"
xml_img_path = "C:/Users/giovi/Dropbox (Politecnico Di Torino Studenti)/Polito/Primo anno/Secondo semestre_/Image processing and computer vision/Progetto/DEM_visualization/DEM_visualization/data/m_4311037_ne_12_060_20220719.xml"
xml_dem_path = "C:/Users/giovi/Dropbox (Politecnico Di Torino Studenti)/Polito/Primo anno/Secondo semestre_/Image processing and computer vision/Progetto/DEM_visualization/DEM_visualization/data/srtm_v3_SRTM1N43W111V3.xml"

dem_res = 30 #XML_getResolution(xml_dem_path, isDEM=True) se si trovasse un XML decente
img_res = XML_getResolution(xml_img_path)
res_thresh = 5  # soglia di risoluzione massima: se sotto la soglia, le immagini verranno ricampionate al valore di soglia (espressa in metri/pixel)

img_coo = XML_getCoordinates(xml_img_path)

# le coordinate degli angoli dell'immagine potrebbero essere ricavate automaticamente attraverso la funzione coordinates(...) se il rapporto d'aspetto del DEM fosse quello corretto
DEM_corners = ([44, -111], #NW
               [44, -110], #NE
               [43, -111], #SW
               [43, -110]) #SE

dem = tif.TiffReader(dem_path)
z_raw = cv.imread(dem_path, cv.IMREAD_UNCHANGED)
img = cv.cvtColor(cv.imread(img_path, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB)

z_hole = np.where(z_raw<0.5, np.nan, z_raw) # elimino dati assenti

z_fll = fill_nan(z_hole) # interpolo dati assenti

z_dsq = desquare(z_fll, DEM_corners, dem_res) # correzione del rapporto d'aspetto del DEM a partire dalle coordinate degli angoli

#la seguente chiamata a funzione potrebbe essere eseguita prima se il rapporto d'aspetto del DEM fosse quelo corretto 
dem_coo = geotiff_getCoordinates(dem, dem_res, z_dsq.shape[1], z_dsq.shape[0]) # parsing delle coordinate di centro e angoli DEM

z_os, img = resample(z_dsq, img, dem_res, img_res, res_thresh) # ricampionamento di DEM e immagine (se res troppo alta (piccola))

if img_res < res_thresh:
    img_res = res_thresh # aggiorno risoluzione immagine se modificata

z = crop(z_os, img, dem_coo[0], img_coo[0], img_res) # crop del DEM su una finestra ampia quanto l'immagine

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
