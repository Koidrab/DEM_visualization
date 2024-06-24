*This project was done for and in collaboration with [Argotec](https://www.argotecgroup.com/)*
# 3D DEM Visualization for geolocalized satellite images
The proposed solution is an algorithm that is able to recreate a 3D representation of a terrain, given an orthorectified image and the Digital Elevation Model (DEM) that contains it. The general workflow idea for the algorithm was to create the 3D model directly from the DEM and than applying the image just as a texture.  It has been done in Python.

In order to do that it was necessary that both the image and the DEM were geolocalized, so the first step was to find some geolocated data to begin to work with. For this purpose I used a website that provides raw data downloads of satellite, aircraft and other remote sensing devices named EarthExplorer developed by the United States Geological Survey (EEUSGS). 

Since it is difficult and time consuming to find an image and his directly corresponding DEM, it was necessary to make an image and a DEM that contains that portion of terrain coherent one with the other in terms of resolution.

The main issue encountered was to identify and crop the portion of the DEM related to the image, so that the two could be in a 1-to-1 pixel ratio. I propose a mean to do this by knowing, of both image and DEM:

- dimensions in pixels (width x height)
- latitude and longitude of the pixel at center
- resolution in terms of meters per pixel

## Data cleaning

First thing to do was to make sure that DEM had not missing data (due to sensors errors) and in that cases (the most likely ones) to fill that lacks. 

All the DEM elements that lay below a certain threshold (0.5 experimentally seemed to be a good choice since a value less than that is not very likely to be present) are replaced with NaNs first (`np.where(…)`) and than interpolated with neighbors. Since most of the gaps were negligible, a 1D weighted linear interpolation performed on the reshaped DEM matrix was sufficient to have a decent and fulfilling 3D representation. (`fill_nan(…)`)

In this specific case, it has been also necessary to reshape the DEM because of the strange square format that EEUSGS provided. This was done simply by calculating the distance between corners coordinates, i.e. the sides of the rectangle, using Haversine formula and multiplying them by the resolution of the DEM (`desquare(…)`) 

## Resampling

As mentioned before, it was necessary to process the matrices in order to properly fit the resolution one of the other. Since it is more likely that is the DEM the one to be greater, the algorithm proposed was made with the purpose of adapting the DEM over the image and not vice versa (in future, a flexible function for resampling dynamically the one with the lowest resolution may be implemented).

This was achieved by resampling the DEM by a factor dependent on the ratio between the raw image resolution and the DEM resolution. The ratio was calculated by a function that is a sort of wrapper function of the *resize* function of the OpenCV library that however adds some additional functionalities: it takes as input the DEM matrix, the image matrix, their respectively resolutions and a resolution threshold and it calculates the resolution ratio and the new dimensions of the DEM. Then it calls the OpenCV resize which takes the DEM matrix, the new size we want it to be resized, and the resolution ratio and returns the resized matrix with the values interpolated by one of the interpolation algorithms provided by OpenCV. For this purposes, a simple linear interpolation did the trick. (`resample(...)`)

### Resampling issues

Despite Python is a very good language, it has its own problems. When the image comes with a resolution that is too high (under 2 m/px), the execution of the resampling fails because of an *Insufficient memory error.*

 To solve this issue, the idea was to add a little layer of recursion to the resampling function (with a depth of just two calls in the worst case) in order to resample not only the DEM to the resolution of the image but also the image to a smaller and new resolution: the function compares the resolution of the image with an arbitrary threshold and, if below, it recursively calls itself to resample the image to the value indicated by the threshold, then continue with the proper DEM resampling using the new calculated resolution.

At this point the resolution is the same for the DEM and the image.

## Cropping

As soon as the DEM and the image were setted up at the same resolution it was necessary to crop the DEM over the image region (`crop(...)`). 

It was done by following this steps:

1.  computation of the latitudinal and longitudinal distances (in meters) from the center of the DEM to the center of the image by using the Haversine formula
2. transformation of that distances in terms of pixels dividing them by the new resolution computed before 
3. definition of the window size to crop the DEM to.
