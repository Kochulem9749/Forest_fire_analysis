library(glcm)
library(raster)
Pre_NBR_B8a_B11=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Pre_burn_indices/Pre_NBR_B8a_B11.tif')
Pre_NBR_B8a_B12=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Pre_burn_indices/Pre_NBR_B8a_B12.tif')
Pre_NDVI_B8_B4=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Pre_burn_indices/Pre_NDVI_B8_B4.tif')
Pre_NDVI_B8a_B5=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Pre_burn_indices/Pre_NDVI_B8a_B5.tif')
Pre_NDVI_B8a_B6=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Pre_burn_indices/Pre_NDVI_B8a_B6.tif')
Pre_NDVI_B8a_B7=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Pre_burn_indices/Pre_NDVI_B8a_B7.tif')


Post_NBR_B8a_B11=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Post_burn_indices/Post_NBR_B8a_B11.tif')
Post_NBR_B8a_B12=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Post_burn_indices/Post_NBR_B8a_B12.tif')
Post_NDVI_B8_B4=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Post_burn_indices/Post_NDVI_B8_B4.tif')
Post_NDVI_B8a_B5=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Post_burn_indices/Post_NDVI_B8a_B5.tif')
Post_NDVI_B8a_B6=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Post_burn_indices/Post_NDVI_B8a_B6.tif')
Post_NDVI_B8a_B7=raster('E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Post_burn_indices/Post_NDVI_B8a_B7.tif')


##Calculating GLCM textures in one direction
rglcm_Pre_NBR_B8a_B11 <- glcm(Pre_NBR_B8a_B11, 
              window = c(9,9), 
              shift = c(1,1), 
              statistics = c("mean", "variance", "homogeneity", "contrast", 
                             "dissimilarity", "entropy", "second_moment")
)

plot(rglcm_Pre_NBR_B8a_B11)


##Calculation rotation-invariant texture features
rglcm_Pre_NBR_B8a_B12 <- glcm(Pre_NBR_B8a_B12, 
               window = c(9,9), 
               shift=list(c(0,1), c(1,1), c(1,0), c(1,-1)), 
               statistics = c("mean", "variance", "homogeneity", "contrast", 
                              "dissimilarity", "entropy", "second_moment")
)

plot(rglcm_Pre_NBR_B8a_B12)


rglcm_Post_NBR_B8a_B12 <- glcm(Post_NBR_B8a_B12, 
                               window = c(9,9), 
                               shift=list(c(0,1), c(1,1), c(1,0), c(1,-1)), 
                               statistics = c("mean", "variance", "homogeneity", "contrast", 
                                              "dissimilarity", "entropy", "second_moment")
)
##Export the GLCM output to raster
##writeRaster(rglcm_Pre_NBR_B8a_B12,'NBR_B8a_B12.tif')