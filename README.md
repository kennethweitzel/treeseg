# Treeseg
A simple method for binary segmentation of trees in aerial images. Takes an OpenAerialMap image and outputs a geojson of vector geometries referencing visible trees in the image. 

## Installation
A conda installation is recommended, but any python environment should work. The install was tested under Ubuntu 20.04 and Python 3.13. 
Create a conda environment and install the requirements as folllows:
```bash 
conda create -n treeseg python=3.13
conda activate treeseg 
pip install rasterio torchgeo
```

## Run
Download suitable OAM image and execute the segmentation following this example: 
```bash
python3 -m treeseg d8dc67a5-c359-4d79-839a-3c78699ddfe6.tif --output tree_mask.geojson
```
Default execution device is GPU. If not available add argument `--device cpu`    
The resulting geojson will contain the segmented trees as polygons in the same CRS as the input TIFF. Result can be examined in suitable GIS software.

## Approach
### Model selection
The selected model and the pretrained weights should be suitable for sub-meter spatial resolutions as the original aerial image has a pixel size of 0.016m. Ideally, the model should be trained on aerial imagery with a similar scope so that prediction quality can benefit from the similarity of spatial features in the training and prediciton data. A model trained on satellite data with i.e. 10m spatial resolution would contain a substantially different set of features for the tree class. Resampling the aerial image to 10m resolution would come at the cost of loosing the fine-grained features. 
As the task is to segment the image into tree and non-tree areas with a binary mask, a model trained in binary tree segmentation would suit best. Alternatively, one could use model a trained on a multiclass scenario and only use the tree class for the final product. The [Restor Foundation Tree Crown Delineation Pipeline](https://github.com/Restor-Foundation/tcd/tree/main) provides models and pretrained weights for tree crown detection using the OpenAerialMap (OAM) dataset and thus is fitting for the presented setup. The [TorchGeo](https://github.com/torchgeo/torchgeo) Python package provides a model zoo of semantic segmentation models alongside a selection of pretrained weights, including a UNet trained on the aforementioned tree crown detction scenario. Since no changes to model architecture are needed, an out-of-the-box approach like this is sufficient. 

### Preprocessing
The Unet was trained on OAM images with a 0.1m spatial resolution. To make sure this model is applicable to the example image, a resampling step is conducted. The affine transformation is rescaled alongside the image to ensure predicted tree patches can later-on be translated to georeferenced polygons. The image is then procedurally processed as 1024x1024 tiles, as expected by the model. It is not necessary to normalize the image before passing it to the model since this is handled downstream by the TorchGeo pipeline of the model. The expected modality is RGB with pixel ranges of 0-255. A correct tiling process is ensured by a preliminary constant padding of the image.

### Postprocessing
The per-tile prediction results are pieced together into one image and padding is removed. Then, using the affine transformation, all pixel areas with positive class labels (i.e. tree = 1) are used to form vector geometries in the coordinate reference system of the initial image. This data is then saved as a geojson file, which can be reviewed alongside the original image in a GIS software tool to evaluate prediction quality. 

## Discussion
- Saving the results as vector geometries directly out of the predictions saves the effort of first downsampling the image and then upsampling the results and thus mitigates any alignment issues. 
- Since the pretrained TCD UNet was trained on the OAM dataset which is also the origin of the example image, it can not be ruled out that this image was used during the training process. This affects the expressiveness of the prediciton result.

## Visualization
<table>
  <tr>
    <td align="center">
      <strong>Original image</strong><br>
      <img src="https://github.com/kennethweitzel/treeseg/blob/main/images/tree_seg_orig.png" width="400"/>
    </td>
    <td align="center">
      <strong>Original + Prediction</strong><br>
      <img src="https://github.com/kennethweitzel/treeseg/blob/main/images/tree_seg_complete.png" width="400"/>
    </td>
  </tr>
</table>


