#!/usr/bin/env python3
import argparse
import json

import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
from torchgeo.models import Unet_Weights, unet


def segment_trees(input_path, output_path,  device='cuda'):
    """Segment trees in a TIFF image and save as GeoJSON.
    
    Args:
        input_path: Path to input TIFF file.
        output_path: Path to output GeoJSON file.
        device: Device to run inference on (default: 'cuda').
    """
    # read tiff image
    with rasterio.open(input_path) as src:

        full_spatial_res = 0.016  # spatial resolution of the original image
        scale_factor = full_spatial_res / 0.1  # scale factor to get from 0.016m to 0.1m spatial resolution
        ovv_spatial_res = full_spatial_res / scale_factor
        print(f"Spatial resolution of the scaled image: {ovv_spatial_res} m")

        # read and scale image
        image = src.read(
            out_shape=(
                src.count,
                int(src.height * scale_factor),
                int(src.width * scale_factor)
            ),
            resampling=Resampling.nearest
        )

        # new height and width after scaling
        height = image.shape[1]
        width = image.shape[2]

        # scale image transform
        transform_scaled = src.transform * src.transform.scale(
            (src.width / image.shape[2]),
            (src.height / image.shape[1])
        )

        print('Shape of scaled image: ', image.shape)

    # load Unet model with pretrained weights for tree crown detection
    weights = Unet_Weights.OAM_RGB_RESNET34_TCD
    model = unet(weights)
    if device == 'cuda' and torch.cuda.is_available():
        model.cuda()
    model.eval()

    # calculate padding for tiling
    tile_size = 1024 # model input size
    pad_H = (np.ceil(height / tile_size) * tile_size - height).astype(int) if height % tile_size != 0 else 0
    pad_W = (np.ceil(width / tile_size) * tile_size - width).astype(int) if width % tile_size != 0 else 0

    # pad image before tiling
    padded_image = np.pad(image, ((0, 0), (0, pad_H), (0, pad_W)), mode='constant', constant_values=0)
    # container array for segmentation results
    tree_seg_padded = np.zeros_like(padded_image[0], dtype=np.uint8)

    # loop over all tiles
    num_tiles = 0
    for y in range(0, padded_image.shape[1], tile_size):
        for x in range(0, padded_image.shape[2], tile_size):

            tile = padded_image[:, y:y + tile_size, x:x + tile_size]
            input_tensor = torch.from_numpy(tile).float().unsqueeze(0).to(device)

            # no normalization needed since this is handled downstream by the model. Expected input range is [0, 255]

            # run model on input
            with torch.inference_mode():
                logits = model(input_tensor)

            # use argmay of predicted logits to get class labels (0 or 1)
            pred = logits.argmax(dim=1)[0]
            
            # save predicted tile in the correct location of the padded tree segmentation array
            tree_seg_padded[y:y + tile_size, x:x + tile_size] = pred.detach().cpu().numpy()
            num_tiles += 1
    print(f"Processed image as {num_tiles} tiles.")

    # crop the padded tree mask back to original size
    tree_seg = tree_seg_padded[:height, :width]
    
    # sanity checks
    assert tree_seg.shape == (image.shape[1], image.shape[2]), f"Cropped tree mask shape {tree_seg.shape} does not match expected shape {(height, width)}"
    assert np.unique(tree_seg).tolist() == [0, 1], f"Unexpected class labels in tree segmentation: {np.unique(tree_seg)}"

    # create GeoJSON features from tree segmentation
    # resulting polygons will be in the same coordinate reference system as the input image, and can be visualized in GIS software or used for further analysis
    tree_mask = tree_seg == 1
    features = []
    for geom, value in rasterio.features.shapes(tree_seg, mask=tree_mask, transform=transform_scaled):
        feature = {
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "value": int(value)
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"Save tree segments as vector geometries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment trees in aerial imagery."
    )
    parser.add_argument(
        "input", help="Path to input TIFF file"
    )
    parser.add_argument(
        "--output", help="Path to output GeoJSON file"
    )
    parser.add_argument(
        "--device", default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)"
    )

    args = parser.parse_args()

    segment_trees(
        args.input,
        args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()