import export_pixel_mask
import shapely
from rasterio import features
import rasterio
import load_images
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from shapely import geometry
from shapely.wkt import loads as wkt_loads


gs = pd.read_csv('data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

def mask_to_polygons(mask, xymax):
    all_polygons=[]

    x_scale = xymax[0] / mask.shape[1]
    y_scale = xymax[1] / mask.shape[0]

    for shape, value in features.shapes(mask.astype(np.int16),
                                mask = (mask==1),
                                transform = rasterio.Affine(x_scale, 0, 0, 0, y_scale, 0)):

        all_polygons.append(shapely.geometry.shape(shape))
        
    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        #Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        #need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons

def polycons_to_string(polycons):
    result = '\n'.join(map())

def test():

    image_id = "6120_2_2"
    n = 1.
    image_size = (int(3396*n), int(3349*n))

    xymax = export_pixel_mask._get_xmax_ymin(gs,image_id)
    
    labels = np.zeros(image_size, dtype='uint8')
    print(labels.shape)
    labels[:,:] = 1

    print(labels.shape, labels.dtype)

    polycons = mask_to_polygons(labels, xymax)

    back_to_labels = export_pixel_mask.polygon_list_to_mask(polycons, image_size, xymax)#(image_size[1], image_size[0]))# (image_size[0]+1, image_size[1]+1))

    print('accuracy: ', 1-np.sum(np.abs(back_to_labels.astype('float') - labels.astype('float'))) / np.prod(labels.shape, dtype='float64'))

    print(mask_to_polygons(back_to_labels, xymax))

    print(back_to_labels.shape)

    plt.subplot(2, 2, 1); plt.imshow(labels); plt.title('original labels')
    plt.subplot(2, 2, 2); plt.imshow(back_to_labels); plt.title('back to labels')
    plt.subplot(2, 2, 3); plt.imshow(np.abs(labels - back_to_labels)); plt.title('difference')

    plt.show()

if __name__ == "__main__":
    test()