import utils.img_f as img_f
import numpy as np
from scipy.interpolate import griddata
import sys

INTERPOLATION = {
    "linear": 1,
    "bilinear": 2,
    "cubic": 2
}

def warp_image(img, random_state=None, **kwargs):
    if img.shape[0]<=5 or img.shape[1]<=5:
        return img
    if random_state is None:
        random_state = np.random.RandomState()

    w_mesh_interval = kwargs.get('w_mesh_interval', 12)
    w_mesh_std = kwargs.get('w_mesh_std', 1.5)

    h_mesh_interval = kwargs.get('h_mesh_interval', 12)
    h_mesh_std = kwargs.get('h_mesh_std', 1.5)

    interpolation_method = kwargs.get('interpolation', 'linear')

    h, w = img.shape[:2]

    if kwargs.get("fit_interval_to_image", True):
        # Change interval so it fits the image size
        w_ratio = w / float(w_mesh_interval)
        h_ratio = h / float(h_mesh_interval)

        w_ratio = max(1, round(w_ratio))
        h_ratio = max(1, round(h_ratio))

        w_mesh_interval = w / w_ratio
        h_mesh_interval = h / h_ratio
        ############################################

    # Get control points
    source = np.mgrid[0:h+h_mesh_interval:h_mesh_interval, 0:w+w_mesh_interval:w_mesh_interval]
    source = source.transpose(1,2,0).reshape(-1,2)

    if kwargs.get("draw_grid_lines", False):
        if len(img.shape) == 2:
            color = 0
        else:
            color = np.array([0,0,255])
        for s in source:
            img[int(s[0]):int(s[0])+1,:] = color
            img[:,int(s[1]):int(s[1])+1] = color

    # Perturb source control points
    destination = source.copy()
    source_shape = source.shape[:1]
    destination[:,0] = destination[:,0] + random_state.normal(0.0, h_mesh_std, size=source_shape)
    destination[:,1] = destination[:,1] + random_state.normal(0.0, w_mesh_std, size=source_shape)

    # Warp image
    grid_x, grid_y = np.mgrid[0:h, 0:w]
    grid_z = griddata(destination, source, (grid_x, grid_y), method=interpolation_method).astype(np.float32)
    map_x = grid_z[:,:,1]
    map_y = grid_z[:,:,0]
    meanV = img.mean()
    warped = img_f.remap(img, map_x, map_y, INTERPOLATION[interpolation_method], borderValue=(meanV,meanV,meanV))

    return warped

if __name__ == "__main__":
    input_image = sys.argv[1]
    if len(sys.argv)>2:
        output_image = sys.argv[2]
    else:
        output_image=None
    img = img_f.imread(input_image)
    img = warp_image(img, draw_grid_lines=True)
    if output_image is None:
        img_f.imshow('warped', img)
        img_f.show()
    else:
        img_f.imwrite(output_image, img)
