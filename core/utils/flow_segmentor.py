
import selectivesearch
import numpy as np
import cv2
import frame_utils
import os
import flow_vis
import matplotlib
import matplotlib.pyplot as plt

def segment(image, flow):

    # image_lbl, image_regions = selectivesearch.selective_search(image)
    flow_lbl, flow_regions = selectivesearch.selective_search(flow)

    h, w, _ = flow_lbl.shape
    masks = list()
    for i in range(10):
        mask = flow_lbl[..., -1] == i
        masks.append(mask)

    return masks

if __name__ == "__main__":

    image = frame_utils.read_gen(os.path.join("/Users/damien/Downloads/FlyingChairs_release/data", "00041_img1.ppm"))
    flow = frame_utils.read_gen(os.path.join("/Users/damien/Downloads/FlyingChairs_release/data", "00041_flow.flo"))

    image = np.asarray(image)
    flow = flow_vis.flow_to_color(flow, convert_to_bgr=False)

    masks = segment(image, flow)

    cm = plt.get_cmap("Pastel1")
    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=9)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
    seg_image = np.zeros_like(image)
    for i in range(len(masks)):
        color = scalarMap.to_rgba(i)
        seg_image += masks[i] * color



