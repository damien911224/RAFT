
import selectivesearch
import numpy as np
import cv2
from .utils import frame_utils
import os
import flow_vis
import matplotlib
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift, watershed, morphological_geodesic_active_contour, slic
from skimage.future import graph
import PIL.ImageOps
import PIL.Image
import time
import tgdm

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    # rad = np.sqrt(np.square(u) + np.square(v))
    rad = np.ones_like(u) * 100.0
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def segment(flow):

    flow_lbl, flow_regions = selectivesearch.selective_search(flow, scale=50000, sigma=0, min_size=1)

    flow_lbl = flow_lbl[..., -1]

    # g = graph.rag_mean_color(image, flow_lbl.astype(np.uint8) + 1)
    # flow_lbl = graph.merge_hierarchical(flow_lbl.astype(np.uint8) + 1, g,
    #                                     thresh=35, rag_copy=False,
    #                                     in_place_merge=True,
    #                                     merge_func=merge_mean_color,
    #                                     weight_func=_weight_mean_color)

    uniques = np.unique(flow_lbl)

    # max_index = -1
    # max_count = -1
    # for i in range(len(uniques)):
    #     count = np.sum((flow_lbl == i).astype(np.uint8))
    #     if np.sum((flow_lbl == i).astype(np.uint8)) >= max_count:
    #         max_count = count
    #         max_index = i
    #
    # masks = list()
    # for i in range(len(uniques)):
    #     if i != max_index:
    #         mask = (flow_lbl == i).astype(np.uint8)
    #         masks.append(mask)

    masks = list()
    for i in range(len(uniques)):
        mask = (flow_lbl == i).astype(np.uint8)
        masks.append(mask)

    return np.asarray(masks)


if __name__ == "__main__":
    data_folder = os.path.join("/mnt/hdd1/damien", "FlyingChairs_release/data")
    flow_paths = os.path.join(data_folder, "*.flo")
    for path in tgdm(flow_paths):
        flow = frame_utils.read_gen(os.path.join("/Users/damien/Downloads/FlyingChairs_release/data", "01447_flow.flo"))
        flow = flow_vis.flow_to_color(flow, convert_to_bgr=False)
        flow = PIL.ImageOps.autocontrast(PIL.Image.fromarray(flow))
        flow = np.asarray(flow)
        masks = segment(flow)
        npy_path = os.path.join(data_folder, os.path.basename(path).split(".")[0] + ".npy")
        np.save(npy_path, masks)

    # # image = frame_utils.read_gen(os.path.join("/Users/damien/Downloads/FlyingChairs_release/data", "01456_img1.ppm"))
    # flow = frame_utils.read_gen(os.path.join("/Users/damien/Downloads/FlyingChairs_release/data", "01447_flow.flo"))
    #
    # # image = np.asarray(image)
    # flow = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    #
    # flow = PIL.ImageOps.autocontrast(PIL.Image.fromarray(flow))
    # flow = np.asarray(flow)
    #
    # start_time = time.time()
    # masks = segment(flow)
    # duration = time.time() - start_time
    # print(duration)
    #
    # cm = plt.get_cmap("Pastel1")
    # cNorm = matplotlib.colors.Normalize(vmin=0, vmax=len(masks) - 1)
    # scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
    # seg_image = np.zeros_like(flow)
    # for i in range(len(masks)):
    #     color = np.round(np.array(scalarMap.to_rgba(i)[:3]) * 255.0).astype(np.uint8)
    #     seg_image += np.expand_dims(masks[i], axis=-1) * color
    #
    # cv2.imwrite(os.path.join("/Users/damien/Downloads/FlyingChairs_release", "seg.png"),
    #             cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(os.path.join("/Users/damien/Downloads/FlyingChairs_release", "flow.png"),
    #             cv2.cvtColor(flow, cv2.COLOR_RGB2BGR))



