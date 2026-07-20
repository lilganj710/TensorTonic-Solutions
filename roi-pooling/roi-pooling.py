import math
import itertools as it


def roi_pool(
      feature_map: list[list[float]], rois: list[list[int]], output_size: int
      ) -> list[list[list[int]]]:
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    roi_grid_list: list[list[list[int]]] = []
    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_h, roi_w = y2 - y1, x2 - x1
        cur_grid = [[0 for _ in range(output_size)] for _ in range(output_size)]
        for i, j in it.product(range(output_size), repeat=2):
            h_start = y1 + (i*roi_h) // output_size
            h_end = max(h_start+1, y1 + ((i+1)*roi_h) // output_size)
            w_start = x1 + (j*roi_w) // output_size
            w_end = max(w_start+1, x1 + ((j+1)*roi_w) // output_size)
            cur_features = [feature_map[row][w_start:w_end] for row in range(h_start, h_end)]
            cur_grid[i][j] = max(it.chain.from_iterable(cur_features))
        roi_grid_list.append(cur_grid)
    return roi_grid_list