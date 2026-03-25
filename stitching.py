'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def get_matches(img1: torch.Tensor, img2: torch.Tensor, max_size: int = 800):
    """Get feature matches between two images using LoFTR."""
    img1_f = img1.float() / 255.0
    img2_f = img2.float() / 255.0
    if img1_f.dim() == 3:
        img1_f = img1_f.unsqueeze(0)
    if img2_f.dim() == 3:
        img2_f = img2_f.unsqueeze(0)
    
    # Downscale large images to save memory
    _, _, h1, w1 = img1_f.shape
    _, _, h2, w2 = img2_f.shape
    
    scale1 = 1.0
    scale2 = 1.0
    
    if max(h1, w1) > max_size:
        scale1 = max_size / max(h1, w1)
        new_h1, new_w1 = int(h1 * scale1), int(w1 * scale1)
        img1_f = torch.nn.functional.interpolate(img1_f, size=(new_h1, new_w1), mode='bilinear', align_corners=False)
    
    if max(h2, w2) > max_size:
        scale2 = max_size / max(h2, w2)
        new_h2, new_w2 = int(h2 * scale2), int(w2 * scale2)
        img2_f = torch.nn.functional.interpolate(img2_f, size=(new_h2, new_w2), mode='bilinear', align_corners=False)
    
    gray1 = K.color.rgb_to_grayscale(img1_f)
    gray2 = K.color.rgb_to_grayscale(img2_f)
    
    matcher = K.feature.LoFTR(pretrained='outdoor')
    with torch.no_grad():
        correspondences = matcher({'image0': gray1, 'image1': gray2})
    
    pts1 = correspondences['keypoints0']
    pts2 = correspondences['keypoints1']
    conf = correspondences['confidence']
    
    # Scale keypoints back to original resolution
    if scale1 != 1.0:
        pts1 = pts1 / scale1
    if scale2 != 1.0:
        pts2 = pts2 / scale2
    
    # Filter by confidence (top matches only)
    if len(conf) > 50:
        _, top_idx = conf.topk(min(500, len(conf)))
        pts1 = pts1[top_idx]
        pts2 = pts2[top_idx]
    
    return pts1, pts2

def compute_homography(pts1: torch.Tensor, pts2: torch.Tensor):
    """Compute homography from pts1 to pts2 using RANSAC."""
    H, inliers = K.geometry.ransac.RANSAC(
        model_type='homography',
        inl_th=3.0,
        max_iter=2000,
        confidence=0.9999
    )(pts1, pts2)
    return H.unsqueeze(0)

def compute_gradient_magnitude(img: torch.Tensor):
    """Compute gradient magnitude using Sobel operator."""
    gray = K.color.rgb_to_grayscale(img)
    grad = K.filters.spatial_gradient(gray, order=1)
    grad_x = grad[:, :, 0, :, :]
    grad_y = grad[:, :, 1, :, :]
    mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    return mag

def create_distance_weights(mask: torch.Tensor, iterations: int = 30):
    """Create distance-based weights from mask edges (feathering) using erosion."""
    if mask.sum() == 0:
        return mask
    
    kernel = torch.ones(3, 3)
    weights = torch.zeros_like(mask)
    current = mask.clone()
    
    for i in range(iterations):
        eroded = K.morphology.erosion(current, kernel)
        layer = current - eroded
        weights = weights + layer * (i + 1)
        current = eroded
        if current.sum() == 0:
            break
    
    weights = weights + current * (iterations + 1)
    
    max_w = weights.max()
    if max_w > 0:
        weights = weights / max_w
    
    return weights * mask

def morphological_clean(mask: torch.Tensor, kernel_size: int = 5):
    """Clean mask using morphological operations."""
    kernel = torch.ones(kernel_size, kernel_size)
    mask_opened = K.morphology.opening(mask, kernel)
    mask_closed = K.morphology.closing(mask_opened, kernel)
    return mask_closed

def gaussian_pyramid(img: torch.Tensor, levels: int = 4):
    """Build Gaussian pyramid."""
    pyramid = [img]
    current = img
    for _ in range(levels - 1):
        current = K.filters.gaussian_blur2d(current, (5, 5), (1.5, 1.5))
        current = torch.nn.functional.interpolate(current, scale_factor=0.5, mode='bilinear', align_corners=False)
        pyramid.append(current)
    return pyramid

def laplacian_pyramid(img: torch.Tensor, levels: int = 4):
    """Build Laplacian pyramid."""
    gauss_pyr = gaussian_pyramid(img, levels)
    lap_pyr = []
    for i in range(levels - 1):
        upsampled = torch.nn.functional.interpolate(gauss_pyr[i+1], size=gauss_pyr[i].shape[2:], mode='bilinear', align_corners=False)
        lap_pyr.append(gauss_pyr[i] - upsampled)
    lap_pyr.append(gauss_pyr[-1])
    return lap_pyr

def reconstruct_from_laplacian(lap_pyr):
    """Reconstruct image from Laplacian pyramid."""
    img = lap_pyr[-1]
    for i in range(len(lap_pyr) - 2, -1, -1):
        img = torch.nn.functional.interpolate(img, size=lap_pyr[i].shape[2:], mode='bilinear', align_corners=False)
        img = img + lap_pyr[i]
    return img

def multiband_blend(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor, levels: int = 4):
    """Multi-band (Laplacian pyramid) blending."""
    lap1 = laplacian_pyramid(img1, levels)
    lap2 = laplacian_pyramid(img2, levels)
    
    mask_pyr = gaussian_pyramid(mask.expand_as(img1), levels)
    
    blended_pyr = []
    for l1, l2, m in zip(lap1, lap2, mask_pyr):
        blended_pyr.append(m * l1 + (1 - m) * l2)
    
    return reconstruct_from_laplacian(blended_pyr)

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Stitch two images, eliminating moving foreground.
    Simple and robust approach: detect difference, pick brighter pixel (person is darker).
    """
    img_list = list(imgs.values())
    img1, img2 = img_list[0], img_list[1]
    
    pts1, pts2 = get_matches(img1, img2)
    H = compute_homography(pts2, pts1)
    
    C, H1, W1 = img1.shape
    C, H2, W2 = img2.shape
    
    corners2 = torch.tensor([[0, 0], [W2, 0], [W2, H2], [0, H2]], dtype=torch.float32).unsqueeze(0)
    corners2_t = K.geometry.transform_points(H, corners2).squeeze(0)
    corners1 = torch.tensor([[0, 0], [W1, 0], [W1, H1], [0, H1]], dtype=torch.float32)
    all_corners = torch.cat([corners1, corners2_t], dim=0)
    
    min_xy = all_corners.min(dim=0).values
    max_xy = all_corners.max(dim=0).values
    offset = -min_xy
    canvas_size = (int(max_xy[1] - min_xy[1]), int(max_xy[0] - min_xy[0]))
    
    T = torch.eye(3, dtype=torch.float32)
    T[0, 2], T[1, 2] = offset[0], offset[1]
    
    H1_t = T.unsqueeze(0)
    H2_t = (T @ H.squeeze(0)).unsqueeze(0)
    
    img1_f = img1.float().unsqueeze(0) / 255.0
    img2_f = img2.float().unsqueeze(0) / 255.0
    
    w1 = K.geometry.warp_perspective(img1_f, H1_t, canvas_size)
    w2 = K.geometry.warp_perspective(img2_f, H2_t, canvas_size)
    
    m1 = (w1.sum(dim=1, keepdim=True) > 0).float()
    m2 = (w2.sum(dim=1, keepdim=True) > 0).float()
    overlap = m1 * m2
    
    # Detect foreground: where images differ significantly
    diff = (w1 - w2).abs().mean(dim=1, keepdim=True)
    
    # Adaptive threshold
    overlap_vals = diff[overlap > 0]
    if len(overlap_vals) > 0:
        threshold = overlap_vals.quantile(0.75).item()
        threshold = max(0.04, min(threshold, 0.15))
    else:
        threshold = 0.06
    
    fg_mask = (diff > threshold) * overlap
    
    # Simple morphological cleanup
    kernel = torch.ones(5, 5)
    fg_mask = K.morphology.closing(fg_mask, kernel)
    fg_mask = K.morphology.opening(fg_mask, kernel)
    
    # Background region in overlap
    bg_mask = overlap * (1 - fg_mask)
    
    # For foreground pixels: person is typically darker than background (pumpkins, etc.)
    # So pick the BRIGHTER pixel (higher mean intensity)
    brightness1 = w1.mean(dim=1, keepdim=True)
    brightness2 = w2.mean(dim=1, keepdim=True)
    
    use_w1 = (brightness1 >= brightness2).float()
    fg_result = use_w1 * w1 + (1 - use_w1) * w2
    
    # Final composition
    result = torch.zeros_like(w1)
    
    # Non-overlap regions
    result = result + w1 * m1 * (1 - m2)
    result = result + w2 * m2 * (1 - m1)
    
    # Background overlap: average
    result = result + (w1 + w2) / 2 * bg_mask
    
    # Foreground: use brighter pixel
    result = result + fg_result * fg_mask
    
    return (result.squeeze(0) * 255).clamp(0, 255).to(torch.uint8)

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Stitch multiple images into a panorama.
    Returns panorama image and NxN overlap matrix.
    """
    names = list(imgs.keys())
    img_list = list(imgs.values())
    n = len(img_list)
    
    overlap = torch.zeros((n, n), dtype=torch.float32)
    pairwise_H = {}
    pairwise_matches = {}
    
    MIN_MATCHES = 20
    
    # Step 1: Find pairwise matches and homographies
    for i in range(n):
        overlap[i, i] = 1
        for j in range(i + 1, n):
            try:
                pts_i, pts_j = get_matches(img_list[i], img_list[j])
                if len(pts_i) >= MIN_MATCHES:
                    H_j_to_i = compute_homography(pts_j, pts_i)
                    H_mat = H_j_to_i.squeeze(0)
                    
                    # Check homography is reasonable
                    det = torch.det(H_mat[:2, :2])
                    if 0.2 < det < 5.0:
                        overlap[i, j] = 1
                        overlap[j, i] = 1
                        pairwise_H[(j, i)] = H_j_to_i
                        pairwise_H[(i, j)] = torch.inverse(H_mat).unsqueeze(0)
                        pairwise_matches[(i, j)] = len(pts_i)
            except:
                pass
    
    # Step 2: Find reference image (most connections)
    ref_idx = 0
    max_conn = 0
    for i in range(n):
        conn = overlap[i].sum().item() - 1  # exclude self
        if conn > max_conn:
            max_conn = conn
            ref_idx = i
    
    # Step 3: Build homographies to reference using BFS
    H_to_ref = {ref_idx: torch.eye(3, dtype=torch.float32)}
    visited = {ref_idx}
    queue = [ref_idx]
    
    while queue:
        curr = queue.pop(0)
        for other in range(n):
            if other not in visited and overlap[curr, other] == 1:
                if (other, curr) in pairwise_H:
                    H_other_to_curr = pairwise_H[(other, curr)].squeeze(0)
                    H_to_ref[other] = H_to_ref[curr] @ H_other_to_curr
                    visited.add(other)
                    queue.append(other)
    
    # Step 4: Compute canvas size
    all_corners = []
    for i in H_to_ref:
        C, Hi, Wi = img_list[i].shape
        corners = torch.tensor([[0, 0], [Wi, 0], [Wi, Hi], [0, Hi]], dtype=torch.float32)
        H = H_to_ref[i]
        
        # Transform corners
        corners_h = torch.cat([corners, torch.ones(4, 1)], dim=1)
        transformed = (H @ corners_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        all_corners.append(transformed)
    
    all_corners = torch.cat(all_corners, dim=0)
    min_xy = all_corners.min(dim=0).values
    max_xy = all_corners.max(dim=0).values
    
    # Offset to make all coordinates positive
    offset_x = -min_xy[0].item()
    offset_y = -min_xy[1].item()
    
    canvas_w = int(max_xy[0].item() - min_xy[0].item())
    canvas_h = int(max_xy[1].item() - min_xy[1].item())
    
    # Limit canvas size
    canvas_w = min(canvas_w, 3000)
    canvas_h = min(canvas_h, 3000)
    
    # Translation matrix
    T = torch.tensor([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # Step 5: Warp and blend all images with center weighting
    canvas = torch.zeros((3, canvas_h, canvas_w), dtype=torch.float32)
    weight_canvas = torch.zeros((1, canvas_h, canvas_w), dtype=torch.float32)
    
    for i in H_to_ref:
        img = img_list[i].float() / 255.0
        C, Hi, Wi = img.shape
        
        # Create center-weighted mask for original image
        y_weight = torch.linspace(0, 1, Hi)
        y_weight = torch.min(y_weight, 1 - y_weight) * 2  # 0 at edges, 1 at center
        x_weight = torch.linspace(0, 1, Wi)
        x_weight = torch.min(x_weight, 1 - x_weight) * 2
        
        center_weight = y_weight.view(Hi, 1) * x_weight.view(1, Wi)
        center_weight = center_weight.unsqueeze(0)  # 1 x H x W
        
        # Final homography
        H_final = T @ H_to_ref[i]
        H_final = H_final.unsqueeze(0)
        
        # Warp image and weight
        img_batch = img.unsqueeze(0)
        weight_batch = center_weight.unsqueeze(0)
        
        warped = K.geometry.warp_perspective(img_batch, H_final, (canvas_h, canvas_w))
        warped = warped.squeeze(0)
        
        warped_weight = K.geometry.warp_perspective(weight_batch, H_final, (canvas_h, canvas_w))
        warped_weight = warped_weight.squeeze(0)
        
        # Mask for valid pixels
        mask = (warped.sum(dim=0, keepdim=True) > 0).float()
        warped_weight = warped_weight * mask
        warped_weight = warped_weight.clamp(min=0.001)
        
        canvas += warped * warped_weight
        weight_canvas += warped_weight
    
    # Normalize
    weight_canvas = weight_canvas.clamp(min=1e-6)
    result = canvas / weight_canvas
    
    return (result * 255).clamp(0, 255).to(torch.uint8), overlap
