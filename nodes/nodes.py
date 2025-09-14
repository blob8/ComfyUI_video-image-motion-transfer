from nodes import VAEDecode
import comfy
import numpy as np
import cv2
import torch
from tqdm import tqdm
import torch.nn.functional as F
import comfy.model_management



def rescale_to_megapixels(images, x):
    N, H, W, C = images.shape
    target_pixels = x * 1_000_000
    scale = (target_pixels / (H * W)) ** 0.5
    new_H, new_W = round(H * scale), round(W * scale)

    images = images.permute(0, 3, 1, 2)
    resized = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
    return resized.permute(0, 2, 3, 1)

def pad_to_multiple_of_8(img):
    h, w = img.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    if img.ndim == 2:
        pad_width = ((0, pad_h), (0, pad_w))
    elif img.ndim == 3:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    padded_img = np.pad(img, pad_width, mode="constant")
    return padded_img, (pad_h, pad_w)

def crop_flow(flow, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        flow = flow[..., :-pad_h, :]
    if pad_w > 0:
        flow = flow[..., :, :-pad_w]
    return flow


class VideoImageWarp:     
    

    def __init__(self):
        self.vae_decoder = VAEDecode()
        self.device = comfy.model_management.intermediate_device()
        self.last_video = {"id":None, "precomputed_flow":None,  "compute_flow_resolution":None, "dualtvl1_params": None}

    
    @classmethod
    def INPUT_TYPES(cls):
               
        return {"required": {
                    "video_images": ("IMAGE",),
                    "compute_flow_resolution": ("FLOAT",{"default":0.15, "step":0.01, "min":0.1, "max":2.5}),
                    "stabilization": ("FLOAT",{"default":0.01, "step":0.001, "min":0.0, "max":1}),
                    "motion_scale": ("FLOAT",{"default":1, "step":0.01, "min":-999, "max":999}),
                    "grid_spacing": ("INT", {"default":10, "step":1, "min":1}),
                    "first_frame": ("IMAGE",),
                    "warp_smooting": ("FLOAT",{"default":3, "step":0.01, "min":0., "max":100}),
                    "tau": ("FLOAT",{"default":0.25, "step":0.01, "min":0.1, "max":0.5}),
                    "lambda_": ("FLOAT",{"default":0.15, "step":0.01, "min":0.05, "max":0.5}),
                    "theta": ("FLOAT",{"default":0.3, "step":0.01, "min":0.1, "max":0.5}),
                    "nscales": ("INT",{"default":5, "step":1, "min":3, "max":8}),
                    "warps": ("INT",{"default":5, "step":1, "min":3, "max":10}),
                    "epsilon": ("FLOAT",{"default":0.01, "step":0.001, "min":0.001, "max":0.01}),
                    "inner_iterations": ("INT",{"default":30, "step":1, "min":10, "max":50}),
                    "outer_iterations": ("INT",{"default":10, "step":1, "min":3, "max":15}),
                    
                    },
                "optional":{
                    "influence_map": ("IMAGE",),
                }
                }
        

    FUNCTION = "video_image_warp"
    CATEGORY = "Image to video"
    OUTPUT_NODE = False
    RETURN_TYPES = ("IMAGE",)


    def video_image_warp(self, video_images, grid_spacing, first_frame, compute_flow_resolution, warp_smooting, motion_scale, tau, lambda_, theta, nscales, warps,epsilon, inner_iterations, outer_iterations, stabilization, influence_map=None):
        if len(first_frame.shape) == 4:
            first_frame = first_frame.squeeze(0)
        dualtvl1_params = {
            'tau': tau,
            'lambda_': lambda_,
            'theta': theta,
            'nscales': nscales,
            'warps': warps,
            'epsilon': epsilon,
            'innnerIterations': inner_iterations,
            'outerIterations': outer_iterations
        }
        current_id = id(video_images)
        reuse_precomputed = self.last_video['id'] == current_id and self.last_video['compute_flow_resolution'] == compute_flow_resolution and self.last_video['dualtvl1_params'] == dualtvl1_params

        video_images = rescale_to_megapixels(video_images,compute_flow_resolution)

        if influence_map!=None:
            influence_map = F.interpolate(influence_map.permute(0,3,1,2), size=(first_frame.shape[-3],first_frame.shape[-2]), align_corners=True, mode="bicubic")[0,0]
            influence_map -=influence_map.min(); influence_map /=influence_map.max()
            
        video_images = [image.cpu().numpy() for image in video_images]

        grid_points, warped_grids, self.last_video['precomputed_flow'] = self.compute_video_warp(video_images, grid_spacing=grid_spacing, sigma_spatial=warp_smooting, dualtvl1_params=dualtvl1_params, stabilization=stabilization, precomputed=self.last_video['precomputed_flow'] if reuse_precomputed else None)
        
        self.last_video['id'] = current_id; self.last_video['compute_flow_resolution'] = compute_flow_resolution; self.last_video['dualtvl1_params'] = dualtvl1_params

        scale_ratio = first_frame.shape[0] / video_images[0].shape[0]
        
        warped_frames = []
        for warped_grid in tqdm(warped_grids, desc="Warping frames"):
            if comfy.model_management.interrupt_processing: raise comfy.model_management.InterruptProcessingException
            warped_frames.append(self.warp_frame(grid_points*scale_ratio, warped_grid*scale_ratio, first_frame.clone(), influence_map=influence_map, motion_scale=motion_scale))
            
        
        output = torch.stack(warped_frames,dim=0).to(self.device)
        return (output,)
    

    def compute_video_warp(self, video, grid_spacing, sigma_spatial, dualtvl1_params, stabilization, precomputed):
        
        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(**dualtvl1_params)

        new_precomputed = []
        first_frame = video[0]
        h, w = first_frame.shape[:2]

        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        y, x = np.mgrid[0:h:grid_spacing, 0:w:grid_spacing]
        if x[0, -1] != w - 1:
            x = np.hstack((x, np.full((x.shape[0], 1), w - 1)))
            y = np.hstack((y, y[:, -1:]))
        if y[-1, 0] != h - 1:
            x = np.vstack((x, x[-1:, :]))
            y = np.vstack((y, np.full((1, x.shape[1]), h - 1)))
        grid_points = np.vstack((x.ravel(), y.ravel())).T

        accum_flow = np.zeros((h, w, 2), dtype=np.float32)
        warped_grids = []

        for i, frame in enumerate(tqdm(video, desc="Estimating flow")):
            if comfy.model_management.interrupt_processing:
                raise comfy.model_management.InterruptProcessingException

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if precomputed:
                flow = precomputed[i]
            else:
                flow = optical_flow.calc(prev_gray, curr_gray, None)

            new_precomputed.append(flow)
            accum_flow += flow
            accum_flow *= (1-stabilization)
            if sigma_spatial > 0:
                accum_flow[..., 0] = cv2.GaussianBlur(accum_flow[..., 0], (0, 0), sigma_spatial)
                accum_flow[..., 1] = cv2.GaussianBlur(accum_flow[..., 1], (0, 0), sigma_spatial)

            gx, gy = grid_points[:, 0].astype(int), grid_points[:, 1].astype(int)
            gx = np.clip(gx, 0, w - 1)
            gy = np.clip(gy, 0, h - 1)
            sampled_flow = accum_flow[gy, gx]

            warped_grids.append(grid_points + sampled_flow)

            prev_gray = curr_gray

        return grid_points, warped_grids, new_precomputed


    
    @torch.no_grad()
    def warp_frame(self, grid_points, warped_grid, frame, influence_map, motion_scale):


        if frame.ndim == 3 and frame.shape[2] in (1,3,4):
            frame_t = frame.permute(2,0,1).to(self.device)
        elif frame.ndim == 3 and frame.shape[0] in (1,3,4):
            frame_t = frame.to(self.device)


        C, H, W = frame_t.shape

        gp = torch.as_tensor(np.asarray(grid_points), dtype=torch.float32, device=self.device)
        wg = torch.as_tensor(np.asarray(warped_grid), dtype=torch.float32, device=self.device)
        N = gp.shape[0]

        if influence_map!=None:
            influence_map = influence_map.to(self.device)

        xs = torch.unique(gp[:,0]); ys = torch.unique(gp[:,1])
        gw, gh = xs.numel(), ys.numel()
        if gw * gh != N:
            raise RuntimeError("Cannot detect regular grid shape")

        gp_round = torch.round(gp)
        sort_idx = np.lexsort((gp_round[:,0].cpu().numpy(), gp_round[:,1].cpu().numpy()))
        idx_grid = torch.as_tensor(sort_idx, dtype=torch.long, device=self.device).view(gh, gw)

        px = torch.clamp(gp_round[:,0].long(), 0, W-1)
        py = torch.clamp(gp_round[:,1].long(), 0, H-1)

        if influence_map!=None:
            vertex_influence = influence_map[py, px][:,None] * motion_scale
        else:
            vertex_influence = motion_scale

        u = (wg - gp) * vertex_influence
        u = u[idx_grid.view(-1)].view(gh, gw, 2)

        u_grid = u.permute(2,0,1).unsqueeze(0)
        u_up = F.interpolate(u_grid, size=(H, W), mode='bilinear', align_corners=True)
        u_up = u_up[0].permute(1,2,0)

        yy, xx = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        src_x = torch.clamp(xx - u_up[...,0], 0, W-1)
        src_y = torch.clamp(yy - u_up[...,1], 0, H-1)

        grid_x = 2.0 * src_x / (W-1) - 1.0
        grid_y = 2.0 * src_y / (H-1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        warped = F.grid_sample(frame_t.unsqueeze(0), grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        return warped[0].permute(1,2,0).contiguous()