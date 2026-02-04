import matplotlib
matplotlib.use('Agg') # Essential for headless servers (no GUI)
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

# SMPL Kinetic Chain (Indices for connecting joints)
# 0:Pelvis, 1:L_Hip, 2:R_Hip, 3:Spine1, 6:Spine2, 9:Spine3, 
# 12:Neck, 15:Head, 4:L_Knee, 7:L_Ankle, 5:R_Knee, 8:R_Ankle
# 13:L_Collar, 16:L_Shoulder, 18:L_Elbow, 20:L_Wrist
# 14:R_Collar, 17:R_Shoulder, 19:R_Elbow, 21:R_Wrist
SKELETON_PARENTS = [
    (0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9),
    (9,12), (12,15), (9,13), (9,14), (13,16), (14,17), (16,18), 
    (17,19), (18,20), (19,21)
]

def render_debug_video(video_path, all_joints, floor_model, output_path="debug_output.mp4"):
    """
    Generates a side-by-side video: [Original | 3D Skeleton + Floor]
    """
    print(f"Generating Visual Report: {output_path}...")
    
    # 1. Setup Video Inputs/Outputs
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # We will make the side-by-side video double width
    out_size = (width * 2, height)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)
    
    # 2. Setup Matplotlib Figure
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Pre-calculate floor grid for visualization
    # We create a meshgrid covering the Z-depth (0 to 5m) and X-width (-2 to 2m)
    gx = np.linspace(-2, 2, 10)
    gz = np.linspace(0, 6, 10) # Assuming scene is 0-6m deep
    GX, GZ = np.meshgrid(gx, gz)
    
    # Predict Floor Y for every point on the grid
    # Note: floor_model expects (N, 1) input for Z
    GY = floor_model.predict(GZ.flatten().reshape(-1, 1)).reshape(GX.shape)

    frame_idx = 0
    with tqdm(total=len(all_joints), desc="Rendering Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= len(all_joints):
                break
                
            joints = all_joints[frame_idx] # Shape (24, 3)
            
            # --- RENDER 3D PLOT ---
            ax.clear()
            
            # Plot Floor (Wireframe for visibility)
            # We invert Y for plotting so "Up" looks like Up
            ax.plot_wireframe(GX, GZ, -GY, color='gray', alpha=0.3, linewidth=0.5)
            
            if joints is not None and len(joints) > 0:
                # Extract coordinates
                xs = joints[:, 0]
                ys = joints[:, 1]
                zs = joints[:, 2]
                
                # Plot Joints (Invert Y)
                ax.scatter(xs, zs, -ys, c='red', s=20)
                
                # Plot Bones
                for p1, p2 in SKELETON_PARENTS:
                    if p1 < len(joints) and p2 < len(joints):
                        ax.plot([xs[p1], xs[p2]], 
                                [zs[p1], zs[p2]], 
                                [-ys[p1], -ys[p2]], color='blue')
            
            # Setup Camera View
            ax.set_xlabel('X (Right)')
            ax.set_ylabel('Z (Depth)')
            ax.set_zlabel('-Y (Up)')
            ax.set_title(f"Frame {frame_idx}")
            
            # Lock axis limits to prevent "shaking" camera
            ax.set_xlim(-2, 2)
            ax.set_ylim(0, 5)
            ax.set_zlim(-2, 2) # Adjust depending on your scene scale
            ax.view_init(elev=20, azim=45) # Nice diagonal view

            # Convert Matplotlib Figure to Image Buffer
            fig.canvas.draw()
            
            # MODERN FIX: Get RGBA buffer directly (Works on Matplotlib 3.8+)
            plot_img = np.asarray(fig.canvas.buffer_rgba())
            
            # Convert RGBA (Matplotlib) to BGR (OpenCV)
            # This handles both the Alpha channel removal and color swap
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
            
            # Resize plot to match video height
            plot_img = cv2.resize(plot_img, (width, height))    
        
            # Concatenate Side-by-Side
            combined = np.hstack((frame, plot_img))
            writer.write(combined)
            
            frame_idx += 1
            pbar.update(1)
            
    cap.release()
    writer.release()
    plt.close()
    print(f"Video saved to {output_path}")

def plot_volume_trend(all_volumes, output_path="volume_trend.png"):
    """
    Generates a static plot of the Shape (Volume) component over time.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(all_volumes, color='purple', linewidth=2)
    plt.title("LMA Shape Component: Body Volume vs Time")
    plt.xlabel("Frame")
    plt.ylabel("Convex Hull Volume (m^3)")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"Volume plot saved to {output_path}")
    