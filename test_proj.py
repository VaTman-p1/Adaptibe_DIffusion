import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

def plot_full_uav_sequence(v_curr, p_curr, dt, u1_m, psi=0, tilt_limit_deg=15.0):
    g = 9.81
    limit_rad = np.radians(tilt_limit_deg)
    p_prev = p_curr - v_curr*dt
    
    # 1. State Augmentation: Estimate velocity
    # v_curr = (p_curr - p_prev) / dt
    
    # 2. Generate 16 Reachable Vertices for the NEXT step
    t_limits = [u1_m * 0.2, u1_m * 1.5] 
    phi_limits = [-limit_rad, limit_rad]
    theta_limits = [-limit_rad, limit_rad]
    
    next_vertices = []
    for T in t_limits:
        for phi in phi_limits:
            for theta in theta_limits:
                # Dynamics from your image (at psi=0)
                ax = (np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi)) * T
                ay = (-np.cos(psi)*np.sin(phi) + np.sin(psi)*np.sin(theta)*np.cos(phi)) * T
                az = -g + (np.cos(theta)*np.cos(phi)) * T
                
                accel = np.array([ax, ay, az])
                p_next = p_curr + v_curr * dt + 0.5 * accel * (dt**2)
                next_vertices.append(p_next)
    
    next_vertices = np.array(next_vertices)
    hull = ConvexHull(next_vertices)
    
    # 3. Plotting Setup
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- PLOT 1: PREVIOUS STATE ---
    ax.scatter(p_prev[0], p_prev[1], p_prev[2], 
               color='black', s=150, marker='X', label='Previous Position (p_prev)', zorder=5)
    
    # --- PLOT 2: CURRENT STATE ---
    ax.scatter(p_curr[0], p_curr[1], p_curr[2], 
               color='blue', s=200, marker='D', label='Current Position (p_curr)', zorder=15)
    
    # --- PLOT 3: TRAJECTORY LINE ---
    ax.plot([p_prev[0], p_curr[0]], [p_prev[1], p_curr[1]], [p_prev[2], p_curr[2]], 
            color='black', linestyle='--', linewidth=2, alpha=0.8)

    # --- PLOT 4: REACHABLE VERTICES (Red Dots) ---
    ax.scatter(next_vertices[:, 0], next_vertices[:, 1], next_vertices[:, 2], 
               color='red', s=30, alpha=0.8, label='Reachable Vertices (15°)', zorder=10)
    
    # --- PLOT 5: CONVEX HULL (Transparent Bubble) ---
    for s in hull.simplices:
        ax.plot_trisurf(next_vertices[s, 0], next_vertices[s, 1], next_vertices[s, 2], 
                        alpha=0.1, color='cyan', edgecolor='blue', linewidth=0.5)

    # Adjusting view for visibility
    ax.set_xlabel('Global X (m)')
    ax.set_ylabel('Global Y (m)')
    ax.set_zlabel('Global Z (m)')
    ax.set_title("UAV State Transition and Next Step Reachable Set")
    ax.legend(loc='upper left')

    # Ensure all points are in frame
    all_points = np.vstack([p_prev, p_curr, next_vertices])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2.0
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.show()

# Example: Sharp movement to make the transition clear
p_now = np.array([0, 0, 10])     # Starting at origin
v_cur = np.array([0.6, 0.3, 0]) # Big jump to make states visible
plot_full_uav_sequence(v_cur, p_now, dt=0.2, u1_m=20.0)