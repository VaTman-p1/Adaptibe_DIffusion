import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_full_uav_sequence(v_curr, p_curr, dt, u1_m, psi=0, tilt_limit_deg=15.0):
    g = 9.81
    limit_rad = np.radians(tilt_limit_deg)
    p_prev = p_curr - v_curr * dt
    
    # 1. Define sampling steps for angles (-15, -10, -5, 0, 5, 10, 15)
    angle_steps = np.radians(np.arange(-tilt_limit_deg, tilt_limit_deg + 5, 30))
    print(angle_steps)
    # Using more thrust levels to ensure the volume isn't perfectly flat
    t_limits = [0.2*u1_m, 1.5*u1_m]
    
    next_vertices = []
    for T in t_limits:
        for phi in angle_steps:
            for theta in angle_steps:
                ax = (np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi)) * T
                ay = (-np.cos(psi)*np.sin(phi) + np.sin(psi)*np.sin(theta)*np.cos(phi)) * T
                az = -g + (np.cos(theta)*np.cos(phi)) * T
                
                accel = np.array([ax, ay, az])
                p_next = p_curr + v_curr * dt + 0.5 * accel * (dt**2)
                next_vertices.append(p_next)
    
    next_vertices = np.array(next_vertices)
    
    # 2. Compute Convex Hull
    hull = ConvexHull(next_vertices)
    
    # 3. Plotting Setup
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- PLOT 1: PREVIOUS STATE ---
    ax.scatter(p_prev[0], p_prev[1], p_prev[2], 
               color='black', s=150, marker='X', label='Previous Position', zorder=20)
    
    # --- PLOT 2: CURRENT STATE ---
    ax.scatter(p_curr[0], p_curr[1], p_curr[2], 
               color='blue', s=200, marker='D', label='Current Position', zorder=25)
    
    # --- PLOT 3: TRAJECTORY LINE ---
    ax.plot([p_prev[0], p_curr[0]], [p_prev[1], p_curr[1]], [p_prev[2], p_curr[2]], 
            color='black', linestyle='--', linewidth=2, alpha=0.8)

    # --- PLOT 4: REACHABLE POINTS (Red Dots) ---
    ax.scatter(next_vertices[:, 0], next_vertices[:, 1], next_vertices[:, 2], 
               color='red', s=10, alpha=0.3, label='Sampled Points', zorder=1)
    
    # --- PLOT 5: CONVEX HULL (The fix for Trisurf error) ---
    # Create the polygons for the hull faces
    for simplex in hull.simplices:
        # Get the points for this face
        v = next_vertices[simplex]
        # Create a 3D polygon
        tri = Poly3DCollection([v])
        tri.set_alpha(0.15)
        tri.set_facecolor('cyan')
        tri.set_edgecolor('blue')
        ax.add_collection3d(tri)

    # Formatting
    ax.set_xlabel('Global X (m)')
    ax.set_ylabel('Global Y (m)')
    ax.set_zlabel('Global Z (m)')
    ax.set_title(f"Reachable Set (5° steps, {tilt_limit_deg}° limit)")
    ax.legend(loc='upper left')

    # Zoom logic
    all_points = np.vstack([p_prev, p_curr, next_vertices])
    mins, maxs = all_points.min(axis=0), all_points.max(axis=0)
    center = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 1.8 # Slightly tighter zoom
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.show()

# Run
p_now = np.array([0, 0, 10])
v_cur = np.array([0.6, 0.3, 0.2])
plot_full_uav_sequence(v_cur, p_now, dt=0.2, u1_m=9.81)