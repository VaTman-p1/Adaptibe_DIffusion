import numpy as np

def perform_grid_search(U1_m):
    # Create a grid from 0 to 15 degrees
    angles = np.arange(0, 16, 1) # [0, 1, 2, ..., 15]
    
    max_ax = -np.inf
    max_ay = -np.inf
    best_config_x = (0, 0)
    best_config_y = (0, 0)

    for theta_deg in angles:
        for phi_deg in angles:
            # Convert to radians for math functions
            theta = np.radians(theta_deg)
            phi = np.radians(phi_deg)
            
            # Equations from your provided image
            # Assumes psi = 0
            ax = (np.sin(theta) * np.cos(phi)) * U1_m
            ay = (-np.sin(phi)) * U1_m
            
            # Update X maximum
            if abs(ax) > max_ax:
                max_ax = abs(ax)
                best_config_x = (theta_deg, phi_deg)
                
            # Update Y maximum
            if abs(ay) > max_ay:
                max_ay = abs(ay)
                best_config_y = (theta_deg, phi_deg)

    return {
        "max_ax": max_ax, "best_theta_x": best_config_x[0], "best_phi_x": best_config_x[1],
        "max_ay": max_ay, "best_theta_y": best_config_y[0], "best_phi_y": best_config_y[1]
    }

# Example: Assume Thrust-to-mass (U1/m) is 1.2 * gravity (11.77)
thrust_param = 11.77 
results = perform_grid_search(thrust_param)

print(f"Max ax: {results['max_ax']:.3f} at Pitch={results['best_theta_x']}°, Roll={results['best_phi_x']}°")
print(f"Max ay: {results['max_ay']:.3f} at Pitch={results['best_theta_y']}°, Roll={results['best_phi_y']}°")