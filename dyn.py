import sympy as sp
sp.init_printing(use_unicode=True)
# =============================================================================
# 0. SETUP: Symbols and Constants
# =============================================================================
theta1, theta2 = sp.symbols('theta_1 theta_2', real=True)
theta1d, theta2d = sp.symbols('thetadot_1 thetadot_2', real=True)
theta1dd, theta2dd = sp.symbols('thetaddot_1 thetaddot_2', real=True)  
sp.pprint(theta1dd)


l1, l2 = sp.symbols('l1 l2', positive=True)
m1, m2 = sp.symbols('m1 m2', positive=True)
g = sp.symbols('g', real=True)

# Unit z-axis in any frame
Z_hat = sp.Matrix([0, 0, 1])

# =============================================================================
# 1. ROTATION MATRICES 
# =============================================================================
# ^0_1R
R_01 = sp.Matrix([
    [sp.cos(theta1), -sp.sin(theta1), 0],
    [sp.sin(theta1),  sp.cos(theta1), 0],
    [0,               0,              1]
])

# ^1_2R
R_12 = sp.Matrix([
    [sp.cos(theta2), -sp.sin(theta2), 0],
    [sp.sin(theta2),  sp.cos(theta2), 0],
    [0,               0,              1]
])

# Inverse rotations (transpose)
R_10 = R_01.T   
R_21 = R_12.T   

# =============================================================================
# 2. INITIAL CONDITIONS (Base Frame 0)
# =============================================================================
w0 = sp.Matrix([0, 0, 0])        # ^0ω₀
wd0 = sp.Matrix([0, 0, 0])       # ^0ω̇₀
vd0 = sp.Matrix([0, g, 0])       # ^0v̇₀ = g * Ŷ₀ (as per your slide)

# =============================================================================
# 3. OUTWARD ITERATIONS
# =============================================================================

print("=== OUTWARD ITERATIONS ===\n")

# ----------------------------
# Link 1 (i = 0 → 1)
# ----------------------------
print(" Link 1 (0 -> 1):")

w1 = R_10 * w0 + theta1d * Z_hat
print("  w(11) =")
sp.pprint(w1)


wd1 = R_10 * wd0 + R_10 * (w0.cross(theta1d * Z_hat)) + theta1dd * Z_hat
print("\n  w_dot(11) =")
sp.pprint(wd1)


v1d = R_10 * vd0
print("\n  v_dot(11) =")
sp.pprint(v1d)

P_C1_1 = sp.Matrix([l1, 0, 0])

vC1d_1 = wd1.cross(P_C1_1) + w1.cross(w1.cross(P_C1_1)) + v1d
print("\n  v_dot(1C1) =")
sp.pprint(vC1d_1)

F1 = m1 * vC1d_1
N1 = sp.zeros(3, 1)  # point mass ⇒ I = 0
print("\n  F(11) =")
sp.pprint(F1)
print("\n  N(11) =")
sp.pprint(N1)


P2_1 = sp.Matrix([l1, 0, 0])


# ----------------------------
# Link 2 (1 -> 2)
# ----------------------------
print("\n\n→ Link 2 (i=1 → 2):")


w2 = R_21 * w1 + theta2d * Z_hat
print("  w(22) =")
sp.pprint(sp.simplify(w2))


wd2 = R_21 * wd1 + R_21 * (w1.cross(theta2d * Z_hat)) + theta2dd * Z_hat
print("\n  w_dot(22) =")
sp.pprint(sp.simplify(wd2))


v2d = R_21 * v1d + R_21 * (w1.cross(P2_1))
print("\n  v_dot(2) =")
sp.pprint(sp.simplify(v2d))


P_C2_2 = sp.Matrix([l2, 0, 0])
vC2d_2 = wd2.cross(P_C2_2) + w2.cross(w2.cross(P_C2_2)) + v2d
print("\n  v_dot2C2 =")
sp.pprint(sp.simplify(vC2d_2))


F2 = m2 * vC2d_2
N2 = sp.zeros(3, 1)
print("\n  F(22) =")
sp.pprint(sp.simplify(F2))
print("\n  N(22) =")
sp.pprint(sp.simplify(N2))

# =============================================================================
# 4. INWARD ITERATIONS 
# =============================================================================
print("\n\n=== INWARD ITERATIONS ===\n")


f3 = sp.zeros(3, 1)
n3 = sp.zeros(3, 1)

# ----------------------------
# Step  2 -> 1
# ----------------------------
f2 = f3 + F2
print("  f(22) =")
sp.pprint(f2)

n2 = N2 + n3 + P_C2_2.cross(f2)
print("\n  n(22) =")
sp.pprint(sp.simplify(n2))

f2_in_1 = R_12 * f2         
n2_in_1 = R_12 * n2 + P2_1.cross(f2_in_1)  

print("\n  Transformed to frame 1:")
print("  f(12) =")
sp.pprint(sp.simplify(f2_in_1))
print("  n(12) =")
sp.pprint(sp.simplify(n2_in_1))


# ----------------------------
# Step  1 -> 0
# ----------------------------
f1 = f2_in_1 + F1
print("  f(11) =")
sp.pprint(sp.simplify(f1))

n1 = N1 + n2_in_1 + P_C1_1.cross(F1) + P2_1.cross(f2_in_1)
print("\n  n(11) =")
sp.pprint(sp.simplify(n1))

# =============================================================================
# 5. JOINT TORQUES
# =============================================================================

tau2 = n2[2] 
tau1 = n1[2]

print("\n\n=== FINAL JOINT TORQUES ===")
print("tau_1 =")
sp.pprint(sp.simplify(tau1))
print("\n tau_2 =")
sp.pprint(sp.simplify(tau2))


