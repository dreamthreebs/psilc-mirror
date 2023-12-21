import numpy as np

def normalize_vector(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def calculate_angle_arccos(vec1, vec2):
    """Calculate angle using arccos of dot product."""
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # Clipping values to avoid numerical issues outside the domain of arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.arccos(cosine_angle)

def calculate_angle_arctan2(vec1, vec2):
    """Calculate angle using arctan2 of cross product and dot product."""
    cross_prod = np.cross(vec1, vec2)
    dot_prod = np.dot(vec1, vec2)
    return np.arctan2(np.linalg.norm(cross_prod), dot_prod)

# Nearly parallel vectors
vec_a = np.array([1.0, 0.0, 0.0])
vec_b = np.array([1.0, 1e-20, 0.0])  # Slight perturbation

# Nearly antiparallel vectors
vec_c = np.array([1.0, 0.0, 0.0])
vec_d = np.array([-1.0, 1e-20, 0.0])  # Slight perturbation

# Normalizing the vectors
vec_a_norm = normalize_vector(vec_a)
vec_b_norm = normalize_vector(vec_b)
vec_c_norm = normalize_vector(vec_c)
vec_d_norm = normalize_vector(vec_d)

# Recalculating angles with normalized vectors
angle_parallel_arccos_norm = calculate_angle_arccos(vec_a_norm, vec_b_norm)
angle_parallel_arctan2_norm = calculate_angle_arctan2(vec_a_norm, vec_b_norm)

angle_antiparallel_arccos_norm = calculate_angle_arccos(vec_c_norm, vec_d_norm)
angle_antiparallel_arctan2_norm = calculate_angle_arctan2(vec_c_norm, vec_d_norm)

# Output the results
print((angle_parallel_arccos_norm, angle_parallel_arctan2_norm, 
 angle_antiparallel_arccos_norm, angle_antiparallel_arctan2_norm))


'''
    np.arctan2 are more stable than np.arccos when calculating the angular distance between two vector on the sphere.
    the np.arccos lose effectiveness when perturbation is smaller than around 1e-8
'''
