# CIE 1931 xy Chromaticity Diagram with smooth spectral locus and dimmed interior
# - 1 nm edge sampling (interpolated from 5 nm CMFs)
# - physically correct color fill (xyY -> XYZ -> sRGB D65), then global dimming
# - sRGB and Adobe RGB triangles, D65 marker, wavelength labels
#
# Sources:
#   CVRL/CIE data context for CIE 1931 CMFs and spectral locus:
#     https://www.cie.co.at/datatable/cie-1931-chromaticity-coordinates-spectrum-loci-2-degree-observer
#     http://www.cvrl.org/database/text/cmfs/ciexyz31.htm
#   sRGB <-> XYZ matrices (D65):
#     http://brucelindbloom.com/Eqn_RGB_XYZ_Matrix.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patheffects as pe

# ----------------------------------------------------------------------
# Input parameters
# ----------------------------------------------------------------------
OUTPUT_WIDTH = 2000             # final output image width in pixels
OUTPUT_HEIGHT = 2000            # final output image height in pixels
OUTPUT_DPI = 300                # output DPI for saved image
LABEL_FONTSIZE = 15             # font size for main labels (triangles, D65)
AXIS_FONTSIZE = 14              # font size for axis labels (x, y)

# Calculate figure size in inches based on desired pixel dimensions
FIG_WIDTH = OUTPUT_WIDTH / OUTPUT_DPI
FIG_HEIGHT = OUTPUT_HEIGHT / OUTPUT_DPI

IMG_W, IMG_H = OUTPUT_WIDTH, OUTPUT_HEIGHT  # internal image resolution matches output
GAIN = 0.95                    # dim the interior fill (0.7 darker ... 1.0 original)
LABEL_STEP = 20                # wavelength label step in nm
SAVE_AS = "CIE1931_xy.png"

# Plot extent
XMIN, XMAX = 0.0, 0.8
YMIN, YMAX = 0.0, 0.9

# ----------------------------------------------------------------------
# CIE 1931 2-deg CMFs at 5 nm steps (wavelength, X_bar, Y_bar, Z_bar)
# These values mirror standard published CMFs in compact 5 nm form.
# Interpolate to 1 nm for a smoother spectral locus.
# ----------------------------------------------------------------------
cmf_rows = [
    (380, 0.001368000000, 0.000039000000, 0.006450001000),
    (385, 0.002236000000, 0.000064000000, 0.010549990000),
    (390, 0.004243000000, 0.000120000000, 0.020050010000),
    (395, 0.007650000000, 0.000217000000, 0.036210000000),
    (400, 0.014310000000, 0.000396000000, 0.067850010000),
    (405, 0.023190000000, 0.000640000000, 0.110200000000),
    (410, 0.043510000000, 0.001210000000, 0.207400000000),
    (415, 0.077630000000, 0.002180000000, 0.371300000000),
    (420, 0.134380000000, 0.004000000000, 0.645600000000),
    (425, 0.214770000000, 0.007300000000, 1.039050100000),
    (430, 0.283900000000, 0.011600000000, 1.385600000000),
    (435, 0.328500000000, 0.016840000000, 1.622960000000),
    (440, 0.348280000000, 0.023000000000, 1.747060000000),
    (445, 0.348060000000, 0.029800000000, 1.782600000000),
    (450, 0.336200000000, 0.038000000000, 1.772110000000),
    (455, 0.318700000000, 0.048000000000, 1.744100000000),
    (460, 0.290800000000, 0.060000000000, 1.669200000000),
    (465, 0.251100000000, 0.073900000000, 1.528100000000),
    (470, 0.195360000000, 0.090980000000, 1.287640000000),
    (475, 0.142100000000, 0.112600000000, 1.041900000000),
    (480, 0.095640000000, 0.139020000000, 0.812950100000),
    (485, 0.057950010000, 0.169300000000, 0.616200000000),
    (490, 0.032010000000, 0.208020000000, 0.465180000000),
    (495, 0.014700000000, 0.258600000000, 0.353300000000),
    (500, 0.004900000000, 0.323000000000, 0.272000000000),
    (505, 0.002400000000, 0.407300000000, 0.212300000000),
    (510, 0.009300000000, 0.503000000000, 0.158200000000),
    (515, 0.029100000000, 0.608200000000, 0.111700000000),
    (520, 0.063270000000, 0.710000000000, 0.078249990000),
    (525, 0.109600000000, 0.793200000000, 0.057250010000),
    (530, 0.165500000000, 0.862000000000, 0.042160000000),
    (535, 0.225749900000, 0.914850100000, 0.029840000000),
    (540, 0.290400000000, 0.954000000000, 0.020300000000),
    (545, 0.359700000000, 0.980300000000, 0.013400000000),
    (550, 0.433449900000, 0.994950100000, 0.008749999000),
    (555, 0.512050100000, 1.000000000000, 0.005749999000),
    (560, 0.594500000000, 0.995000000000, 0.003900000000),
    (565, 0.678400000000, 0.978600000000, 0.002749999000),
    (570, 0.762100000000, 0.952000000000, 0.002100000000),
    (575, 0.842500000000, 0.915400000000, 0.001800000000),
    (580, 0.916300000000, 0.870000000000, 0.001650001000),
    (585, 0.978600000000, 0.816300000000, 0.001400000000),
    (590, 1.026300000000, 0.757000000000, 0.001100000000),
    (595, 1.056700000000, 0.694900000000, 0.001000000000),
    (600, 1.062200000000, 0.631000000000, 0.000800000000),
    (605, 1.045600000000, 0.566800000000, 0.000600000000),
    (610, 1.002600000000, 0.503000000000, 0.000340000000),
    (615, 0.938400000000, 0.441200000000, 0.000240000000),
    (620, 0.854449900000, 0.381000000000, 0.000190000000),
    (625, 0.751400000000, 0.321000000000, 0.000100000000),
    (630, 0.642400000000, 0.265000000000, 0.000049999990),
    (635, 0.541900000000, 0.217000000000, 0.000030000000),
    (640, 0.447900000000, 0.175000000000, 0.000020000000),
    (645, 0.360800000000, 0.138200000000, 0.000010000000),
    (650, 0.283500000000, 0.107000000000, 0.000000000000),
    (655, 0.218700000000, 0.081600000000, 0.000000000000),
    (660, 0.164900000000, 0.061000000000, 0.000000000000),
    (665, 0.121200000000, 0.044580000000, 0.000000000000),
    (670, 0.087400000000, 0.032000000000, 0.000000000000),
    (675, 0.063600000000, 0.023200000000, 0.000000000000),
    (680, 0.046770000000, 0.017000000000, 0.000000000000),
    (685, 0.032900000000, 0.011920000000, 0.000000000000),
    (690, 0.022700000000, 0.008210000000, 0.000000000000),
    (695, 0.015840000000, 0.005723000000, 0.000000000000),
    (700, 0.011359160000, 0.004102000000, 0.000000000000),
    (705, 0.008110916000, 0.002929000000, 0.000000000000),
    (710, 0.005790346000, 0.002091000000, 0.000000000000),
    (715, 0.004109457000, 0.001484000000, 0.000000000000),
    (720, 0.002899327000, 0.001047000000, 0.000000000000),
    (725, 0.002049190000, 0.000740000000, 0.000000000000),
    (730, 0.001439971000, 0.000520000000, 0.000000000000),
    (735, 0.000999949300, 0.000361100000, 0.000000000000),
    (740, 0.000690078600, 0.000249200000, 0.000000000000),
    (745, 0.000476021300, 0.000171900000, 0.000000000000),
    (750, 0.000332301100, 0.000120000000, 0.000000000000),
    (755, 0.000234826100, 0.000084800000, 0.000000000000),
    (760, 0.000166150500, 0.000060000000, 0.000000000000),
    (765, 0.000117413000, 0.000042400000, 0.000000000000),
    (770, 0.000083075270, 0.000030000000, 0.000000000000),
    (775, 0.000058706520, 0.000021200000, 0.000000000000),
    (780, 0.000041509940, 0.000014990000, 0.000000000000),
]

# Interpolate CMFs to 1 nm for a smooth spectral locus using polynomial interpolation
wl_5 = np.array([r[0] for r in cmf_rows])
Xbar_5 = np.array([r[1] for r in cmf_rows])
Ybar_5 = np.array([r[2] for r in cmf_rows])
Zbar_5 = np.array([r[3] for r in cmf_rows])

wl = np.arange(380, 781, 1)

# Use piecewise polynomial interpolation (spline-like) for smoother curves
# Split into segments and use cubic polynomial interpolation
def smooth_interpolate(x_orig, y_orig, x_new):
    """Piecewise cubic polynomial interpolation for smoother curves"""
    result = np.zeros_like(x_new, dtype=float)
    
    for i in range(len(x_new)):
        x_val = x_new[i]
        
        # Find the closest 4 points around x_val for cubic interpolation
        idx = np.searchsorted(x_orig, x_val)
        
        # Ensure we have enough points for cubic interpolation
        start_idx = max(0, min(idx - 2, len(x_orig) - 4))
        end_idx = start_idx + 4
        
        if end_idx > len(x_orig):
            end_idx = len(x_orig)
            start_idx = max(0, end_idx - 4)
        
        # Extract local points
        x_local = x_orig[start_idx:end_idx]
        y_local = y_orig[start_idx:end_idx]
        
        # Fit polynomial and evaluate
        if len(x_local) >= 2:
            poly_coeffs = np.polyfit(x_local, y_local, min(3, len(x_local)-1))
            result[i] = np.polyval(poly_coeffs, x_val)
        else:
            # Fallback to linear interpolation if not enough points
            result[i] = np.interp(x_val, x_orig, y_orig)
    
    return result

Xbar = smooth_interpolate(wl_5, Xbar_5, wl)
Ybar = smooth_interpolate(wl_5, Ybar_5, wl)
Zbar = smooth_interpolate(wl_5, Zbar_5, wl)

XYZ_sum = Xbar + Ybar + Zbar
x_locus = Xbar / XYZ_sum
y_locus = Ybar / XYZ_sum
poly = np.stack([x_locus, y_locus], axis=1)

# Background grid in xy
x = np.linspace(XMIN, XMAX, IMG_W)
y = np.linspace(YMIN, YMAX, IMG_H)
xx, yy = np.meshgrid(x, y)

valid = (xx >= 0) & (yy >= 0) & (xx + yy <= 1) & (yy > 0)

# Convert xyY (with Y=1) to XYZ
X = np.zeros_like(xx)
Y = np.zeros_like(yy)
Z = np.zeros_like(xx)

X[valid] = xx[valid] / yy[valid]
Y[valid] = 1.0
Z[valid] = (1.0 - xx[valid] - yy[valid]) / yy[valid]

# XYZ to linear sRGB (D65) - standard inverse matrix
M_inv = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
])

XYZ = np.stack([X, Y, Z], axis=-1)
RGB_lin = np.tensordot(XYZ, M_inv.T, axes=1)

# sRGB companding
a = 0.055
threshold = 0.0031308
RGB = np.where(RGB_lin <= threshold, 12.92*RGB_lin,
               (1+a)*np.power(np.maximum(RGB_lin, 0), 1/2.4) - a)
RGB = np.clip(RGB, 0, 1)

# Dim interior uniformly to avoid overly bright look
RGB *= GAIN

# Mask to the spectral locus polygon (including line of purples)
path = Path(poly)
pts = np.column_stack([xx.ravel(), yy.ravel()])
inside = path.contains_points(pts).reshape(xx.shape)

RGBA = np.ones((IMG_H, IMG_W, 4), dtype=float)
RGBA[..., :3] = RGB
RGBA[..., 3] = inside.astype(float)

# Plot
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=OUTPUT_DPI)
ax.imshow(RGBA, extent=[XMIN, XMAX, YMIN, YMAX], origin="lower", interpolation="bilinear")

# Draw spectral locus boundary and purple line
ax.plot(poly[:,0], poly[:,1], color="k", linewidth=1.1)
ax.plot([poly[-1,0], poly[0,0]], [poly[-1,1], poly[0,1]], color="k", linewidth=1.0)

# sRGB and Adobe RGB triangles
sRGB_tri = np.array([[0.6400, 0.3300], [0.3000, 0.6000], [0.1500, 0.0600], [0.6400, 0.3300]])
adobe_tri = np.array([[0.6400, 0.3300], [0.2100, 0.7100], [0.1500, 0.0600], [0.6400, 0.3300]])
ax.plot(sRGB_tri[:,0], sRGB_tri[:,1], color="white", linewidth=2.0, linestyle="--", alpha=0.95)
ax.plot(adobe_tri[:,0], adobe_tri[:,1], color="#ffd400", linewidth=2.0, alpha=0.95)

# Add labels for the triangles
# sRGB label
ax.text(0.3, 0.42, "sRGB", fontsize=LABEL_FONTSIZE, color="black", 
        ha="center", va="center")

# Adobe RGB label 
ax.text(0.11, 0.672, "Adobe RGB", fontsize=LABEL_FONTSIZE, color="black", 
        ha="center", va="center")

# D65 marker
xD65, yD65 = 0.3127, 0.3290
ax.scatter([xD65], [yD65], s=24, facecolor="white", edgecolor="black", zorder=5)

# D65 label
ax.text(xD65 + 0.015, yD65 - 0.018, "D65", fontsize=LABEL_FONTSIZE, color="black", 
        ha="left", va="bottom")

# Wavelength labels
labels = list(range(380, 701, LABEL_STEP))
x_lbl = np.interp(labels, wl, x_locus)
y_lbl = np.interp(labels, wl, y_locus)

# simple outward offsets tuned by region
offsets = {l:(0.006, 0.004) for l in labels}
offsets.update({
    380:(-0.010,  0.010), 400:(-0.008, 0.012), 420:(-0.010, 0.012), 440:(-0.008, 0.010),
    460:(-0.006,  0.008), 480:(-0.004, 0.006), 500:(-0.004, 0.006), 520:( 0.006, 0.004),
    540:( 0.008,  0.002), 560:( 0.010, 0.000), 580:( 0.008,-0.006), 600:( 0.006,-0.010),
    620:( 0.004, -0.012), 640:( 0.002,-0.012), 660:( 0.000,-0.012), 680:(-0.002,-0.012),
    700:(-0.006, -0.010)
})

for wl_i, xi, yi in zip(labels, x_lbl, y_lbl):
    dx, dy = offsets.get(wl_i, (0.006, 0.006))
    ax.plot([xi, xi+dx*0.6], [yi, yi+dy*0.6], color="k", linewidth=0.8)
    txt = ax.text(xi+dx, yi+dy, f"{wl_i} nm", fontsize=8.2, color="k", ha="center", va="center")
    txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

# Axes formatting
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.set_xlabel("x", fontsize=AXIS_FONTSIZE, weight="bold")
ax.set_ylabel("y", rotation=0, fontsize=AXIS_FONTSIZE, weight="bold")
ax.set_title("CIE 1931 xy Chromaticity Diagram")
ax.grid(True, color="white", alpha=0.15, linestyle=":")

ax.set_xticks(np.arange(0, 0.81, 0.1))
ax.set_yticks(np.arange(0, 0.91, 0.1))

plt.tight_layout()
plt.savefig(SAVE_AS, dpi=OUTPUT_DPI)

print
