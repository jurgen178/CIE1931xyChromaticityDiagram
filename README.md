### CIE 1931 xy Chromaticity Diagram

This project draws a **CIE 1931 xy Chromaticity Diagram** using a Python script with the following:

- **1 nm edge sampling** (interpolated from 5 nm color matching functions)
- **Physically accurate color fill**  
  Conversion pipeline: `xyY → XYZ → sRGB (D65)` followed by global dimming for display
- **Visual elements included**:  
  - sRGB and Adobe RGB gamut triangles  
  - D65 white point marker  


<img width="2000" height="2000" alt="CIE1931_xy" src="https://github.com/user-attachments/assets/ff00d349-c196-47e1-a3e3-75c274555647" />

See this [blog post](https://www.bitfabrik.io/blog/index.php?id_post=258)
