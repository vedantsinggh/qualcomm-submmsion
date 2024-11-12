# Reflection Removal Tool

This Python script removes reflections from an input image using the algorithm described in the paper "Fast Single Image Reflection Suppression via Convex Optimization".

## Dependencies

The script requires the following Python libraries:

- `numpy`
- `matplotlib.pyplot`
- `scipy.fftpack`
- `PIL`

You can install these dependencies using `pip`:

```
pip install numpy matplotlib scipy pillow
```

## Usage

1. Place the input image (e.g., `input_image.jpg`) in the same directory as the `remove.py` script.

2. Run the script:

   ```
   python remove.py
   ```

   This will create an `output` directory (if it doesn't already exist) and save the reflection-removed image as `output_image.jpg`.

3. Adjust the algorithm parameters if needed:

   - `h`: The higher the value, the more reflections will be removed, but the quality of the image may suffer. Recommended values are between 0 and 0.13.
   - `lmbd`: Recommended value is 0.
   - `mu`: Recommended value is 1.
   - `epsilon`: A small value (e.g., 1e-8) to avoid division by 0.

   You can modify these parameters by changing the arguments passed to the `remove_reflection` function:

   ```python
   output_image = remove_reflection(img, h=0.1, lmbd=0, mu=1, epsilon=1e-8, output_dir="output")
   ```

## How it Works

The script uses the following steps to remove reflections from the input image:

1. Compute the Laplacian of the input image using the `laplacian` function.
2. Compute the right-hand side of the equation 7 from the paper using the `_compute_rhs` function.
3. Compute the T matrix (the original matrix with reflection suppressed) using the `_compute_T` function.
4. Scale the T matrix to the range [0, 1] using the `min_max_scale` function.
5. Save the output image to the `output` directory.

The intermediate debug images are also saved in the `output` directory for analysis purposes.

## License

This project is released under the MIT License.
Paper implemented https://arxiv.org/pdf/1903.03889 
