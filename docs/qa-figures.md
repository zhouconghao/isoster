# QA Figure Rules

Guidelines for making QA figures to compare `isoster` results with truth or `photutils.isophote` results.

## Layout

- Show the original image with a few selective isophotes on top.
- If possible, reconstruct the 2-D model and subtract it from the image to highlight residual patterns.
- Arrange all sub-plots for 1-D information (surface brightness, residual/difference, position angle, axis ratio, centroid) vertically, sharing the same X-axis to save space.
- The 1-D surface brightness profile should occupy a larger area than the other panels.

## X-Axis Convention

- Use `SMA ** 0.25` as the X-axis for 1-D profiles. This compresses the outer profile (typically a shallow slope) while not zooming into the center too much.

## Plotting Style

- Use scatter plots with errorbars (`plt.scatter()`) for measured profiles, not lines (`plt.plot()`).
- Use dashed lines for true/reference model profiles.
- When comparing with truth or different methods, use the relative 1-D residual: `(intensity_isoster - intensity_reference) / intensity_reference`.

## Axis Ranges

- Intensity, position angle, axis ratio, and centroid results in the outskirts can have huge errorbars. When setting Y-axis ranges, do not include the error bars.

## Position Angle Normalization

- Normalize the position angle: sudden jumps larger than 90 degrees often indicate a normalization issue.

## Stop Code Visualization

- For `isoster` or `photutils.isophote` results, visually separate valid and problematic 1-D data points using the stop code.

## Default Style Baseline

- Treat the finalized IC2597 Huang2013 basic-QA style (`build_method_qa_figure` / `build_comparison_qa_figure`) as the default style baseline for future QA figures unless a task explicitly requests a different style.
