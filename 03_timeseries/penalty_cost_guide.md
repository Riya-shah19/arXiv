# Methodological Guide: PELT Changepoint Penalty & Cost Selection

When presenting your time series segmentation to your professor, you can use this document to explain the mathematical and statistical rationale behind the cost function, the penalty parameter, and how to select them.

---

## 1. What We Are Doing

We are modifying the changepoint detection pipeline to expose the **raw segmentation cost** (sum of segment approximation errors) alongside the detected changepoint dates. 

Specifically:
1. **Adding Cost Calculation**: In `timeseries_utils.py`, the `detect_changepoints` function will calculate and return the raw sum of costs $\sum_{i=1}^{k} \mathcal{C}(y_{\tau_{i-1}:\tau_i})$ for the detected segments.
2. **Backwards-Compatible Integration**: We use a tuple subclass `DetectionResult` so that existing scripts can unpack `cp_dates, decomp` without breaking, while allowing new code to access `.cost`.
3. **Exposing Logs**: We are updating individual scripts (e.g. `timeseries_cs.py`) to print the numeric cost of the fit.

---

## 2. Why We Are Doing It (Academic Justification)

### The Optimization Problem
The PELT (Pruned Exact Linear Time) algorithm detects structural changes by solving an optimization problem that minimizes a penalized cost function over a signal $y$ of length $n$:

$$\min_{k, \tau_{1..k}} \sum_{i=1}^{k} \mathcal{C}(y_{\tau_{i-1}:\tau_i}) + \beta \cdot k$$

Where:
* **$k$** is the number of changepoints (model complexity).
* **$\tau_{1..k}$** are the changepoint indices.
* **$\mathcal{C}$** is the cost function measuring the approximation error in each segment (goodness-of-fit).
* **$\beta$** is the **Penalty** parameter (`pen` in the code) which penalizes model complexity to prevent overfitting.

### The Trade-off: Underfitting vs. Overfitting
* **If $\beta$ is too small (low penalty)**: The algorithm will overfit, creating too many changepoints. The raw cost will be very low (near 0), but the model is too complex and captures noise.
* **If $\beta$ is too large (high penalty)**: The algorithm will underfit, creating too few changepoints. The raw cost will be very high because the model fails to capture real structural breaks.

---

## 3. How the Cost Value Helps Pick the Penalty (The Elbow Method)

By measuring the **raw cost** for a range of penalties (e.g., $3, 5, 8, 10, 12, 15, 20$), we can perform an **Elbow Analysis** (similar to choosing $K$ in $K$-means clustering):

1. Plot the **Number of Changepoints** (or Penalty) on the X-axis against the **Raw Cost** on the Y-axis.
2. As the number of changepoints increases, the cost will decrease.
3. Locate the **"elbow"**—the point where adding more changepoints results in diminishing returns (i.e., the cost curve flattens out).
4. The penalty that corresponds to this elbow is the mathematically optimal choice that balances goodness-of-fit with model simplicity.

---

## 4. Comparing Cost Models ($RBF$ vs. $L1$ vs. $L2$)

Different cost functions ($\mathcal{C}$) assume different characteristics about the underlying data:
* **$L2$ (Least Squares / Mean-shift)**: Assumes segment errors are normally distributed and measures squared deviations from the mean. It is highly sensitive to mean shifts but also sensitive to outliers.
* **$L1$ (Manhattan / Median-shift)**: Measures absolute deviations from the median. It is robust to outliers and transient spikes.
* **$RBF$ (Radial Basis Function)**: A non-parametric kernel cost function. It makes no assumptions about the distribution shape and detects changes in the overall distribution (mean, variance, and shape).
