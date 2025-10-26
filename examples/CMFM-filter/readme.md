# Current-Mode Multiple-Feedback (CMFM) Filter Example

This example demonstrates the symbolic analysis and automated sizing of a Current-Mode Multiple-Feedback (CMFM) biquadratic filter.

## Circuit Overview

CMFM filters are a class of active filters that use current as the signal variable. They are known for their potential for high-frequency operation and wide dynamic range. This example focuses on a biquad topology, which can be configured to realize low-pass, high-pass, or band-pass filter responses.

## Directory Structure

-   **/sym-exploration/**: Contains a rich set of Jupyter notebooks for the symbolic analysis of the CMFM filter. These notebooks use `SymXplorer` to derive and analyze the filter's transfer function, including exploring the impact of non-idealities like the finite Gain-Bandwidth product of the amplifiers.
-   **/sizing/**: This directory showcases multiple approaches to sizing the filter to meet specific performance targets.
    -   **/sizing/ltspice-sizing/**: Demonstrates a sizing methodology using simulations in LTspice.
    -   **/sizing/sym-sizing-bode-ax/**: Uses a combination of symbolic modeling and the `Ax` (Bayesian) optimizer to size the filter based on its Bode plot response.
    -   **/sizing/sym-sizing-bode-nevergrad/**: Provides an alternative sizing approach using the `Nevergrad` (Evolutionary) optimizer.
