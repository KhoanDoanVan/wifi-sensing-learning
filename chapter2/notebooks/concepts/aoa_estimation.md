# AoA (Angle of Arrival) Concepts

## 1. AoA (Angle of Arrival)

AoA is the estimated direction from which a wireless signal arrives at an antenna array.

Main idea:

* A radio wave reaches different antennas at slightly different times.
* This creates phase differences between antennas.
* From phase differences + antenna geometry, we estimate the incoming direction.

---

# 2. CSI (Channel State Information)

CSI is the complex channel response between transmitter and receiver.

Typical form:

[
H = A e^{j\phi}
]

Where:

* (A): amplitude
* (\phi): phase

AoA estimation mainly uses CSI phase.

---

# 3. Complex Signal

Wireless baseband signals are represented as complex numbers:

[
x = I + jQ
]

Where:

* (I): in-phase component
* (Q): quadrature component

Complex representation allows:

* phase analysis
* frequency shifting
* modulation
* channel estimation

---

# 4. Phase

Phase represents the angle of a complex number.

[
\phi = \angle x
]

Extracted using:

```python
np.angle(x)
```

Phase is the key information for AoA.

---

# 5. Phase Difference

AoA relies on phase differences between antennas:

[
\Delta \phi = \phi_2 - \phi_1
]

Because the wave arrives at different antennas with different propagation delays.

---

# 6. Antenna Array

Multiple antennas arranged in space.

Examples:

* Linear array (ULA)
* Planar array (UPA)
* Circular array

Each antenna has a known 3D position:

[
[x, y, z]
]

---

# 7. Antenna Baseline Vector

Vector between two antennas:

[
\vec r = \vec p_2 - \vec p_1
]

This determines how much phase difference a wave produces.

---

# 8. Wavelength

[
\lambda = \frac{c}{f}
]

Where:

* (c): speed of light
* (f): carrier frequency

Example:

* WiFi 5 GHz:
  [
  \lambda \approx 6\text{ cm}
  ]

---

# 9. Wave Propagation

Radio waves propagate through space.

Plane-wave assumption:

* signal wavefront is approximately flat
* usually valid in far-field

---

# 10. Plane Wave

AoA algorithms often assume:

[
s(t) = e^{j2\pi ft}
]

Wavefront reaches all antennas with different delays.

---

# 11. Time Delay

Arrival delay between antennas:

[
\tau = \frac{d\cos(\theta)}{c}
]

Where:

* (d): antenna spacing
* (\theta): arrival angle

---

# 12. Phase Delay

Time delay creates phase delay:

[
\Delta \phi = 2\pi f \tau
]

Substitute wavelength:

[
\Delta \phi = \frac{2\pi d \cos(\theta)}{\lambda}
]

Core AoA equation.

---

# 13. Direction Vector

Incoming wave direction:

[
\vec d = [x, y, z]
]

Usually normalized:

[
||\vec d|| = 1
]

---

# 14. Steering Vector

Expected phase pattern across antennas for a given direction.

General form:

[
a(\theta)
=========

\left[
e^{-jk\vec r_1 \cdot \vec d},
e^{-jk\vec r_2 \cdot \vec d},
...
\right]
]

Where:

[
k = \frac{2\pi}{\lambda}
]

Very important in:

* beamforming
* MUSIC
* ESPRIT

---

# 15. Wavenumber

[
k = \frac{2\pi}{\lambda}
]

Represents spatial frequency.

---

# 16. Dot Product Geometry

AoA depends on projection:

[
\vec r \cdot \vec d
]

Meaning:

* how much antenna spacing aligns with wave direction.

---

# 17. Azimuth

Horizontal arrival angle.

Usually:

[
\theta = \arctan2(y, x)
]

Range:

* ([-180^\circ, 180^\circ])

---

# 18. Elevation

Vertical arrival angle.

Represents up/down direction.

---

# 19. Unwrap Phase

Phase is naturally limited to:

[
[-\pi, \pi]
]

So:

```text
179°
→ -179°
```

appears discontinuous.

Phase unwrapping reconstructs continuous phase.

Used with:

```python
np.unwrap()
```

---

# 20. Carrier Frequency Offset (CFO)

Transmitter and receiver oscillators are not perfectly synchronized.

Causes rotating phase error:

[
e^{j2\pi \Delta f t}
]

Effects:

* constellation rotation
* CSI phase drift
* AoA instability

---

# 21. Residual Carrier Offset (RCO)

Remaining hardware phase offset after synchronization.

Often calibrated or removed before AoA.

---

# 22. Noise

Wireless signals contain noise.

Common model:

[
y = x + n
]

Where:

[
n \sim \mathcal{CN}(0, \sigma^2)
]

Noise affects phase estimation accuracy.

---

# 23. AWGN

Additive White Gaussian Noise.

Properties:

* additive
* Gaussian distributed
* flat spectrum

Most common wireless noise model.

---

# 24. Multipath

Signal arrives through multiple paths:

* reflection
* diffraction
* scattering

AoA becomes harder because many angles exist simultaneously.

---

# 25. LOS (Line of Sight)

Direct path exists between TX and RX.

Naive AoA usually assumes dominant LOS.

---

# 26. NLOS (Non-Line of Sight)

Signal arrives mainly from reflections.

AoA estimation becomes less reliable.

---

# 27. Subcarriers

OFDM divides bandwidth into many narrow frequencies.

Each subcarrier has its own CSI:

[
H_k
]

AoA may average across subcarriers.

---

# 28. OFDM

Orthogonal Frequency Division Multiplexing.

Used in:

* WiFi
* LTE
* 5G

Converts frequency-selective channel into many narrowband channels.

---

# 29. Spatial Aliasing

Occurs when antenna spacing is too large:

[
d > \frac{\lambda}{2}
]

Causes ambiguous angles.

Equivalent to Nyquist violation in space.

---

# 30. Far Field

AoA formulas usually assume far-field:

[
R \gg \frac{2D^2}{\lambda}
]

Where:

* (R): distance to source
* (D): array size

Wavefront approximated as planar.

---

# 31. Near Field

Wavefront becomes spherical.

Simple AoA equations no longer accurate.

---

# 32. Beamforming

Electronically focusing array response toward a direction.

Uses steering vectors.

---

# 33. MUSIC Algorithm

Advanced AoA algorithm.

Uses:

* covariance matrix
* eigendecomposition
* signal/noise subspaces

Higher resolution than naive phase methods.

---

# 34. ESPRIT

Subspace-based AoA estimation method.

Uses rotational invariance between antenna pairs.

---

# 35. Covariance Matrix

Signal spatial statistics:

[
R = E[xx^H]
]

Important in modern AoA algorithms.

---

# 36. Eigenvectors / Signal Subspace

Used to separate:

* signal components
* noise components

Critical in MUSIC/ESPRIT.

---

# 37. Least Squares Estimation

AoA can be solved as:

[
A x = b
]

using least squares:

```python
np.linalg.lstsq()
```

Useful when measurements are noisy.

---

# 38. Spatial Frequency

Equivalent frequency concept in space:

[
u = \frac{d\cos(\theta)}{\lambda}
]

---

# 39. Circular Complex Gaussian

Wireless noise often modeled as:

[
\mathcal{CN}(0, \sigma^2)
]

Meaning:

* real and imaginary parts are Gaussian
* identical variance
* rotationally symmetric

---

# 40. DSP Pipeline for AoA

Typical processing chain:

```text
RF Signal
→ ADC
→ OFDM Demodulation
→ CSI Extraction
→ Phase Calibration
→ Phase Difference
→ AoA Estimation
→ Angle Output
```
