# ToF (Time of Flight) Concepts

## 1. ToF (Time of Flight)

ToF is the propagation time required for a wireless signal to travel from transmitter to receiver.

Symbol:

[
\tau
]

Distance relation:

[
d = c\tau
]

Where:

* (d): propagation distance
* (c): speed of light
* (\tau): propagation delay

---

# 2. Propagation Delay

Wireless signals do not arrive instantly.

A signal traveling through space experiences:

[
\tau = \frac{d}{c}
]

Example:

* 1 meter propagation:
  [
  \tau \approx 3.3ns
  ]

---

# 3. CSI (Channel State Information)

CSI is the frequency-domain wireless channel response.

General form:

[
H(f)=Ae^{j\phi}
]

Contains:

* amplitude information
* phase information

ToF estimation mainly relies on phase evolution across frequencies.

---

# 4. Frequency Domain vs Time Domain

Wireless channels can be represented in two equivalent forms:

| Domain           | Representation |
| ---------------- | -------------- |
| Frequency domain | CSI / CFR      |
| Time domain      | CIR            |

---

# 5. CFR (Channel Frequency Response)

Frequency-domain channel:

[
H(f)
]

Represents how each frequency is modified by the wireless channel.

In OFDM:

[
H[k]
]

is the CSI of subcarrier (k).

---

# 6. CIR (Channel Impulse Response)

Time-domain channel representation:

[
h(t)
]

Shows:

* propagation delays
* multipath structure
* path amplitudes

---

# 7. Fourier Relationship

CFR and CIR are Fourier pairs.

[
h(t)=\mathcal{F}^{-1}{H(f)}
]

Meaning:

* IFFT(CSI) → CIR
* FFT(CIR) → CSI

---

# 8. OFDM

OFDM divides bandwidth into many narrowband subcarriers.

Each subcarrier experiences:

[
H[k]
]

ToF estimation uses phase relationships across subcarriers.

---

# 9. Subcarriers

Subcarriers are equally spaced frequencies inside OFDM bandwidth.

Example:

```text id="cfxzbi"
f0, f1, f2, ...
```

Each subcarrier contains channel phase information.

---

# 10. Delay-Induced Phase Rotation

Propagation delay creates phase rotation across frequency:

[
H(f)=Ae^{-j2\pi f\tau}
]

Important interpretation:

* larger delay
* steeper phase slope across frequencies

---

# 11. Linear Phase Slope

ToF fundamentally appears as:

```text id="sryjxe"
linear phase change across subcarriers
```

This is one of the most important concepts in wireless sensing.

---

# 12. IFFT-Based ToF Estimation

Pipeline:

```text id="58j1pn"
CSI (frequency domain)
→ IFFT
→ CIR (time domain)
→ peak detection
→ ToF
```

---

# 13. CIR Peaks

Each CIR peak corresponds to a propagation path.

Example:

| Peak   | Meaning            |
| ------ | ------------------ |
| Peak 1 | LOS path           |
| Peak 2 | Reflection         |
| Peak 3 | Another reflection |

---

# 14. Multipath

Wireless signals travel through multiple paths:

* wall reflection
* floor reflection
* scattering
* diffraction

Result:

```text id="s4bz0n"
multiple CIR peaks
```

---

# 15. LOS (Line of Sight)

Direct propagation path between TX and RX.

Usually produces earliest arrival peak.

---

# 16. NLOS (Non-Line of Sight)

Signal mainly arrives through reflections.

Strongest peak may not correspond to shortest path.

---

# 17. Peak Detection

Naive ToF methods often assume:

```text id="12g5gg"
strongest CIR peak = true propagation delay
```

Using:

```python id="av4fmk"
np.argmax(np.abs(cir))
```

This assumption can fail in multipath environments.

---

# 18. FFT / IFFT

FFT converts:

```text id="s9k2bk"
time → frequency
```

IFFT converts:

```text id="h1yjlwm"
frequency → time
```

ToF estimation heavily relies on IFFT.

---

# 19. FFT Size

Often selected as:

[
N = 2^k
]

because FFT algorithms are computationally efficient for powers of two.

---

# 20. Delay Resolution

Time resolution depends on bandwidth:

[
\Delta t \approx \frac{1}{BW}
]

Where:

* (BW): signal bandwidth

---

# 21. Distance Resolution

Distance resolution:

[
\Delta d = c\Delta t
]

Equivalent form:

[
\Delta d \approx \frac{c}{BW}
]

---

# 22. Why Large Bandwidth Matters

Larger bandwidth provides:

* finer delay resolution
* sharper CIR peaks
* more accurate ToF estimation

---

Example:

| Bandwidth | Delay Resolution |
| --------- | ---------------- |
| 20 MHz    | 50 ns            |
| 160 MHz   | 6.25 ns          |

---

# 23. Delay Bin

IFFT output index corresponds to a delay bin.

Conversion:

[
\tau_n = \frac{n}{BW}
]

Where:

* (n): CIR peak index

---

# 24. Half-Spectrum Usage

IFFT outputs periodic structure.

Often only first half is physically meaningful:

```python id="y0z9m2"
cir[:N//2]
```

---

# 25. Noise

Wireless measurements contain noise:

[
y=x+n
]

Noise affects:

* CIR peak clarity
* delay estimation accuracy

---

# 26. AWGN

Most common noise model:

[
n\sim\mathcal{CN}(0,\sigma^2)
]

Used in simulations and DSP pipelines.

---

# 27. Frequency Selective Channel

Different frequencies experience different attenuation and phase.

Multipath causes:

```text id="6ivm0s"
frequency-dependent fading
```

---

# 28. Delay Spread

Difference between earliest and latest propagation paths.

Large delay spread causes:

* ISI
* frequency selectivity

---

# 29. Cyclic Prefix (CP)

OFDM adds guard interval called cyclic prefix.

Purpose:

* absorb multipath delay spread
* prevent inter-symbol interference

---

# 30. Super-Resolution ToF

FFT resolution is limited by bandwidth.

Advanced methods exceed FFT-bin resolution:

* MUSIC
* ESPRIT
* SAGE
* Matrix Pencil
* Sparse recovery

---

# 31. CFO (Carrier Frequency Offset)

Frequency mismatch introduces additional phase rotation:

[
e^{j2\pi \Delta f t}
]

Can corrupt ToF estimation.

---

# 32. STO (Sampling Time Offset)

Timing synchronization error between transmitter and receiver.

Causes phase distortion across subcarriers.

---

# 33. Phase Calibration

Practical CSI often contains hardware phase offsets.

Calibration is necessary before precise ToF estimation.

---

# 34. ToF vs AoA vs Doppler

Wireless phase varies in different dimensions:

| Dimension          | Physical Meaning |
| ------------------ | ---------------- |
| Across antennas    | AoA              |
| Across frequencies | ToF              |
| Across time        | Doppler          |

This is a fundamental wireless sensing principle.

---

# 35. Doppler

Motion creates time-varying phase:

[
f_D = \frac{v}{\lambda}
]

Where:

* (v): relative velocity

Used for motion sensing and tracking.

---

# 36. Unified Wireless Sensing View

CSI contains structured phase information:

```text id="t2j4d5"
space dimension → AoA
frequency dimension → ToF
time dimension → Doppler
```

Modern wireless sensing systems combine all three.

---

# 37. Typical ToF Pipeline

```text id="13n9ew"
RF Signal
→ ADC
→ OFDM Demodulation
→ CSI Extraction
→ Phase Calibration
→ IFFT
→ CIR
→ Peak Detection
→ ToF Estimation
```
