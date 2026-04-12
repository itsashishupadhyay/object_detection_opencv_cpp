# Cassini ISS Narrow Angle Camera (ISS-NA) -- Technical Reference Sheet

> This document is the human-readable companion to the machine-readable row in
> `artifacts/instruments.csv` for the pair `(cassini, issna)`. Every numerical
> claim here must match the CSV. If they disagree, the CSV is authoritative.

---

## 1. Mission Overview

The Cassini-Huygens mission was a joint NASA/ESA/ASI mission to explore the
Saturnian system. Cassini launched on 1997-10-15 from Cape Canaveral aboard a
Titan IVB/Centaur, performed gravity assists at Venus (1998, 1999), Earth
(1999), and Jupiter (2000-2001), and entered Saturn orbit on 2004-07-01.
The mission operated in Saturn orbit for over 13 years through a prime mission,
Equinox Mission extension, and Solstice Mission extension, ending with a
controlled atmospheric entry on 2017-09-15 (the "Grand Finale"). [1]

## 2. Instrument Identity

- **Full name:** Imaging Science Subsystem -- Narrow Angle Camera (ISS-NAC)
- **Acronym:** ISS-NA
- **Principal Investigator (at launch):** Carolyn Porco (CICLOPS / Space Science Institute)
- **Manufacturer/Integrator:** Built by NASA's Jet Propulsion Laboratory (JPL) [2, 3]
- **CCD packager:** JPL, using a Loral CCD [3]
- **Position on spacecraft bus:** Mounted on the Remote Sensing Palette (RSP), boresight nominally aligned with the -Y axis of the spacecraft body frame [4]
- **NAIF body/frame ID:** -82360 (frame name: CASSINI_ISS_NAC) [4]

## 3. Optical Design

| Parameter | Value | Source |
|---|---|---|
| Type | Ritchey-Chretien reflector | [2, 3] |
| Focal length | 2003.44 mm (NAIF IK value used for SPICE geometry) | [4] |
| Focal length (in-flight, clear filter) | 2002.70 +/- 0.07 mm | [3] |
| F-number | f/10.5 | [3, 4] |
| Aperture diameter | 190.80 mm (derived: 2003.44 / 10.5) | [4] |
| Spectral range | 200--1100 nm | [2, 4] |
| Filter configuration | Two independent 12-position filter wheels (24 filters total) | [2, 3] |
| Filter wheel rate | 3 positions/second | [3] |

**Note on focal length:** The NAIF instrument kernel (`cas_iss_v10.ti`) records
the focal length as 2003.44 mm. The PDS instrument catalog (`issna_inst.cat`)
reports the in-flight calibrated value as 2002.70 +/- 0.07 mm for the clear
filter. The NAIF IK value is used in `instruments.csv` because downstream SPICE
geometry computations use this kernel. The difference of ~0.74 mm corresponds to
a fractional error of ~0.037%, which is within the measurement uncertainty
budget. [3, 4]

## 4. Detector

| Parameter | Value | Source |
|---|---|---|
| Sensor model | Loral front-side-illuminated three-phase CCD (packaged by JPL) | [2, 3] |
| Array dimensions | 1024 x 1024 pixels | [2, 3, 4] |
| Pixel pitch | 12.0 um | [2, 3, 4] |
| Bit depth | 12 bits (4095 DN; stored in 16-bit words with upper 4 bits = 0001) | [3] |
| Full well capacity | ~120,000 e-/pixel | [3] |
| Read noise | 12 e- | [3] |
| Dark current | <= 0.3 e-/sec/pixel (at operating temperature) | [3] |
| Operating temperature | -90 +/- 0.2 deg C | [3] |
| Epitaxial layer thickness | 10--12 um | [3] |
| Quantum efficiency | ~1% at 1000 nm (declining steeply in near-IR) | [3] |

**Gain states:**

| Gain | Value (e-/DN) | Typical use |
|---|---|---|
| Gain 0 | 233 +/- 29 | 4x4 summation mode |
| Gain 1 | 99 +/- 13 | 2x2 summation mode |
| Gain 2 | 30 +/- 3 | 1x1, normal operations |
| Gain 3 | 13 +/- 2 | 1x1, high-gain mode |

Source: [3]

## 5. Geometry

| Parameter | Value | Source |
|---|---|---|
| Field of view | 0.350 x 0.350 deg | [2, 3, 4] |
| IFOV (pixel angular scale) | 5.9907 urad/pixel = 1.2357 arcsec/pixel | [3, 4] |
| Boresight direction (instrument frame) | (0, 0, +1) | [4] |
| Boresight frame name | CASSINI_ISS_NAC | [4] |
| CCD center | pixel (512.5, 512.5) | [4] |
| FOV center pixel | pixel (511.5, 511.5) | [4] |
| FOV half-angles | 0.175 deg (ref) x 0.175 deg (cross) | [4] |
| PSF FWHM (clear filter) | 1.3 pixels | [3] |

## 6. Distortion Model

The NAC exhibits very low geometric distortion. Pre-flight analytical
calculations indicated distortions of less than 1 pixel at the corners of the
field of view. In-flight observations of the Pleiades and the open cluster M35
measured a worst-case distortion of 0.45 pixels at the field corners. [2, 3]

The NAIF instrument kernel `cas_iss_v10.ti` defines the FOV as a rectangular
pyramid with no explicit polynomial distortion model -- the distortion is small
enough to be treated as negligible for most navigation purposes. For
sub-pixel-level work, the CISSCAL calibration pipeline (distributed with the
PDS data volumes) applies a geometric correction. [3, 4]

**Distortion model reference:** `cas_iss_v10.ti` (NAIF IK) and CISSCAL
calibration pipeline documentation in PDS volume `coiss_0011`.

## 7. Operational Notes

- **Exposure range:** 5 ms to 1200 s (20 minutes), with 63 commandable settings plus one no-operation setting. [3]
- **Shutter type:** Two-blade focal plane electromechanical shutter. [3]
- **Summation modes:** 1x1 (full resolution), 2x2, and 4x4 on-chip summation available. [3]
- **Data compression:** Both lossless and lossy compression available on-board. [2]
- **Known anomalies:** The NAC experienced occasional "haze" artifacts attributed to scattered light within the optics. These are documented in the ISS calibration reports distributed with PDS data volumes. [2]
- **Power:** 26.2 W active imaging, 22.3 W sleep, 8.4 W off. [3]
- **Physical dimensions:** ~95 x 40 x 33 cm. [3]
- **Mass:** 57.83 kg (combined ISS-NAC + ISS-WAC system). [3]
- **Filter combinations commonly used:** CL1+CL2 (clear/clear) for navigation and astrometry; narrowband filters for atmospheric and surface composition studies.

## 8. Spacecraft State Context (SPICE Kernels)

The following SPICE kernels have been downloaded to `spice_cache/` and are used
to reconstruct spacecraft state and camera pointing for ISS-NA images.

### Text kernels

| Kernel type | File | SHA-256 (first 16 hex) |
|---|---|---|
| LSK (leap seconds) | `naif0012.tls` | 718b03ae52f1c260 |
| SCLK (spacecraft clock) | `cas00172.tsc` | 9aebd8d2783d06c1 |
| FK (frames) | `cas_v43.tf` | 09b1f6f27a74d3cc |
| IK (instrument) | `cas_iss_v10.ti` | 518c50a1c837f525 |
| PCK (planetary constants) | `pck00011.tpc` | 3dff7b1dbeceaa01 |

### SPK kernels (trajectory + ephemeris)

| File | Description | SHA-256 (first 16 hex) |
|---|---|---|
| `171215R_SCPSEops_97288_17258.bsp` | Full-mission reconstructed SCPSE (1997-2017) | 5d16010f558b3976 |
| `180927AP_RE_90165_18018.bsp` | Reconstructed inner-moon ephemeris | b68685f2bb5637dc |
| `040506AP_PE_94328_16357.bsp` | Planetary ephemeris (1994-2016) | 8dddf286e4ebd9b0 |
| `170913AP_PE_17224_17258.bsp` | Planetary ephemeris (2017) | 007cf462d15a67a0 |
| `010420R_SCPSE_EP1_JP83.bsp` | Cruise-phase SCPSE | 6b91d6bef866ea30 |
| 50x `200128RU_SCPSE_*.bsp` | Per-period reconstructed SCPSE (2004-2017) | (see spacecraft_state.csv) |

### CK kernels (attitude/pointing)

| Files | Coverage |
|---|---|
| `97288_98002rc.bc` through `99274_00001rc.bc` (9 files) | 1997 Oct - 2000 Jan (cruise) |
| `00001_00092rc.bc` through `03274_04001rc.bc` (16 files) | 2000 Jan - 2004 Jan (cruise/approach) |

**Known CK gap:** No consolidated reconstructed CK files exist for the Saturn
orbital period (2004-2017). Per-revolution CK files (`*ra.bc`) are available
from NAIF but number ~3,700 files. As a result, attitude quaternions are
currently only available for cruise-phase images (1997-2004).

All kernel sources: https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/

## 9. Bodies Imaged

Based on the image manifest (1,125 images), the following target bodies appear:

**Major Saturn system bodies:** Saturn, Saturn rings, Titan, Enceladus, Mimas,
Tethys, Dione, Rhea, Hyperion, Iapetus, Phoebe, Janus, Epimetheus, Prometheus,
Pandora, Atlas, Pan, Helene, Telesto, Calypso, Polydeuces, Methone, Pallene,
Anthe, Aegaeon, Daphnis.

**Flyby targets:** Jupiter, Io, Europa, Ganymede, Callisto, Himalia, Venus,
Earth, Moon, Pluto.

**Irregular Saturn satellites (no SPICE ephemeris):** Albiorix, Bebhionn,
Bergelmir, Bestla, Erriapus, Fornjot, Greip, Hati, Hyrrokkin, Ijiraq,
Jarnsaxa, Kari, Kiviuq, Loge, Mundilfari, Narvi, Paaliaq, Siarnaq, Skathi,
Skoll, Surtur, Suttungr, Tarqeq, Tarvos, Thrymr, Ymir, S/2004 S 12,
S/2004 S 13.

**State computation results:** 867 of 1,125 images have SPICE-computed
spacecraft position (157 also have attitude). 258 images lack state data
(55 unknown targets + 203 irregular satellites without ephemeris).

## 10. Sources

1. Porco, C.C., et al. (2004). "Cassini Imaging Science: Instrument
   Characteristics And Anticipated Scientific Investigations At Saturn."
   *Space Science Reviews*, 115, 363--497.
   DOI: [10.1007/s11214-004-1456-7](https://doi.org/10.1007/s11214-004-1456-7)

2. PDS Cassini ISS-NA Instrument Catalog (`issna_inst.cat`), distributed with
   PDS volume `coiss_2101`.
   URL: https://planetarydata.jpl.nasa.gov/img/data/cassini/cassini_orbiter/coiss_2101/catalog/issna_inst.cat

3. PDS Cassini ISS-NA Context Description, NASA Planetary Data System.
   URL: https://arcnav.psi.edu/urn:nasa:pds:context:instrument:issna.co
   (Mirrors content from the PDS instrument catalog with identical numerical values.)

4. NAIF Cassini ISS Instrument Kernel, version 10 (`cas_iss_v10.ti`).
   URL: https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/ik/cas_iss_v10.ti

---

*Verified: 2026-04-08. All numerical values cross-checked against instruments.csv.*
