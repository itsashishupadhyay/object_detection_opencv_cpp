# OPUS Astronomical Image Downloader

Interactive Python tool for browsing and downloading astronomical images from NASA's OPUS (Outer Planets Unified Search) database, including spacecraft missions and ground-based observatories.

## Features

✨ **Key Features:**
- **Interactive Menu System**: Browse available images by Mission → Planet → Target (Moons, Asteroids, etc.)
- **Ground-Based Support**: Includes data from ground-based observatories (occultation profiles, diagrams)
- **Real-time Image Counts**: See exactly how many images are available for each category
- **Flexible Filtering**: Filter by mission, planet, target, or any combination
- **Smart Skip Options**: Download every Nth image (3rd, 5th, 10th, 25th, 50th, 100th, etc.)
- **Comprehensive Metadata**: Saves full metadata for every image in JSON format
- **Detailed Logging**: Complete log file documenting every image with all available information
- **Organized Structure**: Automatically organizes images by mission, planet, target, and category
- **Resume-friendly**: Checks for existing files and skips re-downloads

## Requirements

```bash
pip install requests
```

## Usage

### Basic Usage

```bash
python3 opus_image_downloader.py
```

### Step-by-Step Guide

1. **Run the script**:
   ```bash
   cd supporting_scripts
   python3 opus_image_downloader.py
   ```

2. **Choose output directory** (default: `./opus_dataset`):
   ```
   Enter output directory (default: ./opus_dataset):
   ```

3. **Select Mission(s)**:
   ```
   📡 AVAILABLE MISSIONS (All Observation Types)
   ──────────────────────────────────────────────
     1) Cassini                     - 1,489,721 observations
     2) Voyager                     -    76,732 observations
     3) New Horizons                -    23,623 observations
     4) Hubble                      -    21,766 observations
     5) Galileo                     -    14,470 observations
     6) Ground-based                -       769 observations

   Enter mission number(s) (comma-separated), 'all', or 'skip':
   > 1
   ```

4. **Select Planet(s)**:
   ```
   🪐 AVAILABLE PLANETS
   ──────────────────────────────────────────────
     1) Saturn                      - 350,124 images
     2) Jupiter                     -  45,678 images
     3) Mars                        -  12,345 images

   Enter planet number(s) (comma-separated), 'all', or 'skip':
   > 1
   ```

5. **Select Target(s)** (Moons, Asteroids, etc.):
   ```
   🌙 AVAILABLE TARGETS
   ──────────────────────────────────────────────

   MOONS
     1) Titan                       -  45,123 images
     2) Enceladus                   -  23,456 images
     3) Rhea                        -  12,789 images

   PLANETS
     4) Saturn                      - 234,567 images

   RINGS
     5) Saturn Rings                -  89,012 images

   Enter target number(s) (comma-separated), 'all', or 'skip':
   > 1,2
   ```

6. **Set Download Limits**:
   ```
   📊 Total images matching your criteria: 68,579

   💾 DOWNLOAD OPTIONS
   ──────────────────────────────────────────────
   How many images would you like to download?
     Enter a number, 'all', or 'max' for all available
   > 100
   ```

7. **Choose Skip Interval**:
   ```
   ⏭️  SKIP OPTIONS
   Download every Nth image:
     1 - Download all (no skip)
     3 - Download every 3rd image
     5 - Download every 5th image
     10 - Download every 10th image
     25, 50, 100, 250, 500, 1000 - etc.
   Enter skip interval (default=1):
   > 10
   ```

8. **Confirm and Download**:
   ```
   ════════════════════════════════════════════════════════════════════════════════
   DOWNLOAD SUMMARY
   ════════════════════════════════════════════════════════════════════════════════
   Missions: ['Cassini']
   Planets: ['Saturn']
   Targets: ['Titan', 'Enceladus']
   Total available: 68,579 images
   Will download: 100 images
   Skip interval: Every 10 image(s)
   Output directory: opus_dataset
   ════════════════════════════════════════════════════════════════════════════════

   Proceed with download? (yes/no): yes
   ```

## Output Structure

```
opus_dataset/
├── images/
│   ├── by_mission/           # Organized by spacecraft/observatory
│   │   ├── Cassini/
│   │   ├── Voyager/
│   │   └── Hubble/
│   ├── by_planet/            # Organized by planet
│   │   ├── Saturn/
│   │   ├── Jupiter/
│   │   └── Mars/
│   ├── by_target/            # Organized by specific target
│   │   ├── Titan/
│   │   ├── Enceladus/
│   │   └── Europa/
│   └── by_category/          # Organized by object type
│       ├── moons/
│       ├── planets/
│       ├── asteroids/
│       └── other/
├── metadata/
│   └── by_target/            # JSON metadata for each image
│       ├── Titan/
│       │   ├── co-iss-n1234567890.json
│       │   └── co-iss-n1234567891.json
│       └── Enceladus/
├── download_log.txt          # Complete download log
└── DOWNLOAD_SUMMARY.md       # Summary statistics
```

## Metadata Information

Each image has comprehensive metadata saved in JSON format, including:

### General Information
- **OPUS ID**: Unique identifier
- **Mission**: Spacecraft or observatory name
- **Instrument**: Camera/detector used
- **Target**: Celestial body observed
- **Planet**: Host planet
- **Observation Time**: Date and time of observation

### Image Details
- **Exposure Duration**: Length of exposure
- **Image Dimensions**: Size in pixels
- **Intensity Levels**: Bit depth
- **Image Type**: Frame, mosaic, etc.

### Geometry Information
- **Distance to Target**: Observer-target distance
- **Phase Angle**: Sun-target-observer angle
- **Emission Angle**: Target surface normal to observer
- **Incidence Angle**: Sun to target surface angle
- **Resolution**: Km/pixel or m/pixel

### Camera Information
- **Filter**: Wavelength filter used
- **Camera Mode**: Imaging mode/settings
- **Compression**: Data compression type

## Example Use Cases

### Download Cassini Images of Titan
```python
# Select:
# Mission: 1 (Cassini)
# Planet: 1 (Saturn)
# Target: 1 (Titan)
# Download: 500 images
# Skip: 10 (every 10th image)
```

### Download All Hubble Images of Jupiter's Moons
```python
# Select:
# Mission: 3 (Hubble)
# Planet: 2 (Jupiter)
# Target: all moons
# Download: all
# Skip: 1 (download all)
```

### Sample Dataset for Testing
```python
# Select:
# Mission: skip (all missions)
# Planet: skip (all planets)
# Target: 1,2,3 (first 3 targets)
# Download: 50
# Skip: 25 (every 25th image)
```

## Metadata Log Example

The `metadata_log_YYYYMMDD_HHMMSS.txt` file contains detailed information for each downloaded image:

```
════════════════════════════════════════════════════════════════════════════════════
OPUS ID: co-iss-n1635813867
Mission: Cassini
Instrument: Cassini ISS NAC
Planet: Saturn
Target: Enceladus
Category: moons
Observation Time: 2009-11-02T00:01:22.626
Image URL: https://opus.pds-rings.seti.org/holdings/previews/COISS_2xxx/...
Local Path: opus_dataset/images/by_target/Enceladus/co-iss-n1635813867.jpg
Metadata File: opus_dataset/metadata/by_target/Enceladus/co-iss-n1635813867.json

Full Metadata:
{
  "General Constraints": {
    "planet": "Saturn",
    "target": "Enceladus",
    "mission": "Cassini",
    "instrument": "Cassini ISS NAC",
    "time1": "2009-11-02T00:01:22.626",
    "time2": "2009-11-02T00:01:26.426",
    "observationduration": "3.8",
    "rightasc1": "314.537",
    "declination1": "-18.234"
  },
  "PDS Constraints": {
    "bundleid": "COISS_2063",
    "datasetid": "CO-S-ISSNA/ISSWA-2-EDR-V1.0",
    "productid": "1_N1635813867.118",
    "opusid": "co-iss-n1635813867"
  },
  "Image Constraints": {
    "duration": "3.8",
    "greaterpixelsize": "1024",
    "lesserpixelsize": "1024",
    "levels": "4096"
  },
  "Enceladus Surface Geometry Constraints": {
    "centerphaseangle": "161.414",
    "planetographiclatitude1": "70.234",
    "rangetobody1": "123456.789",
    "centerresolution": "0.234"
  }
}
```

## Tips and Best Practices

1. **Start Small**: Test with 10-20 images first to verify the setup
2. **Use Skip Wisely**: For large datasets (>10,000 images), use skip intervals (50, 100, 250)
3. **Check Disk Space**: Each image is ~50-500KB; plan accordingly
4. **Resume Capability**: Script checks for existing files, so you can safely re-run
5. **Metadata First**: Review metadata logs to understand image characteristics
6. **API Rate Limiting**: Script includes 0.5s delays between requests to be respectful

## Advanced Filtering

You can filter by:
- **Mission + Planet**: E.g., "Cassini images of Saturn"
- **Planet + Target**: E.g., "All Jupiter moon images"
- **Mission + Target**: E.g., "Voyager images of Titan"
- **Multiple Targets**: E.g., "Io, Europa, Ganymede"

## Troubleshooting

### Connection Errors
```
ERROR: API request failed: Connection timeout
```
- Check internet connection
- OPUS servers may be temporarily unavailable
- Try again later

### No Images Found
```
📊 Total images matching your criteria: 0
```
- Try broader filters (select 'skip' or 'all')
- Check if target name is spelled correctly
- Some combinations may have no images

### Permission Errors
```
ERROR: Permission denied creating directory
```
- Check write permissions in output directory
- Try a different output location

## Dataset Information

**Source**: NASA Planetary Data System (PDS) Ring-Moon Systems Node
**URL**: https://opus.pds-rings.seti.org
**Data**: Spacecraft images from Cassini, Voyager, Galileo, New Horizons, Hubble, and ground-based observatories
**Coverage**: Outer planets (Jupiter, Saturn, Uranus, Neptune, Pluto) and their moons, rings, and asteroids

### Ground-Based Observations

The tool now includes support for ground-based observatory data:
- **Observatory Types**: Kuiper Airborne Observatory, Earth-based telescopes
- **Data Types**: Occultation profiles, stellar occultation diagrams, photometric data
- **Observation Targets**: Planetary rings, planets, moons, and asteroids
- **Image Types**: Diagrams and visualizations of occultation data (PNG format)
- **Coverage**: Historical observations dating back to the 1970s

Ground-based observations are particularly valuable for:
- Ring system studies (occultation profiles)
- Historical planetary observations
- Comparative analysis with spacecraft data
- Supplementing spacecraft mission datasets

## Citation

If using this dataset for research, please cite:
- NASA PDS Ring-Moon Systems Node OPUS: https://opus.pds-rings.seti.org
- Original mission data sources (Cassini, Voyager, etc.)
- Your research publication

## License

Images are provided by NASA and are in the public domain.
Tool created for astronomical object detection research.

## Support

For issues with:
- **OPUS API**: https://opus.pds-rings.seti.org
- **This tool**: Check the documentation above or review the source code

---

**Created for**: Real-Time Celestial Object Detection and Autonomous Navigation Research
**Compatible with**: YOLO26, YOLOv8, and other object detection frameworks
**Last Updated**: 2026-02-16
