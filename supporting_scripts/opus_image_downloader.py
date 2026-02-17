#!/usr/bin/env python3
"""
OPUS Image Dataset Downloader
Interactive Python tool for browsing and downloading astronomical images from NASA's OPUS database
"""

import requests
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict

# OPUS API Configuration
BASE_URL = "https://opus.pds-rings.seti.org/opus/api"
DEFAULT_LIMIT = 100

class OPUSImageDownloader:
    def __init__(self, output_dir: str = "./opus_dataset"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        self.log_file = self.output_dir / "download_log.txt"

        # Create directory structure
        self.setup_directories()

        # Cache for API responses
        self.cache = {}

    def setup_directories(self):
        """Create necessary directory structure"""
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.images_dir / "by_mission").mkdir(exist_ok=True)
        (self.images_dir / "by_planet").mkdir(exist_ok=True)
        (self.images_dir / "by_target").mkdir(exist_ok=True)
        (self.images_dir / "by_category").mkdir(exist_ok=True)
        (self.metadata_dir / "by_target").mkdir(exist_ok=True)

        print(f"✓ Directory structure created at: {self.output_dir}")

    def log(self, message: str):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling"""
        url = f"{BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.log(f"ERROR: API request failed: {e}")
            return {}

    def get_available_missions(self) -> List[Tuple[str, int]]:
        """Get all available missions with image counts (includes ground-based observatories)"""
        self.log("Fetching available missions...")

        # Get mission field options - no observationtype filter to include ground-based
        data = self.api_request("meta/mults/mission.json")

        if not data or 'mults' not in data:
            return []

        missions = []
        for mission, count in data['mults'].items():
            if count > 0:
                missions.append((mission, count))

        # Sort by count descending
        missions.sort(key=lambda x: x[1], reverse=True)
        return missions

    def get_available_planets(self, mission: str = None) -> List[Tuple[str, int]]:
        """Get all available planets with image counts"""
        self.log("Fetching available planets...")

        params = {}
        if mission:
            params["mission"] = mission

        data = self.api_request("meta/mults/planet.json", params)

        if not data or 'mults' not in data:
            return []

        planets = []
        for planet, count in data['mults'].items():
            if count > 0:
                planets.append((planet, count))

        planets.sort(key=lambda x: x[1], reverse=True)
        return planets

    def get_available_targets(self, mission: str = None, planet: str = None) -> Dict[str, List[Tuple[str, int]]]:
        """Get all available targets categorized by type"""
        self.log("Fetching available targets...")

        params = {"limit": 1000}
        if mission:
            params["mission"] = mission
        if planet:
            params["planet"] = planet

        data = self.api_request("meta/mults/target.json", params)

        if not data or 'mults' not in data:
            return {}

        # Categorize targets
        categories = {
            "Planets": [],
            "Moons": [],
            "Asteroids": [],
            "Rings": [],
            "Other": []
        }

        planet_keywords = ['venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto']
        moon_keywords = ['moon', 'io', 'europa', 'ganymede', 'callisto', 'titan', 'enceladus',
                        'rhea', 'iapetus', 'dione', 'mimas', 'tethys', 'hyperion', 'phoebe',
                        'triton', 'charon', 'phobos', 'deimos']
        asteroid_keywords = ['asteroid', 'ida', 'gaspra', 'eros', 'mathilde', 'steins', 'lutetia',
                           'vesta', 'ceres', 'bennu', 'ryugu']

        for target, count in data['mults'].items():
            if count == 0:
                continue

            target_lower = target.lower()

            if any(kw in target_lower for kw in planet_keywords):
                categories["Planets"].append((target, count))
            elif any(kw in target_lower for kw in moon_keywords):
                categories["Moons"].append((target, count))
            elif any(kw in target_lower for kw in asteroid_keywords):
                categories["Asteroids"].append((target, count))
            elif 'ring' in target_lower:
                categories["Rings"].append((target, count))
            else:
                categories["Other"].append((target, count))

        # Sort each category by count
        for category in categories:
            categories[category].sort(key=lambda x: x[1], reverse=True)

        return categories

    def get_available_instruments(self, mission: str = None) -> List[Tuple[str, int]]:
        """Get all available instruments with image counts"""
        self.log("Fetching available instruments...")

        params = {}
        if mission:
            params["mission"] = mission

        data = self.api_request("meta/mults/instrument.json", params)

        if not data or 'mults' not in data:
            return []

        instruments = []
        for instrument, count in data['mults'].items():
            if count > 0:
                instruments.append((instrument, count))

        instruments.sort(key=lambda x: x[1], reverse=True)
        return instruments

    def display_menu(self):
        """Display interactive menu for browsing"""
        print("\n" + "="*80)
        print("OPUS ASTRONOMICAL IMAGE DATASET BROWSER")
        print("NASA Planetary Data System Ring-Moon Systems Node")
        print("="*80)

        # Step 1: Select Mission
        missions = self.get_available_missions()

        print("\n📡 AVAILABLE MISSIONS (Image Category Only)")
        print("-" * 80)
        for idx, (mission, count) in enumerate(missions, 1):
            print(f"{idx:3d}) {mission:30s} - {count:>8,d} images")

        print("\nEnter mission number(s) (comma-separated), 'all', or 'skip':")
        mission_input = input("> ").strip()

        if mission_input.lower() == 'skip':
            selected_missions = None
        elif mission_input.lower() == 'all':
            selected_missions = [m[0] for m in missions]
        else:
            try:
                indices = [int(x.strip()) for x in mission_input.split(',')]
                selected_missions = [missions[i-1][0] for i in indices if 1 <= i <= len(missions)]
            except (ValueError, IndexError):
                print("Invalid selection. Exiting.")
                return None

        # Step 2: Select Planet
        print("\n🪐 AVAILABLE PLANETS")
        print("-" * 80)

        mission_filter = selected_missions[0] if selected_missions and len(selected_missions) == 1 else None
        planets = self.get_available_planets(mission_filter)

        for idx, (planet, count) in enumerate(planets, 1):
            print(f"{idx:3d}) {planet:30s} - {count:>8,d} images")

        print("\nEnter planet number(s) (comma-separated), 'all', or 'skip':")
        planet_input = input("> ").strip()

        if planet_input.lower() == 'skip':
            selected_planets = None
        elif planet_input.lower() == 'all':
            selected_planets = [p[0] for p in planets]
        else:
            try:
                indices = [int(x.strip()) for x in planet_input.split(',')]
                selected_planets = [planets[i-1][0] for i in indices if 1 <= i <= len(planets)]
            except (ValueError, IndexError):
                print("Invalid selection. Exiting.")
                return None

        # Step 3: Select Targets (Moons, Asteroids, etc.)
        print("\n🌙 AVAILABLE TARGETS")
        print("-" * 80)

        planet_filter = selected_planets[0] if selected_planets and len(selected_planets) == 1 else None
        target_categories = self.get_available_targets(mission_filter, planet_filter)

        all_targets = []
        idx = 1
        for category, targets in target_categories.items():
            if not targets:
                continue

            print(f"\n{category.upper()}")
            for target, count in targets:
                print(f"{idx:3d}) {target:30s} - {count:>8,d} images")
                all_targets.append((target, count))
                idx += 1

        print("\nEnter target number(s) (comma-separated), 'all', or 'skip':")
        target_input = input("> ").strip()

        if target_input.lower() == 'skip':
            selected_targets = None
        elif target_input.lower() == 'all':
            selected_targets = [t[0] for t in all_targets]
        else:
            try:
                indices = [int(x.strip()) for x in target_input.split(',')]
                selected_targets = [all_targets[i-1][0] for i in indices if 1 <= i <= len(all_targets)]
            except (ValueError, IndexError):
                print("Invalid selection. Exiting.")
                return None

        # Step 4: Get total count and download options
        search_params = {}
        if selected_missions and len(selected_missions) == 1:
            search_params["mission"] = selected_missions[0]
        if selected_planets:
            search_params["planet"] = ','.join(selected_planets)
        if selected_targets:
            search_params["target"] = ','.join(selected_targets)

        count_data = self.api_request("meta/result_count.json", search_params)
        total_count = count_data.get('data', [{}])[0].get('result_count', 0)

        print(f"\n📊 Total images matching your criteria: {total_count:,}")

        # Download options
        print("\n💾 DOWNLOAD OPTIONS")
        print("-" * 80)
        print("How many images would you like to download?")
        print("  Enter a number, 'all', or 'max' for all available")
        max_images = input("> ").strip()

        if max_images.lower() in ['all', 'max']:
            max_images = total_count
        else:
            try:
                max_images = int(max_images)
            except ValueError:
                max_images = 100

        print("\n⏭️  SKIP OPTIONS")
        print("Download every Nth image:")
        print("  1 - Download all (no skip)")
        print("  3 - Download every 3rd image")
        print("  5 - Download every 5th image")
        print("  10 - Download every 10th image")
        print("  25, 50, 100, 250, 500, 1000 - etc.")
        skip_n = input("Enter skip interval (default=1): ").strip()

        try:
            skip_n = int(skip_n)
        except ValueError:
            skip_n = 1

        return {
            'missions': selected_missions,
            'planets': selected_planets,
            'targets': selected_targets,
            'search_params': search_params,
            'max_images': max_images,
            'skip_n': skip_n,
            'total_available': total_count
        }

    def download_images(self, config: Dict):
        """Download images based on configuration"""
        if not config:
            return

        self.log(f"Starting download: {config['max_images']} images (skip every {config['skip_n']})")
        self.log(f"Total available: {config['total_available']}")

        search_params = config['search_params'].copy()
        search_params['cols'] = 'opusid,instrument,mission,target,planet,time1,time2,observationduration'

        downloaded = 0
        startobs = 1
        skip_counter = 0

        # Create metadata log header
        metadata_log = self.metadata_dir / f"metadata_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(metadata_log, 'w') as f:
            f.write("="*100 + "\n")
            f.write("OPUS IMAGE DOWNLOAD LOG\n")
            f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Criteria: {json.dumps(config['search_params'], indent=2)}\n")
            f.write("="*100 + "\n\n")

        while downloaded < config['max_images']:
            # Calculate how many to fetch in this batch
            batch_size = min(DEFAULT_LIMIT, (config['max_images'] - downloaded) * config['skip_n'])

            search_params['startobs'] = startobs
            search_params['limit'] = batch_size

            # Fetch observations
            self.log(f"Fetching batch starting at observation {startobs}...")
            data = self.api_request("data.json", search_params)

            if not data or 'page' not in data:
                self.log("No more data available")
                break

            observations = data['page']
            if not observations:
                break

            # Process each observation
            for obs_data in observations:
                skip_counter += 1

                # Skip based on skip_n
                if skip_counter % config['skip_n'] != 0:
                    continue

                opus_id = obs_data[0]

                # Download this image
                success = self.download_single_image(opus_id, metadata_log)

                if success:
                    downloaded += 1
                    self.log(f"Progress: {downloaded}/{config['max_images']} images downloaded")

                if downloaded >= config['max_images']:
                    break

                # Rate limiting
                time.sleep(0.5)

            startobs += len(observations)

            # Check if we've exhausted all available observations
            if len(observations) < batch_size:
                break

        self.log(f"Download complete! Total images: {downloaded}")
        self.log(f"Metadata log: {metadata_log}")

    def download_single_image(self, opus_id: str, metadata_log: Path) -> bool:
        """Download a single image and its metadata"""
        try:
            # Get full metadata
            metadata = self.api_request(f"metadata/{opus_id}.json")

            if not metadata:
                self.log(f"Failed to get metadata for {opus_id}")
                return False

            # Get image URL
            image_data = self.api_request(f"image/med/{opus_id}.json")

            if not image_data or 'data' not in image_data or not image_data['data']:
                self.log(f"No image available for {opus_id}")
                return False

            image_url = image_data['data'][0].get('url')

            if not image_url:
                self.log(f"No image URL for {opus_id}")
                return False

            # Extract key metadata
            general = metadata.get('General Constraints', {})
            pds = metadata.get('PDS Constraints', {})
            image_info = metadata.get('Image Constraints', {})

            mission = general.get('mission', 'Unknown')
            planet = general.get('planet', 'Unknown')
            target = general.get('target', 'Unknown')
            instrument = general.get('instrument', 'Unknown')
            time1 = general.get('time1', 'Unknown')

            # Determine category
            target_lower = target.lower()
            if any(kw in target_lower for kw in ['moon', 'io', 'europa', 'titan', 'enceladus']):
                category = 'moons'
            elif any(kw in target_lower for kw in ['asteroid', 'ida', 'ceres']):
                category = 'asteroids'
            elif any(kw in target_lower for kw in ['jupiter', 'saturn', 'mars', 'venus']):
                category = 'planets'
            else:
                category = 'other'

            # Create directory structure
            target_safe = target.replace(' ', '_').replace('/', '_')
            mission_safe = mission.replace(' ', '_').replace('/', '_')

            target_dir = self.images_dir / "by_target" / target_safe
            mission_dir = self.images_dir / "by_mission" / mission_safe
            category_dir = self.images_dir / "by_category" / category
            metadata_target_dir = self.metadata_dir / "by_target" / target_safe

            target_dir.mkdir(parents=True, exist_ok=True)
            mission_dir.mkdir(parents=True, exist_ok=True)
            category_dir.mkdir(parents=True, exist_ok=True)
            metadata_target_dir.mkdir(parents=True, exist_ok=True)

            # Download image
            file_ext = Path(image_url).suffix or '.jpg'
            image_filename = f"{opus_id}{file_ext}"
            image_path = target_dir / image_filename

            if image_path.exists():
                self.log(f"Image already exists: {image_filename}")
            else:
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()

                with open(image_path, 'wb') as f:
                    f.write(response.content)

                self.log(f"Downloaded: {image_filename}")

                # Create symlinks in other directories
                try:
                    mission_link = mission_dir / image_filename
                    category_link = category_dir / image_filename

                    if not mission_link.exists():
                        mission_link.symlink_to(image_path)
                    if not category_link.exists():
                        category_link.symlink_to(image_path)
                except OSError:
                    pass  # Symlinks may not be supported on all systems

            # Save detailed metadata
            metadata_file = metadata_target_dir / f"{opus_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Append to metadata log
            with open(metadata_log, 'a') as f:
                f.write("="*100 + "\n")
                f.write(f"OPUS ID: {opus_id}\n")
                f.write(f"Mission: {mission}\n")
                f.write(f"Instrument: {instrument}\n")
                f.write(f"Planet: {planet}\n")
                f.write(f"Target: {target}\n")
                f.write(f"Category: {category}\n")
                f.write(f"Observation Time: {time1}\n")
                f.write(f"Image URL: {image_url}\n")
                f.write(f"Local Path: {image_path}\n")
                f.write(f"Metadata File: {metadata_file}\n")
                f.write(f"\nFull Metadata:\n")
                f.write(json.dumps(metadata, indent=2))
                f.write("\n\n")

            return True

        except Exception as e:
            self.log(f"Error downloading {opus_id}: {e}")
            return False

    def generate_summary(self):
        """Generate download summary report"""
        self.log("Generating summary report...")

        summary_file = self.output_dir / "DOWNLOAD_SUMMARY.md"

        # Count images by category
        by_mission = defaultdict(int)
        by_planet = defaultdict(int)
        by_target = defaultdict(int)
        by_category = defaultdict(int)

        for img_dir in [self.images_dir / "by_mission", self.images_dir / "by_planet",
                        self.images_dir / "by_target", self.images_dir / "by_category"]:
            if not img_dir.exists():
                continue

            for subdir in img_dir.iterdir():
                if subdir.is_dir():
                    count = len(list(subdir.glob("*")))

                    if img_dir.name == "by_mission":
                        by_mission[subdir.name] = count
                    elif img_dir.name == "by_planet":
                        by_planet[subdir.name] = count
                    elif img_dir.name == "by_target":
                        by_target[subdir.name] = count
                    elif img_dir.name == "by_category":
                        by_category[subdir.name] = count

        with open(summary_file, 'w') as f:
            f.write("# OPUS Image Dataset Download Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset Location:** `{self.output_dir}`\n\n")

            f.write("## Dataset Statistics\n\n")

            if by_mission:
                f.write("### Images by Mission\n\n")
                for mission, count in sorted(by_mission.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- **{mission}**: {count:,} images\n")
                f.write("\n")

            if by_planet:
                f.write("### Images by Planet\n\n")
                for planet, count in sorted(by_planet.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- **{planet}**: {count:,} images\n")
                f.write("\n")

            if by_category:
                f.write("### Images by Category\n\n")
                for category, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- **{category.title()}**: {count:,} images\n")
                f.write("\n")

            if by_target:
                f.write("### Images by Target (Top 20)\n\n")
                top_targets = sorted(by_target.items(), key=lambda x: x[1], reverse=True)[:20]
                for target, count in top_targets:
                    f.write(f"- **{target}**: {count:,} images\n")
                f.write("\n")

            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write(f"{self.output_dir.name}/\n")
            f.write("├── images/\n")
            f.write("│   ├── by_mission/\n")
            f.write("│   ├── by_planet/\n")
            f.write("│   ├── by_target/\n")
            f.write("│   └── by_category/\n")
            f.write("├── metadata/\n")
            f.write("│   └── by_target/\n")
            f.write("├── download_log.txt\n")
            f.write("└── DOWNLOAD_SUMMARY.md\n")
            f.write("```\n\n")

            f.write("## Usage Notes\n\n")
            f.write("- All images are organized by mission, planet, target, and category\n")
            f.write("- Detailed metadata is stored in JSON format for each image\n")
            f.write("- A comprehensive log file tracks all downloads and metadata\n")
            f.write("- Source: [NASA PDS OPUS](https://opus.pds-rings.seti.org)\n")

        self.log(f"Summary report saved: {summary_file}")


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("OPUS ASTRONOMICAL IMAGE DATASET DOWNLOADER")
    print("Interactive Python Tool for NASA's Planetary Data System")
    print("="*80 + "\n")

    # Get output directory
    output_dir = input("Enter output directory (default: ./opus_dataset): ").strip()
    if not output_dir:
        output_dir = "./opus_dataset"

    downloader = OPUSImageDownloader(output_dir)

    # Display interactive menu and get user selections
    config = downloader.display_menu()

    if not config:
        print("No valid configuration. Exiting.")
        return

    # Confirm before downloading
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Missions: {config['missions'] or 'All'}")
    print(f"Planets: {config['planets'] or 'All'}")
    print(f"Targets: {config['targets'] or 'All'}")
    print(f"Total available: {config['total_available']:,} images")
    print(f"Will download: {config['max_images']:,} images")
    print(f"Skip interval: Every {config['skip_n']} image(s)")
    print(f"Output directory: {downloader.output_dir}")
    print("="*80)

    confirm = input("\nProceed with download? (yes/no): ").strip().lower()

    if confirm not in ['yes', 'y']:
        print("Download cancelled.")
        return

    # Start download
    downloader.download_images(config)

    # Generate summary
    downloader.generate_summary()

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print(f"Images saved to: {downloader.images_dir}")
    print(f"Metadata saved to: {downloader.metadata_dir}")
    print(f"Log file: {downloader.log_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user. Exiting...")
        sys.exit(0)
