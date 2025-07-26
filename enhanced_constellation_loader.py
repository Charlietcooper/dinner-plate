#!/usr/bin/env python3
"""
Enhanced Constellation Loader - Parses comprehensive constellation data
Based on Martin Krzywinski's constellation database
"""

import requests
import re
from typing import Dict, List, Tuple, Optional
import json
import os

class EnhancedConstellationLoader:
    """Loads comprehensive constellation data from professional sources."""
    
    def __init__(self):
        self.constellation_data_url = "https://mk.bcgsc.ca/constellations/constellation_figures.txt"
        self.constellations = {}
        self.stars = {}
        
        # Alternative URLs to try
        self.alternative_urls = [
            "https://mk.bcgsc.ca/constellations/constellation_figures.txt",
            "https://mk.bcgsc.ca/constellations/constellation_figures",
            "https://mk.bcgsc.ca/constellations/constellation_figures.dat"
        ]
    
    def download_constellation_data(self) -> bool:
        """Download the comprehensive constellation data."""
        print("📥 Downloading constellation data from Martin Krzywinski's database...")
        
        try:
            response = requests.get(self.constellation_data_url, timeout=30)
            response.raise_for_status()
            
            # Save raw data
            with open("Processing/constellation_data_raw.txt", "w") as f:
                f.write(response.text)
            
            print(f"✅ Downloaded {len(response.text)} characters of constellation data")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download constellation data: {e}")
            return False
    
    def parse_constellation_data(self) -> Dict:
        """Parse the constellation data file."""
        print("🔍 Parsing constellation data...")
        
        try:
            with open("Processing/constellation_data_raw.txt", "r") as f:
                data = f.read()
            
            # Parse stars
            self.stars = self._parse_stars(data)
            print(f"   Found {len(self.stars)} stars")
            
            # Parse constellation paths
            self.constellations = self._parse_constellation_paths(data)
            print(f"   Found {len(self.constellations)} constellations")
            
            return {
                "stars": self.stars,
                "constellations": self.constellations
            }
            
        except Exception as e:
            print(f"❌ Failed to parse constellation data: {e}")
            return {}
    
    def _parse_stars(self, data: str) -> Dict:
        """Parse star entries from the data."""
        stars = {}
        
        # Find star entries (lines starting with 'star')
        star_lines = [line for line in data.split('\n') if line.startswith('star ')]
        
        for line in star_lines:
            parts = line.split()
            if len(parts) >= 12:
                try:
                    hr_number = parts[1]
                    constellation = parts[2]
                    bayer_letter = parts[5] if parts[5] != '-' else None
                    bayer_greek = parts[6] if parts[6] != '-' else None
                    flamsteed = parts[8] if parts[8] != '-' else None
                    name = parts[10] if parts[10] != '-' else None
                    ra = float(parts[11])
                    dec = float(parts[12])
                    magnitude = float(parts[14]) if parts[14] != '-' else None
                    spectral_type = parts[15] if parts[15] != '-' else None
                    
                    stars[hr_number] = {
                        "hr_number": hr_number,
                        "constellation": constellation,
                        "bayer_letter": bayer_letter,
                        "bayer_greek": bayer_greek,
                        "flamsteed": flamsteed,
                        "name": name,
                        "ra": ra,
                        "dec": dec,
                        "magnitude": magnitude,
                        "spectral_type": spectral_type
                    }
                except (ValueError, IndexError):
                    continue
        
        return stars
    
    def _parse_constellation_paths(self, data: str) -> Dict:
        """Parse constellation figure paths."""
        constellations = {}
        
        # Find constellation path entries
        conpath_lines = [line for line in data.split('\n') if line.startswith('conpath ')]
        
        for line in conpath_lines:
            parts = line.split(' ', 2)  # Split into 'conpath', constellation_name, and path_data
            if len(parts) >= 3:
                constellation_name = parts[1]
                path_data = parts[2]
                
                # Parse the path connections
                connections = self._parse_path_connections(path_data)
                
                constellations[constellation_name] = {
                    "name": constellation_name,
                    "connections": connections,
                    "stars": self._get_constellation_stars(constellation_name)
                }
        
        return constellations
    
    def _parse_path_connections(self, path_data: str) -> List[Tuple[str, str]]:
        """Parse constellation path connections."""
        connections = []
        
        # Split by commas and parse each connection
        connection_parts = path_data.split(',')
        
        for part in connection_parts:
            part = part.strip()
            if '-' in part:
                # Format: "HR1(star1)-connection_type-HR2(star2)"
                match = re.match(r'(\d+)\s*\([^)]*\)\s*-\s*\d+\s*-\s*(\d+)\s*\([^)]*\)', part)
                if match:
                    star1_hr = match.group(1)
                    star2_hr = match.group(2)
                    connections.append((star1_hr, star2_hr))
        
        return connections
    
    def _get_constellation_stars(self, constellation_name: str) -> List[Dict]:
        """Get all stars for a specific constellation."""
        constellation_stars = []
        
        for hr_number, star_data in self.stars.items():
            if star_data["constellation"] == constellation_name:
                constellation_stars.append(star_data)
        
        return constellation_stars
    
    def create_enhanced_constellation_database(self) -> Dict:
        """Create an enhanced constellation database with complete data."""
        print("🏗️ Creating enhanced constellation database...")
        
        enhanced_constellations = {}
        
        for const_name, const_data in self.constellations.items():
            print(f"   Processing {const_name}...")
            
            # Get stars for this constellation
            const_stars = const_data["stars"]
            
            # Create star list with proper format
            stars_list = []
            for star in const_stars:
                stars_list.append({
                    "name": star["name"] or f"{star['bayer_greek'] or star['bayer_letter'] or star['hr_number']}",
                    "ra": star["ra"],
                    "dec": star["dec"],
                    "mag": star["magnitude"] or 5.0,
                    "hr_number": star["hr_number"],
                    "bayer": star["bayer_greek"] or star["bayer_letter"],
                    "flamsteed": star["flamsteed"]
                })
            
            # Create line connections
            lines = []
            for star1_hr, star2_hr in const_data["connections"]:
                star1_name = self._get_star_name_by_hr(star1_hr)
                star2_name = self._get_star_name_by_hr(star2_hr)
                if star1_name and star2_name:
                    lines.append((star1_name, star2_name))
            
            enhanced_constellations[const_name] = {
                "name": const_name,
                "description": f"IAU constellation {const_name}",
                "stars": stars_list,
                "lines": lines,
                "total_stars": len(stars_list),
                "total_connections": len(lines)
            }
        
        return enhanced_constellations
    
    def _get_star_name_by_hr(self, hr_number: str) -> Optional[str]:
        """Get star name by HR number."""
        if hr_number in self.stars:
            star = self.stars[hr_number]
            return star["name"] or star["bayer_greek"] or star["bayer_letter"] or hr_number
        return None
    
    def save_enhanced_database(self, enhanced_constellations: Dict) -> bool:
        """Save the enhanced constellation database."""
        try:
            # Save as JSON
            with open("Processing/enhanced_constellation_database.json", "w") as f:
                json.dump(enhanced_constellations, f, indent=2)
            
            # Create summary
            summary = {
                "total_constellations": len(enhanced_constellations),
                "total_stars": sum(len(const["stars"]) for const in enhanced_constellations.values()),
                "constellations": list(enhanced_constellations.keys())
            }
            
            with open("Processing/constellation_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            print(f"✅ Saved enhanced database with {len(enhanced_constellations)} constellations")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save enhanced database: {e}")
            return False

def main():
    """Load and process comprehensive constellation data."""
    print("🌟 Enhanced Constellation Data Loader")
    print("=" * 50)
    
    # Create loader
    loader = EnhancedConstellationLoader()
    
    # Download data
    if not loader.download_constellation_data():
        print("❌ Cannot proceed without constellation data")
        return
    
    # Parse data
    parsed_data = loader.parse_constellation_data()
    if not parsed_data:
        print("❌ Failed to parse constellation data")
        return
    
    # Create enhanced database
    enhanced_constellations = loader.create_enhanced_constellation_database()
    
    # Save database
    if loader.save_enhanced_database(enhanced_constellations):
        print(f"\n📊 Enhanced Database Summary:")
        print(f"   Total constellations: {len(enhanced_constellations)}")
        print(f"   Total stars: {sum(len(const['stars']) for const in enhanced_constellations.values())}")
        
        print(f"\n🌌 Sample constellations:")
        for i, (name, data) in enumerate(list(enhanced_constellations.items())[:10]):
            print(f"   {i+1:2d}. {name}: {data['total_stars']} stars, {data['total_connections']} connections")
        
        print(f"\n📁 Files created:")
        print(f"   - Processing/enhanced_constellation_database.json")
        print(f"   - Processing/constellation_summary.json")
        print(f"   - Processing/constellation_data_raw.txt")
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Use this data to replace our basic constellation database")
        print(f"   2. Create visualizations for all 88 constellations")
        print(f"   3. Implement shape-preserving fitting with complete data")
        
        return True
    else:
        print("❌ Failed to save enhanced database")
        return False

if __name__ == "__main__":
    main() 