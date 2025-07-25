#!/usr/bin/env python3
"""
Enhanced Constellation Data Module

This module provides comprehensive constellation line definitions and star coordinates
for use with the constellation annotator. It includes more constellations and stars
than the basic version.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class EnhancedConstellationData:
    """Enhanced constellation data with comprehensive star catalogs."""
    
    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize constellation data.
        
        Args:
            data_file: Optional path to JSON file with constellation data
        """
        if data_file and os.path.exists(data_file):
            self.constellation_lines, self.bright_stars = self._load_from_file(data_file)
        else:
            self.constellation_lines = self._load_constellation_lines()
            self.bright_stars = self._load_bright_stars()
    
    def _load_from_file(self, data_file: str) -> Tuple[Dict, Dict]:
        """Load constellation data from JSON file."""
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            return data.get('constellation_lines', {}), data.get('bright_stars', {})
        except Exception as e:
            print(f"Error loading constellation data from {data_file}: {e}")
            return self._load_constellation_lines(), self._load_bright_stars()
    
    def _load_constellation_lines(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load comprehensive constellation line definitions.
        Returns a dictionary mapping constellation names to lists of star pairs.
        """
        return {
            # Northern Hemisphere Constellations
            'Ursa Major': [
                ('Dubhe', 'Merak'),
                ('Merak', 'Phecda'),
                ('Phecda', 'Megrez'),
                ('Megrez', 'Alioth'),
                ('Alioth', 'Mizar'),
                ('Mizar', 'Alkaid'),
                ('Dubhe', 'Phecda'),
                ('Merak', 'Megrez'),
            ],
            'Ursa Minor': [
                ('Polaris', 'Kochab'),
                ('Kochab', 'Pherkad'),
                ('Pherkad', 'Yildun'),
                ('Yildun', 'Polaris'),
                ('Polaris', 'Pherkad'),
            ],
            'Cassiopeia': [
                ('Schedar', 'Caph'),
                ('Caph', 'Cih'),
                ('Cih', 'Ruchbah'),
                ('Ruchbah', 'Segin'),
                ('Segin', 'Schedar'),
            ],
            'Cepheus': [
                ('Alderamin', 'Alfirk'),
                ('Alfirk', 'Errai'),
                ('Errai', 'Alrai'),
                ('Alrai', 'Alderamin'),
                ('Alderamin', 'Errai'),
            ],
            'Draco': [
                ('Thuban', 'Rastaban'),
                ('Rastaban', 'Eltanin'),
                ('Eltanin', 'Altais'),
                ('Altais', 'Aldibain'),
                ('Aldibain', 'Thuban'),
            ],
            'Lyra': [
                ('Vega', 'Sheliak'),
                ('Sheliak', 'Sulafat'),
                ('Sulafat', 'Vega'),
                ('Vega', 'Delta Lyrae'),
                ('Delta Lyrae', 'Sheliak'),
            ],
            'Cygnus': [
                ('Deneb', 'Sadr'),
                ('Sadr', 'Gienah'),
                ('Gienah', 'Delta Cygni'),
                ('Delta Cygni', 'Albireo'),
                ('Albireo', 'Sadr'),
                ('Deneb', 'Albireo'),
            ],
            'Perseus': [
                ('Mirfak', 'Algol'),
                ('Algol', 'Rho Persei'),
                ('Rho Persei', 'Mirfak'),
                ('Mirfak', 'Atik'),
                ('Atik', 'Algol'),
            ],
            'Andromeda': [
                ('Alpheratz', 'Mirach'),
                ('Mirach', 'Almach'),
                ('Almach', 'Delta Andromedae'),
                ('Delta Andromedae', 'Mirach'),
            ],
            'Pegasus': [
                ('Markab', 'Scheat'),
                ('Scheat', 'Algenib'),
                ('Algenib', 'Alpheratz'),
                ('Alpheratz', 'Markab'),
                ('Markab', 'Enif'),
                ('Enif', 'Scheat'),
            ],
            
            # Zodiacal Constellations
            'Aries': [
                ('Hamal', 'Sheratan'),
                ('Sheratan', 'Mesarthim'),
                ('Mesarthim', 'Hamal'),
            ],
            'Taurus': [
                ('Aldebaran', 'Elnath'),
                ('Elnath', 'Alcyone'),
                ('Alcyone', 'Atlas'),
                ('Atlas', 'Pleiades'),
                ('Pleiades', 'Aldebaran'),
            ],
            'Gemini': [
                ('Castor', 'Pollux'),
                ('Pollux', 'Alhena'),
                ('Alhena', 'Castor'),
                ('Castor', 'Mebsuta'),
                ('Mebsuta', 'Pollux'),
            ],
            'Cancer': [
                ('Acubens', 'Asellus Borealis'),
                ('Asellus Borealis', 'Asellus Australis'),
                ('Asellus Australis', 'Acubens'),
            ],
            'Leo': [
                ('Regulus', 'Denebola'),
                ('Denebola', 'Algieba'),
                ('Algieba', 'Regulus'),
                ('Regulus', 'Zosma'),
                ('Zosma', 'Denebola'),
            ],
            'Virgo': [
                ('Spica', 'Vindemiatrix'),
                ('Vindemiatrix', 'Porrima'),
                ('Porrima', 'Spica'),
                ('Spica', 'Auva'),
                ('Auva', 'Vindemiatrix'),
            ],
            'Libra': [
                ('Zubenelgenubi', 'Zubeneschamali'),
                ('Zubeneschamali', 'Brachium'),
                ('Brachium', 'Zubenelgenubi'),
            ],
            'Scorpius': [
                ('Antares', 'Shaula'),
                ('Shaula', 'Lesath'),
                ('Lesath', 'Dschubba'),
                ('Dschubba', 'Antares'),
                ('Antares', 'Acrab'),
                ('Acrab', 'Dschubba'),
            ],
            'Sagittarius': [
                ('Kaus Australis', 'Kaus Media'),
                ('Kaus Media', 'Kaus Borealis'),
                ('Kaus Borealis', 'Nunki'),
                ('Nunki', 'Ascella'),
                ('Ascella', 'Kaus Australis'),
            ],
            'Capricornus': [
                ('Deneb Algedi', 'Dabih'),
                ('Dabih', 'Nashira'),
                ('Nashira', 'Deneb Algedi'),
            ],
            'Aquarius': [
                ('Sadalsuud', 'Sadalmelik'),
                ('Sadalmelik', 'Sadachbia'),
                ('Sadachbia', 'Sadalsuud'),
            ],
            'Pisces': [
                ('Alrescha', 'Alpherg'),
                ('Alpherg', 'Torcularis'),
                ('Torcularis', 'Alrescha'),
            ],
            
            # Southern Hemisphere Constellations
            'Orion': [
                ('Betelgeuse', 'Bellatrix'),
                ('Bellatrix', 'Mintaka'),
                ('Mintaka', 'Alnilam'),
                ('Alnilam', 'Alnitak'),
                ('Alnitak', 'Saiph'),
                ('Saiph', 'Rigel'),
                ('Rigel', 'Betelgeuse'),
                ('Mintaka', 'Alnitak'),  # Belt
                ('Alnilam', 'Saiph'),    # Belt
            ],
            'Canis Major': [
                ('Sirius', 'Mirzam'),
                ('Mirzam', 'Wezen'),
                ('Wezen', 'Adhara'),
                ('Adhara', 'Sirius'),
            ],
            'Canis Minor': [
                ('Procyon', 'Gomeisa'),
                ('Gomeisa', 'Procyon'),
            ],
            'Lepus': [
                ('Arneb', 'Nihal'),
                ('Nihal', 'Arneb'),
            ],
            'Eridanus': [
                ('Achernar', 'Cursa'),
                ('Cursa', 'Zaurak'),
                ('Zaurak', 'Achernar'),
            ],
            'Centaurus': [
                ('Alpha Centauri', 'Hadar'),
                ('Hadar', 'Menkent'),
                ('Menkent', 'Alpha Centauri'),
            ],
            'Crux': [
                ('Acrux', 'Mimosa'),
                ('Mimosa', 'Gacrux'),
                ('Gacrux', 'Acrux'),
                ('Acrux', 'Delta Crucis'),
                ('Delta Crucis', 'Mimosa'),
            ],
            'Carina': [
                ('Canopus', 'Miaplacidus'),
                ('Miaplacidus', 'Avior'),
                ('Avior', 'Canopus'),
            ],
            'Vela': [
                ('Suhail', 'Markeb'),
                ('Markeb', 'Suhail'),
            ],
            'Puppis': [
                ('Naos', 'Tureis'),
                ('Tureis', 'Naos'),
            ],
            'Pyxis': [
                ('Alpha Pyxidis', 'Beta Pyxidis'),
                ('Beta Pyxidis', 'Alpha Pyxidis'),
            ],
            'Antlia': [
                ('Alpha Antliae', 'Beta Antliae'),
                ('Beta Antliae', 'Alpha Antliae'),
            ],
            'Hydra': [
                ('Alphard', 'Gamma Hydrae'),
                ('Gamma Hydrae', 'Alphard'),
            ],
            'Corvus': [
                ('Alchiba', 'Kraz'),
                ('Kraz', 'Gienah'),
                ('Gienah', 'Algorab'),
                ('Algorab', 'Alchiba'),
            ],
            'Crater': [
                ('Alkes', 'Labrum'),
                ('Labrum', 'Alkes'),
            ],
        }
    
    def _load_bright_stars(self) -> Dict[str, Tuple[float, float]]:
        """
        Load comprehensive bright star coordinates (RA, Dec in degrees).
        Returns a dictionary mapping star names to (RA, Dec) tuples.
        """
        return {
            # Ursa Major
            'Dubhe': (165.9319, 61.7510),
            'Merak': (165.4603, 56.3824),
            'Phecda': (178.4577, 53.6948),
            'Megrez': (183.8565, 57.0326),
            'Alioth': (193.5073, 55.9598),
            'Mizar': (200.9814, 54.9254),
            'Alkaid': (206.8852, 49.3133),
            
            # Ursa Minor
            'Polaris': (37.9529, 89.2642),
            'Kochab': (222.6764, 74.1555),
            'Pherkad': (230.1822, 71.8340),
            'Yildun': (263.0541, 86.5864),
            
            # Cassiopeia
            'Schedar': (10.1268, 56.5373),
            'Caph': (2.2945, 59.1498),
            'Cih': (14.1651, 60.7167),
            'Ruchbah': (21.4538, 60.2353),
            'Segin': (28.5988, 63.6701),
            
            # Cepheus
            'Alderamin': (319.6449, 62.5856),
            'Alfirk': (322.1644, 70.5607),
            'Errai': (340.6672, 77.6323),
            'Alrai': (332.5490, 69.7889),
            
            # Draco
            'Thuban': (211.0977, 64.3758),
            'Rastaban': (262.6081, 52.3014),
            'Eltanin': (275.2645, 51.4889),
            'Altais': (288.1388, 67.6615),
            'Aldibain': (295.0244, 72.7328),
            
            # Lyra
            'Vega': (279.2347, 38.7836),
            'Sheliak': (282.5198, 33.3627),
            'Sulafat': (284.7359, 32.6896),
            'Delta Lyrae': (285.2828, 36.8986),
            
            # Cygnus
            'Deneb': (310.3580, 45.2803),
            'Sadr': (305.5571, 40.2567),
            'Gienah': (318.2341, 40.2567),
            'Delta Cygni': (308.3039, 45.1313),
            'Albireo': (292.6804, 27.9597),
            
            # Perseus
            'Mirfak': (51.0807, 49.8612),
            'Algol': (47.0422, 40.9556),
            'Rho Persei': (46.1991, 38.8403),
            'Atik': (56.0489, 31.8836),
            
            # Andromeda
            'Alpheratz': (2.0969, 29.0904),
            'Mirach': (17.4330, 35.6206),
            'Almach': (30.9748, 42.3297),
            'Delta Andromedae': (30.7150, 30.8612),
            
            # Pegasus
            'Markab': (346.1902, 15.2053),
            'Scheat': (341.6718, 28.0828),
            'Algenib': (5.4383, 15.1836),
            'Enif': (326.7601, 9.8750),
            
            # Zodiacal Constellations
            'Hamal': (31.7933, 23.4624),
            'Sheratan': (28.6600, 20.8080),
            'Mesarthim': (28.3828, 19.2939),
            
            'Aldebaran': (68.9802, 16.5093),
            'Elnath': (81.5728, 28.6075),
            'Alcyone': (56.8711, 24.1051),
            'Atlas': (56.0500, 24.0533),
            'Pleiades': (56.7500, 24.1167),
            
            'Castor': (113.6495, 31.8883),
            'Pollux': (116.3289, 28.0262),
            'Alhena': (99.4276, 16.3993),
            'Mebsuta': (100.9830, 25.1311),
            
            'Acubens': (134.6215, 11.8577),
            'Asellus Borealis': (130.8214, 21.4685),
            'Asellus Australis': (131.1713, 18.1543),
            
            'Regulus': (152.0929, 11.9672),
            'Denebola': (177.2649, 14.5720),
            'Algieba': (154.9931, 19.8415),
            'Zosma': (168.5270, 20.5237),
            
            'Spica': (201.2983, -11.1613),
            'Vindemiatrix': (184.9765, 10.9592),
            'Porrima': (190.4151, -1.4494),
            'Auva': (204.9720, -6.0006),
            
            'Zubenelgenubi': (222.7196, -16.0418),
            'Zubeneschamali': (229.2517, -9.3829),
            'Brachium': (233.8819, -25.2819),
            
            'Antares': (247.3519, -26.4320),
            'Shaula': (263.4022, -37.1038),
            'Lesath': (264.3297, -37.2958),
            'Dschubba': (240.0833, -22.6217),
            'Acrab': (241.3592, -19.8054),
            
            'Kaus Australis': (276.0430, -34.3846),
            'Kaus Media': (274.4067, -29.8281),
            'Kaus Borealis': (271.4520, -25.4217),
            'Nunki': (283.8164, -26.2967),
            'Ascella': (287.4407, -29.8811),
            
            'Deneb Algedi': (326.7601, -16.1273),
            'Dabih': (305.2528, -14.7814),
            'Nashira': (326.0361, -16.6623),
            
            'Sadalsuud': (322.8896, -5.5711),
            'Sadalmelik': (331.4459, -0.3198),
            'Sadachbia': (349.3579, -1.3873),
            
            'Alrescha': (23.6264, 2.7636),
            'Alpherg': (28.6600, 15.3458),
            'Torcularis': (28.3828, 3.8203),
            
            # Southern Hemisphere
            'Betelgeuse': (88.7929, 7.4071),
            'Bellatrix': (81.2828, 6.3497),
            'Mintaka': (83.0016, -0.2991),
            'Alnilam': (84.0534, -1.2019),
            'Alnitak': (85.1897, -1.9426),
            'Saiph': (86.9391, -9.6696),
            'Rigel': (78.6345, -8.2016),
            
            'Sirius': (101.2872, -16.7161),
            'Mirzam': (95.6748, -17.9559),
            'Wezen': (107.0978, -26.3932),
            'Adhara': (111.0238, -28.9721),
            
            'Procyon': (114.8255, 5.2250),
            'Gomeisa': (111.7877, 8.2893),
            
            'Arneb': (82.0614, -17.8223),
            'Nihal': (82.0614, -20.7594),
            
            'Achernar': (24.4285, -57.2368),
            'Cursa': (76.6284, -5.0864),
            'Zaurak': (76.6284, -13.5085),
            
            'Alpha Centauri': (219.8731, -60.8322),
            'Hadar': (210.9559, -60.3730),
            'Menkent': (204.9719, -36.3699),
            
            'Acrux': (186.6495, -63.0991),
            'Mimosa': (191.9303, -59.6888),
            'Gacrux': (187.7915, -57.1133),
            'Delta Crucis': (183.7863, -58.7489),
            
            'Canopus': (95.9880, -52.6957),
            'Miaplacidus': (138.2999, -69.7172),
            'Avior': (125.6284, -59.5095),
            
            'Suhail': (128.1359, -43.4326),
            'Markeb': (140.5284, -55.0107),
            
            'Naos': (120.8960, -40.0031),
            'Tureis': (116.3127, -28.2166),
            
            'Alpha Pyxidis': (134.7054, -33.1864),
            'Beta Pyxidis': (131.1759, -35.3083),
            
            'Alpha Antliae': (157.2345, -31.0678),
            'Beta Antliae': (154.3912, -37.1373),
            
            'Alphard': (141.8968, -8.6586),
            'Gamma Hydrae': (138.5912, -23.1714),
            
            'Alchiba': (186.7346, -24.7285),
            'Kraz': (189.2956, -23.3967),
            'Gienah': (194.0071, -17.5419),
            'Algorab': (191.9303, -16.5151),
            
            'Alkes': (164.9436, -18.2988),
            'Labrum': (168.5200, -18.1411),
        }
    
    def get_constellation_stars(self, constellation_name: str) -> List[Tuple[str, str]]:
        """Get the star pairs that form lines for a given constellation."""
        return self.constellation_lines.get(constellation_name, [])
    
    def get_star_coordinates(self, star_name: str) -> Optional[Tuple[float, float]]:
        """Get the RA/Dec coordinates for a given star."""
        return self.bright_stars.get(star_name)
    
    def get_all_constellations(self) -> List[str]:
        """Get list of all available constellations."""
        return list(self.constellation_lines.keys())
    
    def save_to_file(self, filename: str) -> bool:
        """Save constellation data to JSON file."""
        try:
            data = {
                'constellation_lines': self.constellation_lines,
                'bright_stars': self.bright_stars
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving constellation data: {e}")
            return False
    
    def get_constellations_by_hemisphere(self, hemisphere: str = 'both') -> List[str]:
        """
        Get constellations by hemisphere.
        
        Args:
            hemisphere: 'north', 'south', or 'both'
            
        Returns:
            List of constellation names
        """
        northern = [
            'Ursa Major', 'Ursa Minor', 'Cassiopeia', 'Cepheus', 'Draco',
            'Lyra', 'Cygnus', 'Perseus', 'Andromeda', 'Pegasus'
        ]
        
        southern = [
            'Orion', 'Canis Major', 'Canis Minor', 'Lepus', 'Eridanus',
            'Centaurus', 'Crux', 'Carina', 'Vela', 'Puppis', 'Pyxis',
            'Antlia', 'Hydra', 'Corvus', 'Crater'
        ]
        
        zodiacal = [
            'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
            'Libra', 'Scorpius', 'Sagittarius', 'Capricornus', 'Aquarius', 'Pisces'
        ]
        
        if hemisphere == 'north':
            return northern + zodiacal
        elif hemisphere == 'south':
            return southern + zodiacal
        else:
            return northern + southern + zodiacal

# Example usage
if __name__ == "__main__":
    # Create enhanced constellation data
    data = EnhancedConstellationData()
    
    print(f"Loaded {len(data.get_all_constellations())} constellations")
    print(f"Loaded {len(data.bright_stars)} bright stars")
    
    # Save to file for future use
    data.save_to_file("constellation_data.json")
    print("Saved constellation data to constellation_data.json") 