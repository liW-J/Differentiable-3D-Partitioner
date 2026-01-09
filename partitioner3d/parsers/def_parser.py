"""
DEF (Design Exchange Format) Parser
Parses DEF files for VLSI design data
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional

# Constants
DEFAULT_CELL_SIZE_FACTOR = 100  # Factor for cell size estimation
DEFAULT_CELL_SIZE = 100  # Default cell size when die area not available


class DEFParser:
    """
    Parser for DEF (Design Exchange Format) files.
    
    Extracts component positions, sizes, and other design information
    from DEF files commonly used in VLSI design.
    """
    
    def __init__(self):
        self.design_name = None
        self.die_area = None
        self.components = []
        self.pins = []
        self.nets = []
        self.units_distance_microns = 1000  # Default DEF units
        
    def parse_file(self, filepath: str) -> Dict:
        """
        Parse a DEF file.
        
        Args:
            filepath: Path to the DEF file
            
        Returns:
            Dictionary containing parsed design data
        """
        with open(filepath, 'r') as f:
            content = f.read()
            
        return self.parse_content(content)
        
    def parse_content(self, content: str) -> Dict:
        """
        Parse DEF content.
        
        Args:
            content: DEF file content as string
            
        Returns:
            Dictionary containing parsed design data
        """
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
                
            # Parse design name
            if line.startswith('DESIGN'):
                self.design_name = line.split()[1]
                
            # Parse units
            elif line.startswith('UNITS DISTANCE MICRONS'):
                match = re.search(r'MICRONS\s+(\d+)', line)
                if match:
                    self.units_distance_microns = int(match.group(1))
                    
            # Parse die area
            elif line.startswith('DIEAREA'):
                self._parse_die_area(line)
                
            # Parse components
            elif line.startswith('COMPONENTS'):
                i = self._parse_components(lines, i)
                
            # Parse pins
            elif line.startswith('PINS'):
                i = self._parse_pins(lines, i)
                
            # Parse nets
            elif line.startswith('NETS'):
                i = self._parse_nets(lines, i)
                
            i += 1
            
        return self.get_parsed_data()
        
    def _parse_die_area(self, line: str):
        """Parse DIEAREA line."""
        # DIEAREA ( x1 y1 ) ( x2 y2 )
        matches = re.findall(r'\(\s*(-?\d+)\s+(-?\d+)\s*\)', line)
        if len(matches) >= 2:
            x1, y1 = int(matches[0][0]), int(matches[0][1])
            x2, y2 = int(matches[1][0]), int(matches[1][1])
            self.die_area = {
                'lower_left': (x1, y1),
                'upper_right': (x2, y2),
                'width': x2 - x1,
                'height': y2 - y1
            }
            
    def _parse_components(self, lines: List[str], start_idx: int) -> int:
        """Parse COMPONENTS section."""
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('END COMPONENTS'):
                return i
                
            if line.startswith('-'):
                component = self._parse_component_line(line)
                if component:
                    self.components.append(component)
                    
            i += 1
            
        return i
        
    def _parse_component_line(self, line: str) -> Optional[Dict]:
        """Parse a single component line."""
        # - comp_name cell_type + PLACED ( x y ) orientation ;
        parts = line.split()
        
        if len(parts) < 2:
            return None
            
        component = {
            'name': parts[1],
            'cell_type': parts[2] if len(parts) > 2 else 'UNKNOWN',
            'position': None,
            'orientation': None,
            'status': None,
            'layer': 0  # Default layer
        }
        
        # Look for placement information
        if 'PLACED' in parts or 'FIXED' in parts:
            status_idx = parts.index('PLACED') if 'PLACED' in parts else parts.index('FIXED')
            component['status'] = parts[status_idx]
            
            # Find position coordinates
            for i in range(status_idx, len(parts) - 2):
                if parts[i] == '(':
                    try:
                        x = int(parts[i + 1])
                        y = int(parts[i + 2])
                        component['position'] = (x, y)
                        
                        # Look for orientation
                        if i + 4 < len(parts) and parts[i + 3] == ')':
                            component['orientation'] = parts[i + 4]
                            
                        break
                    except (ValueError, IndexError):
                        pass
                        
        return component
        
    def _parse_pins(self, lines: List[str], start_idx: int) -> int:
        """Parse PINS section."""
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('END PINS'):
                return i
                
            if line.startswith('-'):
                pin = self._parse_pin_line(line)
                if pin:
                    self.pins.append(pin)
                    
            i += 1
            
        return i
        
    def _parse_pin_line(self, line: str) -> Optional[Dict]:
        """Parse a single pin line."""
        parts = line.split()
        
        if len(parts) < 2:
            return None
            
        pin = {
            'name': parts[1],
            'net': None,
            'direction': None,
            'position': None,
            'layer': None
        }
        
        # Extract net name
        if 'NET' in parts:
            net_idx = parts.index('NET')
            if net_idx + 1 < len(parts):
                pin['net'] = parts[net_idx + 1]
                
        # Extract direction
        if 'DIRECTION' in parts:
            dir_idx = parts.index('DIRECTION')
            if dir_idx + 1 < len(parts):
                pin['direction'] = parts[dir_idx + 1]
                
        # Extract position if FIXED or PLACED
        if 'FIXED' in parts or 'PLACED' in parts:
            for i, part in enumerate(parts):
                if part == '(' and i + 2 < len(parts):
                    try:
                        x = int(parts[i + 1])
                        y = int(parts[i + 2])
                        pin['position'] = (x, y)
                        break
                    except ValueError:
                        pass
                        
        return pin
        
    def _parse_nets(self, lines: List[str], start_idx: int) -> int:
        """Parse NETS section."""
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('END NETS'):
                return i
                
            if line.startswith('-'):
                net = self._parse_net_line(line)
                if net:
                    self.nets.append(net)
                    
            i += 1
            
        return i
        
    def _parse_net_line(self, line: str) -> Optional[Dict]:
        """Parse a single net line."""
        parts = line.split()
        
        if len(parts) < 2:
            return None
            
        net = {
            'name': parts[1],
            'connections': []
        }
        
        # Parse connections (component pin pairs)
        i = 0
        while i < len(parts):
            if parts[i] == '(' and i + 2 < len(parts):
                component = parts[i + 1]
                pin = parts[i + 2]
                net['connections'].append({
                    'component': component,
                    'pin': pin
                })
                i += 3
            else:
                i += 1
                
        return net
        
    def get_parsed_data(self) -> Dict:
        """
        Get all parsed data.
        
        Returns:
            Dictionary containing all parsed design information
        """
        return {
            'design_name': self.design_name,
            'die_area': self.die_area,
            'units': self.units_distance_microns,
            'components': self.components,
            'pins': self.pins,
            'nets': self.nets
        }
        
    def get_component_positions(self) -> np.ndarray:
        """
        Extract component positions as numpy array.
        
        Returns:
            Nx3 array of (x, y, z) positions where z is layer/tier
        """
        positions = []
        
        for comp in self.components:
            if comp['position'] is not None:
                x, y = comp['position']
                z = comp.get('layer', 0)
                positions.append([x, y, z])
                
        return np.array(positions) if positions else np.zeros((0, 3))
        
    def get_pin_positions(self) -> np.ndarray:
        """
        Extract pin/terminal positions as numpy array.
        
        Returns:
            Mx3 array of (x, y, z) positions
        """
        positions = []
        
        for pin in self.pins:
            if pin['position'] is not None:
                x, y = pin['position']
                z = 0  # Pins typically on layer 0
                positions.append([x, y, z])
                
        return np.array(positions) if positions else np.zeros((0, 3))
        
    def estimate_component_sizes(self) -> np.ndarray:
        """
        Estimate component sizes based on cell types.
        
        Returns:
            Nx3 array of (width, height, depth) sizes
        """
        # Simple heuristic: estimate size based on die area
        if self.die_area and len(self.components) > 0:
            avg_size = np.sqrt(
                (self.die_area['width'] * self.die_area['height']) /
                (len(self.components) * DEFAULT_CELL_SIZE_FACTOR)
            )
        else:
            avg_size = DEFAULT_CELL_SIZE
            
        sizes = np.ones((len(self.components), 3)) * avg_size
        sizes[:, 2] = 1  # Unit depth
        
        return sizes
