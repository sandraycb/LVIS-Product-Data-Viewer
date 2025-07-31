#!/usr/bin/env python3
"""
LVIS HDF5 Data Server with Coverage Polygon Generation

A Flask-based server for visualizing LVIS L1B and L2 data products. Generates
convex hull and detailed boundary polygons for web-based visualization.

Features:
- Serves LVIS L1B HDF5 files via REST API
- Generates coverage polygons (convex hull and detailed boundaries)
- Integrates optional L2 elevation metrics

Usage:
    python lvis-server.py [options]
    
    Options:
        --skip-polygon-gen    Skip polygon generation at startup
        --force-polygon-gen   Force regenerate all polygons
        --port PORT          Port to run server on (default: 5000)
        --buffer-size SIZE   Buffer size for detailed polygons in meters (default: 10)
        --max-markers N      Maximum markers per view (default: 10000)

Author: Sandra Yaacoub, Paul Montesano
Version: 1.0.1
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from shapely.geometry import Point, mapping, MultiPoint
from shapely.ops import unary_union

# ===== Configuration Constants =====
DEFAULT_PORT = 5000
DEFAULT_BUFFER_METERS = 10
MAX_MARKERS_PER_VIEW = 10000

# ===== Global Variables =====
app = Flask(__name__)
CORS(app)  # Enable CORS for local development
DATA_FOLDER = None
L2_FOLDER = None
PORT = DEFAULT_PORT
BUFFER_METERS = DEFAULT_BUFFER_METERS
MAX_MARKERS = MAX_MARKERS_PER_VIEW
REGENERATE_POLYGONS = False

# ===== Utility Functions =====

def calculate_bounds_from_geojson(geojson: Dict) -> Dict[str, float]:
    """
    Extract geographic bounds from a GeoJSON polygon.
    
    Args:
        geojson: GeoJSON feature with polygon geometry
        
    Returns:
        Dictionary with minLat, maxLat, minLon, maxLon
    """
    coords = geojson['geometry']['coordinates']
    
    min_lat = float('inf')
    max_lat = float('-inf')
    min_lon = float('inf')
    max_lon = float('-inf')
    
    def process_coords(coord_list):
        nonlocal min_lat, max_lat, min_lon, max_lon
        for coord in coord_list:
            if isinstance(coord[0], list):  # Nested coordinates
                process_coords(coord)
            else:
                lon, lat = coord[0], coord[1]
                min_lon = min(min_lon, lon)
                max_lon = max(max_lon, lon)
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)
    
    process_coords(coords)
    
    return {
        'minLat': min_lat,
        'maxLat': max_lat,
        'minLon': min_lon,
        'maxLon': max_lon
    }

def should_generate_polygon(h5_path: str, polygon_path: str) -> bool:
    """
    Determine if a polygon file should be generated.
    
    Args:
        h5_path: Path to HDF5 file
        polygon_path: Path to polygon file
        
    Returns:
        True if polygon should be generated
    """
    if REGENERATE_POLYGONS:
        return True
    
    return not os.path.exists(polygon_path)

# ===== Polygon Generation Functions =====

def create_coverage_polygon(h5_path: str, polygon_path: str) -> None:
    """
    Generate a convex hull polygon from all LVIS shot points.
    
    Args:
        h5_path: Path to LVIS HDF5 file
        polygon_path: Path to save GeoJSON polygon
        
    Raises:
        ValueError: If no valid points found or polygon is empty
    """
    with h5py.File(h5_path, 'r') as f:
        # Load shot locations
        lat0 = f['LAT0'][:]
        lon0 = f['LON0'][:]
        lat1215 = f['LAT1215'][:]
        lon1215 = f['LON1215'][:]
        
        # Calculate center points
        lats = (lat0 + lat1215) / 2
        lons = (lon0 + lon1215) / 2
        
        # Convert longitude if needed
        lons = np.where(lons > 180, lons - 360, lons)
        
        # Filter valid points
        valid_mask = ~(np.isnan(lats) | np.isnan(lons))
        valid_lats = lats[valid_mask]
        valid_lons = lons[valid_mask]
        
        if len(valid_lats) == 0:
            raise ValueError("No valid points found")
        
        # Create points for convex hull
        points = [(valid_lons[i], valid_lats[i]) for i in range(len(valid_lats))]
        
        # Generate convex hull
        hull = MultiPoint(points).convex_hull
        
        if hull.is_empty:
            raise ValueError("Generated polygon is empty")
        
        # Create GeoJSON
        geojson = {
            "type": "Feature",
            "properties": {
                "filename": os.path.basename(h5_path),
                "generated": datetime.now().isoformat(),
                "shot_count": int(np.sum(valid_mask)),
                "hull_type": "convex"
            },
            "geometry": mapping(hull)
        }
        
        # Save polygon
        with open(polygon_path, 'w') as f:
            json.dump(geojson, f)

def create_detailed_boundary_polygon(h5_path: str, polygon_path: str) -> None:
    """
    Generate a detailed boundary polygon by buffering and merging shot points.
    
    Args:
        h5_path: Path to LVIS HDF5 file
        polygon_path: Path to save GeoJSON polygon
        
    Raises:
        ValueError: If no valid points found or polygon is empty
    """
    with h5py.File(h5_path, 'r') as f:
        # Load shot locations
        lat0 = f['LAT0'][:]
        lon0 = f['LON0'][:]
        lat1215 = f['LAT1215'][:]
        lon1215 = f['LON1215'][:]
        
        # Calculate center points
        lats = (lat0 + lat1215) / 2
        lons = (lon0 + lon1215) / 2
        
        # Convert longitude if needed
        lons = np.where(lons > 180, lons - 360, lons)
        
        # Filter valid points
        valid_mask = ~(np.isnan(lats) | np.isnan(lons))
        valid_lats = lats[valid_mask]
        valid_lons = lons[valid_mask]
        
        if len(valid_lats) == 0:
            raise ValueError("No valid points found")
        
        # Calculate buffer in degrees based on latitude
        buffer_degrees_base = BUFFER_METERS * 0.000009
        avg_lat = np.mean(valid_lats)
        lat_factor = np.cos(np.radians(avg_lat))
        buffer_degrees = buffer_degrees_base / lat_factor
        
        # Create buffered points
        buffered_points = []
        for i in range(len(valid_lats)):
            point = Point(valid_lons[i], valid_lats[i])
            buffered = point.buffer(buffer_degrees)
            buffered_points.append(buffered)
        
        # Merge all buffered points into one polygon
        merged_polygon = unary_union(buffered_points)
        
        if merged_polygon.is_empty:
            raise ValueError("Generated detailed boundary is empty")
        
        # Create GeoJSON
        geojson = {
            "type": "Feature",
            "properties": {
                "filename": os.path.basename(h5_path),
                "generated": datetime.now().isoformat(),
                "shot_count": int(np.sum(valid_mask)),
                "boundary_type": "detailed",
                "buffer_meters": BUFFER_METERS
            },
            "geometry": mapping(merged_polygon)
        }
        
        # Save polygon
        with open(polygon_path, 'w') as f:
            json.dump(geojson, f)

def generate_coverage_polygons() -> None:
    """Generate polygon files for any H5 files that don't have them."""
    h5_files = glob.glob(os.path.join(DATA_FOLDER, "*.h5"))
    files_to_process = []
    
    # Check what needs processing
    for h5_path in h5_files:
        h5_filename = os.path.basename(h5_path)
        base_name = os.path.splitext(h5_filename)[0]
        
        convex_path = os.path.join(DATA_FOLDER, f"{base_name}_coverage.geojson")
        detailed_path = os.path.join(DATA_FOLDER, f"{base_name}_detailed.geojson")
        
        needs_convex = should_generate_polygon(h5_path, convex_path)
        needs_detailed = should_generate_polygon(h5_path, detailed_path)
        
        if needs_convex or needs_detailed:
            files_to_process.append((h5_path, convex_path, detailed_path, needs_convex, needs_detailed))
        else:
            print(f"✓ Polygons up to date for {h5_filename}")
    
    if not files_to_process:
        print("✓ All polygon files are up to date!")
        return
    
    print(f"\nFound {len(files_to_process)} files needing polygon generation")
    
    # Process files
    for i, (h5_path, convex_path, detailed_path, needs_convex, needs_detailed) in enumerate(files_to_process, 1):
        h5_filename = os.path.basename(h5_path)
        print(f"\n[{i}/{len(files_to_process)}] Processing {h5_filename}...")
        
        if needs_convex:
            try:
                create_coverage_polygon(h5_path, convex_path)
                print(f"✓ Generated convex hull for {h5_filename}")
            except Exception as e:
                print(f"✗ Error generating convex hull for {h5_filename}: {e}")
        
        if needs_detailed:
            try:
                create_detailed_boundary_polygon(h5_path, detailed_path)
                print(f"✓ Generated detailed boundary for {h5_filename}")
            except Exception as e:
                print(f"✗ Error generating detailed boundary for {h5_filename}: {e}")
    
    print("\n✓ Polygon generation complete!")

# ===== L2 Data Processing Functions =====

def find_l2_file(l1b_filename: str) -> Optional[str]:
    """
    Find matching L2 file based on ID pattern in filename.
    
    Args:
        l1b_filename: L1B filename to match
        
    Returns:
        Path to matching L2 file or None if not found
    """
    if not L2_FOLDER:
        return None
    
    # Extract ID patterns
    id_patterns = [
        r'([A-Za-z]+\d{4}_\d{4}_R\d{4}_\d{6})',  # Standard pattern
        r'([A-Z]+\d{4}_\d{4}_R\d{4}_\d{6})',      # Uppercase campaign
        r'(\w+_\d{4}_R\d{4}_\d{6})',              # Any word_date_flight_id
    ]
    
    file_id = None
    for pattern in id_patterns:
        id_match = re.search(pattern, l1b_filename)
        if id_match:
            file_id = id_match.group(1)
            break
    
    if not file_id:
        return None
    
    # Search for L2 file containing this ID
    l2_files = glob.glob(os.path.join(L2_FOLDER, "*.txt"))
    
    for l2_file in l2_files:
        if file_id in os.path.basename(l2_file):
            return l2_file
    
    return None

def read_l2_data(l2_filepath: str, shot_number: int) -> Dict:
    """
    Read L2 data for a specific shot number.
    
    Args:
        l2_filepath: Path to L2 file
        shot_number: Shot number to find
        
    Returns:
        Dictionary with L2 metrics or error information
    """
    try:
        with open(l2_filepath, 'r') as f:
            # Read header lines to find column names
            header_lines = []
            columns = None
            
            for line in f:
                if line.startswith('#'):
                    header_lines.append(line)
                else:
                    if header_lines:
                        last_header = header_lines[-1]
                        columns = last_header[1:].strip().split()
                    break
            
            if not columns:
                return {'shot_number': shot_number, 'found': False, 'error': 'No column headers found'}
            
            # Find column indices
            col_indices = {}
            for i, col in enumerate(columns):
                if col in ['SHOTNUMBER', 'ZG', 'ZT', 'ZH', 'RH10', 'RH25', 'RH50', 'RH75', 'RH95', 'RH100']:
                    col_indices[col] = i
            
            if 'SHOTNUMBER' not in col_indices or 'ZG' not in col_indices:
                return {'shot_number': shot_number, 'found': False, 'error': 'Missing required columns'}
            
            # Check first data line
            first_line = line
            values = first_line.strip().split()
            
            if len(values) > col_indices['SHOTNUMBER']:
                line_shot_number = int(values[col_indices['SHOTNUMBER']])
                if line_shot_number == shot_number:
                    return extract_l2_values(values, col_indices, shot_number)
            
            # Continue reading file
            for line in f:
                if line.startswith('#'):
                    continue
                    
                values = line.strip().split()
                if len(values) > col_indices['SHOTNUMBER']:
                    line_shot_number = int(values[col_indices['SHOTNUMBER']])
                    if line_shot_number == shot_number:
                        return extract_l2_values(values, col_indices, shot_number)
        
        return {'shot_number': shot_number, 'found': False}
        
    except Exception as e:
        return {'shot_number': shot_number, 'found': False, 'error': str(e)}

def extract_l2_values(values: List[str], col_indices: Dict[str, int], shot_number: int) -> Dict:
    """
    Extract L2 values from a data line.
    
    Args:
        values: Split data values from line
        col_indices: Column name to index mapping
        shot_number: Shot number for reference
        
    Returns:
        Dictionary with extracted L2 metrics
    """
    result = {
        'shot_number': shot_number,
        'found': True
    }
    
    # Get elevation values
    result['ZG'] = float(values[col_indices['ZG']]) if col_indices.get('ZG') is not None else None
    result['ZT'] = float(values[col_indices['ZT']]) if col_indices.get('ZT') is not None else None
    
    # ZH might not exist in all datasets
    if 'ZH' in col_indices and col_indices['ZH'] < len(values):
        try:
            result['ZH'] = float(values[col_indices['ZH']])
        except (ValueError, IndexError):
            result['ZH'] = None
    else:
        result['ZH'] = None
    
    # Get RH values and calculate elevations
    for rh_metric in ['RH10', 'RH25', 'RH50', 'RH75', 'RH95', 'RH100']:
        if rh_metric in col_indices and col_indices[rh_metric] < len(values):
            try:
                rh_value = float(values[col_indices[rh_metric]])
                result[rh_metric] = rh_value
                # Calculate elevation by adding to ZG
                if result['ZG'] is not None:
                    result[f'{rh_metric}_elevation'] = result['ZG'] + rh_value
            except (ValueError, IndexError):
                result[rh_metric] = None
    
    return result

# ===== Flask Routes =====

@app.route('/')
def index():
    """Serve the main server information page."""
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LVIS Server Running</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            .status {{ color: green; }}
        </style>
    </head>
    <body>
        <h1>LVIS Data Server Running</h1>
        <p class="status">✓ Server is active on port {PORT}</p>
        <p>L1B Data folder: <code>{DATA_FOLDER}</code></p>
        <p>L2 Data folder: <code>{L2_FOLDER if L2_FOLDER else 'Not configured'}</code></p>
        <p>Configuration:</p>
        <ul>
            <li>Buffer size: {BUFFER_METERS}m</li>
            <li>Max markers per view: {MAX_MARKERS}</li>
        </ul>
        <p>API Endpoints:</p>
        <ul>
            <li><code>/api/files</code> - List available files</li>
            <li><code>/api/files/polygons</code> - Get coverage polygons</li>
            <li><code>/api/file/{{filename}}/info</code> - Get file metadata</li>
            <li><code>/api/file/{{filename}}/polygon</code> - Get coverage polygon</li>
            <li><code>/api/file/{{filename}}/detailed</code> - Get detailed boundary</li>
            <li><code>/api/file/{{filename}}/shots</code> - Get shots in bounds</li>
            <li><code>/api/file/{{filename}}/waveform/{{index}}</code> - Get waveform data</li>
        </ul>
    </body>
    </html>
    '''

@app.route('/api/files')
def list_files():
    """List all available LVIS H5 files."""
    try:
        h5_files = []
        for file_path in glob.glob(os.path.join(DATA_FOLDER, "*.h5")):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            h5_files.append({
                'filename': os.path.basename(file_path),
                'size_mb': round(file_size, 2)
            })
        
        return jsonify({
            'success': True,
            'files': h5_files,
            'count': len(h5_files)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/file/<filename>/info')
def get_file_info(filename: str):
    """Get metadata about a specific file."""
    try:
        # Sanitize filename
        filename = os.path.basename(filename)
        file_path = os.path.join(DATA_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        # Verify file is within DATA_FOLDER
        if not os.path.abspath(file_path).startswith(os.path.abspath(DATA_FOLDER)):
            return jsonify({'success': False, 'error': 'Invalid file path'}), 400
        
        with h5py.File(file_path, 'r') as f:
            shot_count = len(f['SHOTNUMBER'])
            datasets = list(f.keys())

        # Get bounds from polygon file
        base_name = os.path.splitext(filename)[0]
        polygon_path = os.path.join(DATA_FOLDER, f"{base_name}_coverage.geojson")

        with open(polygon_path, 'r') as f:
            polygon_data = json.load(f)
            bounds = calculate_bounds_from_geojson(polygon_data)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'shot_count': shot_count,
            'bounds': bounds,
            'datasets': datasets
        })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/file/<filename>/polygon')
def get_file_polygon(filename: str):
    """Get coverage polygon (convex hull) for a specific file."""
    try:
        filename = os.path.basename(filename)
        base_name = os.path.splitext(filename)[0]
        polygon_path = os.path.join(DATA_FOLDER, f"{base_name}_coverage.geojson")
        
        if not os.path.exists(polygon_path):
            return jsonify({'success': False, 'error': 'Polygon not found'}), 404
        
        with open(polygon_path, 'r') as f:
            geojson = json.load(f)
        
        return jsonify({
            'success': True,
            'polygon': geojson
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/file/<filename>/detailed')
def get_file_detailed_boundary(filename: str):
    """Get detailed boundary polygon for a specific file."""
    try:
        filename = os.path.basename(filename)
        base_name = os.path.splitext(filename)[0]
        detailed_path = os.path.join(DATA_FOLDER, f"{base_name}_detailed.geojson")
        
        if not os.path.exists(detailed_path):
            return jsonify({'success': False, 'error': 'Detailed boundary not found'}), 404
        
        with open(detailed_path, 'r') as f:
            geojson = json.load(f)
        
        return jsonify({
            'success': True,
            'polygon': geojson
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files/polygons')
def get_all_polygons():
    """Get coverage polygons for all files."""
    try:
        polygons = []
        
        for file_path in glob.glob(os.path.join(DATA_FOLDER, "*.h5")):
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            polygon_path = os.path.join(DATA_FOLDER, f"{base_name}_coverage.geojson")
            detailed_path = os.path.join(DATA_FOLDER, f"{base_name}_detailed.geojson")
        
            try:
                with open(polygon_path, 'r') as pf:
                    geojson = json.load(pf)
                
                # Check if detailed boundary exists
                has_detailed = os.path.exists(detailed_path)
                
                # Extract date from filename
                date_match = re.search(r'(\d{4})_?(\d{2})(\d{2})', filename)
                date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else "Unknown"
                
                polygons.append({
                    'filename': filename,
                    'date': date_str,
                    'shot_count': geojson.get('properties', {}).get('shot_count', 0),
                    'polygon': geojson,
                    'has_detailed': has_detailed
                })
            except Exception:
                continue
        
        return jsonify({
            'success': True,
            'polygons': polygons,
            'count': len(polygons)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/file/<filename>/shots')
def get_shots(filename: str):
    """Get shots within specified bounds."""
    try:
        filename = os.path.basename(filename)
        file_path = os.path.join(DATA_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        # Get bounds from query parameters
        try:
            min_lat = float(request.args.get('minLat', -90))
            max_lat = float(request.args.get('maxLat', 90))
            min_lon = float(request.args.get('minLon', -180))
            max_lon = float(request.args.get('maxLon', 180))
            max_shots = int(request.args.get('maxShots', MAX_MARKERS))
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid bounds parameters'}), 400
        
        with h5py.File(file_path, 'r') as h5f:
            # Load data
            shot_numbers = h5f['SHOTNUMBER'][:]
            lat0 = h5f['LAT0'][:]
            lon0 = h5f['LON0'][:]
            lat1215 = h5f['LAT1215'][:]
            lon1215 = h5f['LON1215'][:]
            
            # Calculate center points
            lats = (lat0 + lat1215) / 2
            lons = (lon0 + lon1215) / 2
            
            # Convert longitude if needed
            lons = np.where(lons > 180, lons - 360, lons)
            
            # Filter by bounds
            mask = (
                (lats >= min_lat) & (lats <= max_lat) &
                (lons >= min_lon) & (lons <= max_lon) &
                ~np.isnan(lats) & ~np.isnan(lons)
            )
            
            indices = np.where(mask)[0]
            
            # Limit number of results
            if len(indices) > max_shots:
                # Sample evenly
                step = len(indices) // max_shots
                indices = indices[::step][:max_shots]
            
            # Build response
            shots = []
            for idx in indices:
                shots.append({
                    'index': int(idx),
                    'shot_number': int(shot_numbers[idx]),
                    'lat': float(lats[idx]),
                    'lon': float(lons[idx])
                })
            
            return jsonify({
                'success': True,
                'shot_count': len(shots),
                'total_in_bounds': int(np.sum(mask)),
                'shots': shots
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/file/<filename>/waveform/<int:index>')
def get_waveform(filename: str, index: int):
    """Get waveform data for a specific shot."""
    try:
        filename = os.path.basename(filename)
        file_path = os.path.join(DATA_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        with h5py.File(file_path, 'r') as f:
            # Check if index is valid
            shot_count = len(f['SHOTNUMBER'])
            if index < 0 or index >= shot_count:
                return jsonify({'success': False, 'error': 'Invalid shot index'}), 400
            
            # Check if waveform data exists
            if 'RXWAVE' not in f:
                return jsonify({'success': False, 'error': 'No waveform data in file'}), 404
            
            # Get waveform and elevation data
            waveform = f['RXWAVE'][index].tolist()
            z0 = float(f['Z0'][index])
            z1215 = float(f['Z1215'][index])
            sigmean = float(f['SIGMEAN'][index])
            shot_number = int(f['SHOTNUMBER'][index])
            
            # Create elevation array
            elevations = np.linspace(z0, z1215, 1216).tolist()
            
            response_data = {
                'success': True,
                'shot_number': shot_number,
                'waveform': waveform,
                'elevations': elevations,
                'z_top': z0,
                'z_bottom': z1215,
                'sigmean': sigmean
            }
            
            # Try to get L2 data if configured
            if L2_FOLDER:
                l2_file = find_l2_file(filename)
                if l2_file:
                    l2_data = read_l2_data(l2_file, shot_number)
                    response_data['l2_data'] = l2_data
                else:
                    response_data['l2_data'] = {'found': False, 'reason': 'No matching L2 file'}
            else:
                response_data['l2_data'] = {'found': False, 'reason': 'L2 folder not configured'}
            
            return jsonify(response_data)
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== Main Entry Point =====

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='LVIS HDF5 Data Server with polygon generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--skip-polygon-gen', action='store_true', 
                       help='Skip polygon generation check at startup')
    parser.add_argument('--force-polygon-gen', action='store_true',
                       help='Force regenerate all polygon files')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                       help=f'Port to run server on (default: {DEFAULT_PORT})')
    parser.add_argument('--buffer-size', type=int, default=DEFAULT_BUFFER_METERS,
                       help=f'Buffer size for detailed polygons in meters (default: {DEFAULT_BUFFER_METERS})')
    parser.add_argument('--max-markers', type=int, default=MAX_MARKERS_PER_VIEW,
                       help=f'Maximum markers per view (default: {MAX_MARKERS_PER_VIEW})')
    args = parser.parse_args()
    
    # Set configuration from arguments
    PORT = args.port
    BUFFER_METERS = args.buffer_size
    MAX_MARKERS = args.max_markers
    REGENERATE_POLYGONS = args.force_polygon_gen
    
    # Get the data folders from user
    print("\n" + "="*50)
    print("LVIS HDF5 Data Server Setup")
    print("="*50)
    
    DATA_FOLDER = input("\nEnter the path to your LVIS L1B .h5 files folder: ").strip()
    
    # Remove quotes if user included them
    if DATA_FOLDER.startswith('"') and DATA_FOLDER.endswith('"'):
        DATA_FOLDER = DATA_FOLDER[1:-1]
    elif DATA_FOLDER.startswith("'") and DATA_FOLDER.endswith("'"):
        DATA_FOLDER = DATA_FOLDER[1:-1]
    
    # Verify the folder exists
    if not os.path.exists(DATA_FOLDER):
        print(f"\n❌ Error: Folder not found: {DATA_FOLDER}")
        print("Please check the path and try again.")
        exit(1)
    
    if not os.path.isdir(DATA_FOLDER):
        print(f"\n❌ Error: Path is not a directory: {DATA_FOLDER}")
        exit(1)
    
    # Check for .h5 files
    h5_files = glob.glob(os.path.join(DATA_FOLDER, "*.h5"))
    if len(h5_files) == 0:
        print(f"\n⚠️  Warning: No .h5 files found in {DATA_FOLDER}")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            exit(0)
    else:
        print(f"\n✓ Found {len(h5_files)} .h5 file(s) in the folder")
    
    # Ask for L2 folder (optional)
    print("\n" + "-"*50)
    L2_FOLDER = input("\nEnter the path to your LVIS L2 files folder (or press Enter to skip): ").strip()
    
    if L2_FOLDER:
        # Remove quotes if user included them
        if L2_FOLDER.startswith('"') and L2_FOLDER.endswith('"'):
            L2_FOLDER = L2_FOLDER[1:-1]
        elif L2_FOLDER.startswith("'") and L2_FOLDER.endswith("'"):
            L2_FOLDER = L2_FOLDER[1:-1]
        
        # Verify L2 folder exists
        if not os.path.exists(L2_FOLDER):
            print(f"\n⚠️  Warning: L2 folder not found: {L2_FOLDER}")
            print("L2 data will not be available.")
            L2_FOLDER = None
        elif not os.path.isdir(L2_FOLDER):
            print(f"\n⚠️  Warning: L2 path is not a directory: {L2_FOLDER}")
            print("L2 data will not be available.")
            L2_FOLDER = None
        else:
            l2_files = glob.glob(os.path.join(L2_FOLDER, "*.txt"))
            if len(l2_files) == 0:
                print(f"\n⚠️  Warning: No .txt files found in L2 folder")
                print("L2 data may not be available.")
            else:
                print(f"\n✓ Found {len(l2_files)} L2 file(s)")
    else:
        print("\n✓ L2 folder skipped - L2 metrics will not be available")
    
    # Show the server configuration
    print(f"""
    ╔════════════════════════════════════════════╗
    ║            LVIS HDF5 Data Server           ║
    ║    With Multi-Level Coverage Polygons      ║
    ╚════════════════════════════════════════════╝
    
    Configuration:
    - L1B Data folder: {DATA_FOLDER}
    - L2 Data folder: {L2_FOLDER if L2_FOLDER else 'Not configured'}
    - Server port: {PORT}
    - Buffer size: {BUFFER_METERS}m
    - Max markers: {MAX_MARKERS}
    
    Polygon types:
    - Convex hull: Full coverage area
    - Detailed boundary: {BUFFER_METERS}m buffered points
    """)
    
    # Generate polygons if needed
    if not args.skip_polygon_gen:
        if REGENERATE_POLYGONS:
            print("\nForce regenerating all polygons (--force-polygon-gen)...")
        else:
            print("\nChecking coverage polygons...")
            
        try:
            generate_coverage_polygons()
        except Exception as e:
            print(f"Error during polygon generation: {e}")
            print("Continuing with server startup...")
    
    print("\nStarting web server...")
    print(f"Server accessible at: http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server\n")
    
    # Run server
    app.run(host='0.0.0.0', port=PORT, debug=False)
