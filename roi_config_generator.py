#!/usr/bin/env python3
"""
Helper script to generate initial roi_config.json from current settings
or modify existing configuration programmatically
"""

import json
import numpy as np

def generate_default_config():
    """Generate default ROI config from hardcoded values"""
    
    # Default ROI coordinates
    l1_start = [4, 1019]
    l1_end = [1188, 219]
    l2_start = [1218, 240]
    l2_end = [1872, 1099]
    
    # Detection line
    det_start = [4, 1019]
    det_end = [1218, 240]
    
    # AOI polygon (between L1 and L2)
    aoi_polygon = [
        l1_start,
        l1_end,
        l2_end,
        l2_start
    ]
    
    config = {
        "l1_line": {
            "start": l1_start,
            "end": l1_end
        },
        "l2_line": {
            "start": l2_start,
            "end": l2_end
        },
        "detection_line": {
            "start": det_start,
            "end": det_end
        },
        "aoi_polygon": aoi_polygon,
        "metadata": {
            "description": "ROI configuration for Mehsana traffic detection",
            "created_by": "roi_config_generator.py",
            "video_resolution": "3840x2160"
        }
    }
    
    return config

def save_config(config, filename="roi_config.json"):
    """Save configuration to JSON file"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Configuration saved to {filename}")
    print_config(config)

def load_config(filename="roi_config.json"):
    """Load configuration from JSON file"""
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded configuration from {filename}")
        return config
    except FileNotFoundError:
        print(f"✗ {filename} not found")
        return None

def print_config(config):
    """Pretty print configuration"""
    print("\n" + "="*70)
    print("  ROI CONFIGURATION")
    print("="*70)
    print(f"\nL1 Line:")
    print(f"  Start: {config['l1_line']['start']}")
    print(f"  End:   {config['l1_line']['end']}")
    
    print(f"\nL2 Line:")
    print(f"  Start: {config['l2_line']['start']}")
    print(f"  End:   {config['l2_line']['end']}")
    
    print(f"\nDetection Line:")
    print(f"  Start: {config['detection_line']['start']}")
    print(f"  End:   {config['detection_line']['end']}")
    
    print(f"\nAOI Polygon ({len(config['aoi_polygon'])} points):")
    for i, point in enumerate(config['aoi_polygon']):
        print(f"  Point {i}: {point}")
    
    if 'metadata' in config:
        print(f"\nMetadata:")
        for key, value in config['metadata'].items():
            print(f"  {key}: {value}")
    print("="*70 + "\n")

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="ROI Configuration Manager")
    parser.add_argument('--generate', action='store_true', help='Generate default config')
    parser.add_argument('--load', action='store_true', help='Load existing config')
    parser.add_argument('--file', default='roi_config.json', help='Config file path')
    parser.add_argument('--l1-start', type=int, nargs=2, help='L1 start point (x y)')
    parser.add_argument('--l1-end', type=int, nargs=2, help='L1 end point (x y)')
    parser.add_argument('--l2-start', type=int, nargs=2, help='L2 start point (x y)')
    parser.add_argument('--l2-end', type=int, nargs=2, help='L2 end point (x y)')
    parser.add_argument('--det-start', type=int, nargs=2, help='Detection line start (x y)')
    parser.add_argument('--det-end', type=int, nargs=2, help='Detection line end (x y)')
    
    args = parser.parse_args()
    
    if args.generate:
        config = generate_default_config()
        save_config(config, args.file)
    elif args.load:
        config = load_config(args.file)
        if config:
            print_config(config)
    elif any([args.l1_start, args.l1_end, args.l2_start, args.l2_end, args.det_start, args.det_end]):
        config = load_config(args.file) or generate_default_config()
        
        if args.l1_start:
            config['l1_line']['start'] = list(args.l1_start)
        if args.l1_end:
            config['l1_line']['end'] = list(args.l1_end)
        if args.l2_start:
            config['l2_line']['start'] = list(args.l2_start)
        if args.l2_end:
            config['l2_line']['end'] = list(args.l2_end)
        if args.det_start:
            config['detection_line']['start'] = list(args.det_start)
        if args.det_end:
            config['detection_line']['end'] = list(args.det_end)
        
        # Recalculate AOI polygon
        config['aoi_polygon'] = [
            config['l1_line']['start'],
            config['l1_line']['end'],
            config['l2_line']['end'],
            config['l2_line']['start']
        ]
        
        save_config(config, args.file)
        print("\n✓ Configuration updated!")
    else:
        print("\nROI Configuration Manager")
        print("\nUsage:")
        print("  Generate default:  python roi_config_generator.py --generate")
        print("  Load config:        python roi_config_generator.py --load")
        print("  Update L1 start:    python roi_config_generator.py --l1-start 10 100")
        print("  Update detection:   python roi_config_generator.py --det-start 100 200 --det-end 500 600")
        print("\nInteractive mode:")
        print("  python mehsana_roi_configurator.py")
        print("  (Draw lines by clicking on the video frame)")
