#!/usr/bin/env python3
"""
Quick launcher for MEHSANA ROI Configurator and Detection Script
Allows easy switching between LINE and RECTANGLE modes
"""
import sys
import subprocess
import json
import os

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_menu():
    print("\nMEHSANA TRAFFIC DETECTION SYSTEM - Main Menu")
    print("-" * 70)
    print("1. Configure ROI (LINE mode) - L1, L2, Detection lines")
    print("2. Configure ROI (RECTANGLE mode) - 2-point rectangle")
    print("3. Run Detection with current config")
    print("4. View current configuration")
    print("5. Clear existing configuration")
    print("6. Exit")
    print("-" * 70)

def show_config():
    """Display current configuration"""
    config_file = "roi_config.json"
    if not os.path.exists(config_file):
        print("\nERROR: No configuration file found!")
        return
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print_header("Current Configuration")
        mode = config.get('mode', 'UNKNOWN')
        print(f"Mode: {mode}")
        
        if mode == 'LINE':
            print("\nL1 Line:")
            print(f"  Start: {config.get('l1_line', {}).get('start', 'N/A')}")
            print(f"  End:   {config.get('l1_line', {}).get('end', 'N/A')}")
            print("\nL2 Line:")
            print(f"  Start: {config.get('l2_line', {}).get('start', 'N/A')}")
            print(f"  End:   {config.get('l2_line', {}).get('end', 'N/A')}")
            print("\nDetection Line:")
            print(f"  Start: {config.get('detection_line', {}).get('start', 'N/A')}")
            print(f"  End:   {config.get('detection_line', {}).get('end', 'N/A')}")
            print("\nAOI Polygon: {} points".format(len(config.get('aoi_polygon', []))))
        
        elif mode == 'RECTANGLE':
            rect = config.get('rectangle_roi', [])
            print(f"\nRectangle ROI: {len(rect)} corners")
            if rect:
                print(f"  Top-Left:     {rect[0]}")
                print(f"  Top-Right:    {rect[1]}")
                print(f"  Bottom-Right: {rect[2]}")
                print(f"  Bottom-Left:  {rect[3]}")

        print("\n" + "="*70)
    
    except Exception as e:
        print(f"ERROR reading config: {e}")

def clear_config():
    """Clear existing configuration"""
    config_file = "roi_config.json"
    if os.path.exists(config_file):
        confirm = input(f"\nDelete {config_file}? (y/n): ").strip().lower()
        if confirm == 'y':
            os.remove(config_file)
            print(f"OK {config_file} deleted")
        else:
            print("Cancelled")

def run_configurator(mode):
    """Run ROI configurator in specified mode"""
    print_header(f"Starting ROI Configurator ({mode} mode)")
    try:
        subprocess.run([sys.executable, "mehsana_roi_configurator.py", mode])
    except Exception as e:
        print(f"ERROR: {e}")

def run_detection():
    """Run vehicle detection script"""
    print_header("Starting Vehicle Detection")
    try:
        subprocess.run([sys.executable, "mehsana_vehicle_detection.py"])
    except Exception as e:
        print(f"ERROR: {e}")

def main():
    print_header("MEHSANA TRAFFIC DETECTION SYSTEM")
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            run_configurator("LINE")
        elif choice == '2':
            run_configurator("RECTANGLE")
        elif choice == '3':
            run_detection()
        elif choice == '4':
            show_config()
        elif choice == '5':
            clear_config()
        elif choice == '6':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
