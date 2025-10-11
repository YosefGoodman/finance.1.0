#!/usr/bin/env python3
#add to requirments to REQUIRED_PACKAGES if you wan to add more
"""
Smart dependency installer - checks versions and only updates what's needed
Usage: python setup_dependencies.py
"""

import subprocess
import sys
from packaging import version

# Define required packages with minimum versions

REQUIRED_PACKAGES = {
    # Core
    'numpy': '1.26',
    'pandas': '2.1',
    # Data
    'yfinance': '0.2.29',
    # Time Series
    'pykalman': '0.9.5',
    'arch': '6.8',
    # ML/DL
    'torch': '2.2',
    'einops': '0.6',
    'scikit-learn': '1.3',
    # Utils
    'joblib': '1.3',
    'matplotlib': '3.8',
    # Optional
    'vaderSentiment': '3.3.2'
}

def get_installed_version(package_name):
    """Get currently installed version of a package"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':')[1].strip()
        return None
    except Exception as e:
        print(f"Error checking {package_name}: {e}")
        return None

def compare_versions(installed, required):
    """Compare version strings"""
    try:
        return version.parse(installed) >= version.parse(required)
    except Exception:
        return False

def install_package(package_name, min_version):
    """Install or upgrade a package"""
    print(f"Installing {package_name}>={min_version}...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', f'{package_name}>={min_version}'],
            check=True
        )
        print(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package_name}")
        return False

def uninstall_package(package_name):
    """Uninstall outdated package"""
    print(f"Uninstalling outdated {package_name}...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'uninstall', '-y', package_name],
            check=True
        )
        print(f"✓ {package_name} uninstalled")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to uninstall {package_name}")
        return False

def main():
    print("=" * 60)
    print("SMART DEPENDENCY CHECKER")
    print("=" * 60)
    
    # First, ensure pip and packaging are up to date
    print("\nUpdating pip and packaging...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'packaging'], 
                   capture_output=True)
    
    to_install = []
    to_uninstall = []
    up_to_date = []
    
    print("\nChecking installed packages...\n")
    
    for package, required_ver in REQUIRED_PACKAGES.items():
        installed_ver = get_installed_version(package)
        
        if installed_ver is None:
            print(f"✗ {package}: NOT INSTALLED")
            to_install.append((package, required_ver))
        elif compare_versions(installed_ver, required_ver):
            print(f"✓ {package}: {installed_ver} (up to date)")
            up_to_date.append(package)
        else:
            print(f"⚠ {package}: {installed_ver} (outdated, need >={required_ver})")
            to_uninstall.append(package)
            to_install.append((package, required_ver))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Up to date: {len(up_to_date)}")
    print(f"Need installation: {len([p for p, _ in to_install if p not in to_uninstall])}")
    print(f"Need upgrade: {len(to_uninstall)}")
    
    # Ask for confirmation if changes needed
    if to_install or to_uninstall:
        print("\n" + "=" * 60)
        response = input("Proceed with installation/upgrades? (y/n): ").lower().strip()
        if response != 'y':
            print("Installation cancelled.")
            return
        
        # Uninstall outdated packages first
        if to_uninstall:
            print("\n" + "=" * 60)
            print("UNINSTALLING OUTDATED PACKAGES")
            print("=" * 60)
            for package in to_uninstall:
                uninstall_package(package)
        
        # Install/upgrade packages
        if to_install:
            print("\n" + "=" * 60)
            print("INSTALLING PACKAGES")
            print("=" * 60)
            failed = []
            for package, min_ver in to_install:
                if not install_package(package, min_ver):
                    failed.append(package)
            
            if failed:
                print(f"\n⚠ WARNING: Failed to install: {', '.join(failed)}")
            else:
                print("\n✓ All packages installed successfully!")
    else:
        print("\n✓ All packages are up to date! No action needed.")
    
    # Final verification
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)
    all_good = True
    for package, required_ver in REQUIRED_PACKAGES.items():
        installed_ver = get_installed_version(package)
        if installed_ver and compare_versions(installed_ver, required_ver):
            print(f"✓ {package}: {installed_ver}")
        else:
            print(f"✗ {package}: STILL MISSING OR OUTDATED")
            all_good = False
    
    if all_good:
        print("\n" + "=" * 60)
        print("✓ SETUP COMPLETE! Ready to start the pipeline.")
        print("=" * 60)
    else:
        print("\n⚠ Some packages are still not properly installed.")
        print("Try running: pip install -r requirements.txt")

if __name__ == '__main__':
    main()
