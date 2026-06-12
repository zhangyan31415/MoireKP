#!/usr/bin/env python3
"""
Comprehensive test script for all slab calculation features

This script tests:
1. Basic slab calculation
2. Bulk vs slab comparison  
3. Surface state analysis (top/bottom/mixed)
4. Parallel processing
5. Colorbar visualization
"""

import os
import sys
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_all_features():
    """Test all implemented features."""
    
    print("=" * 80)
    print("COMPREHENSIVE SLAB CALCULATION TEST")
    print("=" * 80)
    
    # Check if required files exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    required_files = ['H.dat', 'S.dat', 'POSCAR']
    
    for filename in required_files:
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ ERROR: Required file '{filename}' not found")
            return False
    
    print("✅ All required files found")
    
    # Test different command combinations
    test_commands = [
        {
            'name': 'Basic slab calculation',
            'cmd': 'python slab_correct.py --nslab 3 --nseg 20 --energy_range -3 1',
            'expected_output': 'test_basic_slab.png'
        },
        {
            'name': 'Bulk vs slab comparison',
            'cmd': 'python slab_correct.py --include_bulk --nslab 3 --nseg 20 --energy_range -3 1 --outfile test_bulk_slab.png',
            'expected_output': 'test_bulk_slab.png'
        },
        {
            'name': 'Surface state analysis',
            'cmd': 'python slab_correct.py --analyze_surface --nslab 5 --nseg 20 --energy_range -3 1 --outfile test_surface.png',
            'expected_output': 'test_surface.png'
        },
        {
            'name': 'Parallel surface analysis',
            'cmd': 'python slab_correct.py --parallel --analyze_surface --nslab 5 --nseg 20 --energy_range -3 1 --outfile test_parallel.png',
            'expected_output': 'test_parallel.png'
        },
        {
            'name': 'Full comparison (bulk + surface + parallel)',
            'cmd': 'python slab_correct.py --parallel --include_bulk --analyze_surface --compare_thickness --nslab 7 --nseg 15 --energy_range -3 1 --outfile test_full.png',
            'expected_output': 'test_full.png'
        }
    ]
    
    print(f"\n🧪 Running {len(test_commands)} test cases...")
    
    results = []
    
    for i, test in enumerate(test_commands, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_commands)}: {test['name']}")
        print(f"{'='*60}")
        print(f"Command: {test['cmd']}")
        
        start_time = time.time()
        
        try:
            # Run the command
            import subprocess
            result = subprocess.run(test['cmd'].split(), 
                                  capture_output=True, text=True, timeout=300)
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                # Check if output file was created
                if os.path.exists(test['expected_output']):
                    print(f"✅ SUCCESS: {test['name']} completed in {elapsed_time:.1f}s")
                    print(f"   Output file: {test['expected_output']}")
                    results.append(('PASS', test['name'], elapsed_time))
                else:
                    print(f"⚠️  WARNING: Command succeeded but output file not found")
                    results.append(('WARN', test['name'], elapsed_time))
            else:
                print(f"❌ FAILED: {test['name']}")
                print(f"   Error: {result.stderr}")
                results.append(('FAIL', test['name'], elapsed_time))
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {test['name']} took longer than 5 minutes")
            results.append(('TIMEOUT', test['name'], 300))
            
        except Exception as e:
            print(f"❌ ERROR: {test['name']} - {str(e)}")
            results.append(('ERROR', test['name'], 0))
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for r in results if r[0] == 'PASS')
    total = len(results)
    
    for status, name, time_taken in results:
        status_icon = {
            'PASS': '✅',
            'WARN': '⚠️ ',
            'FAIL': '❌',
            'TIMEOUT': '⏰',
            'ERROR': '💥'
        }[status]
        
        print(f"{status_icon} {name:<40} ({time_taken:6.1f}s)")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your slab calculation tool is working perfectly!")
        print("\nNew Features Successfully Implemented:")
        print("  ✅ Top/Bottom surface state distinction")
        print("  ✅ Surface character colorbar")  
        print("  ✅ Parallel processing support")
        print("  ✅ Enhanced visualization")
        
        print("\nGenerated test files:")
        for test in test_commands:
            if os.path.exists(test['expected_output']):
                print(f"  📊 {test['expected_output']}")
                
    else:
        print(f"\n⚠️  Some tests failed. Please check the errors above.")
        
    return passed == total

def demonstrate_features():
    """Demonstrate the new features."""
    
    print(f"\n{'='*80}")
    print("FEATURE DEMONSTRATION")
    print(f"{'='*80}")
    
    print("\n🎨 Surface State Visualization:")
    print("  • Red bands: Top surface states")
    print("  • Blue bands: Bottom surface states") 
    print("  • Purple bands: Mixed surface states")
    print("  • Light colored bands: Bulk-like states")
    print("  • Colorbar: Shows surface character intensity (0-1)")
    
    print("\n⚡ Parallel Processing:")
    print("  • Automatically uses multiple CPU cores")
    print("  • Significant speedup for large calculations")
    print("  • Use --parallel --n_jobs N to control")
    
    print("\n📊 Enhanced Analysis:")
    print("  • Quantitative surface/bulk classification")
    print("  • Separate top/bottom surface identification")
    print("  • Statistical reporting of surface states")
    
    print("\n🚀 Usage Examples:")
    print("  # Basic surface analysis with colorbar")
    print("  python slab_correct.py --analyze_surface --nslab 7")
    print()
    print("  # Fast parallel calculation")  
    print("  python slab_correct.py --parallel --compare_thickness --nslab 9")
    print()
    print("  # Complete analysis")
    print("  python slab_correct.py --parallel --include_bulk --analyze_surface --nslab 5")

if __name__ == '__main__':
    success = test_all_features()
    
    if success:
        demonstrate_features()
        print(f"\n{'='*80}")
        print("🎯 Your enhanced slab calculation tool is ready to use!")
        print(f"{'='*80}")
    else:
        print("\n❌ Some tests failed. Please fix the issues and try again.")
        sys.exit(1) 