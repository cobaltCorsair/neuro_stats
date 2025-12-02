# PyInstaller runtime hook for scipy
# Fixes NameError: name 'obj' is not defined in scipy.stats._distn_infrastructure

import sys
import os

# Force scipy to load properly before anything else
def _patch_scipy():
    try:
        # Force import of scipy in correct order
        import scipy
        import scipy._lib
        import scipy.special
        
        # Import stats infrastructure in the right order
        import scipy.stats
        
        # Force load of distributions to ensure 'obj' is defined
        from scipy.stats import distributions
        
    except Exception as e:
        # If this fails, log it but don't crash
        print(f"Warning: scipy runtime hook failed: {e}", file=sys.stderr)

# Run the patch before anything else
_patch_scipy()
