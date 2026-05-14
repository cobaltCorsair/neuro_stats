# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['work_with_prepared_data/radiobioligy_project/gui/main_window.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        # PyQt6
        'PyQt6.sip',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.QtPrintSupport',
        # matplotlib Qt backend
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_svg',
        'matplotlib.backends.backend_agg',
        # scipy hidden imports
        'scipy._lib.messagestream',
        'scipy.special._ufuncs_cxx',
        'scipy.special._cython_special',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'scipy.linalg._fblas',
        'scipy.linalg._flapack',
        'scipy.spatial.transform._rotation_groups',
        'scipy.io.matlab._streams',
        'scipy.stats._distn_infrastructure',
        'scipy.stats.distributions',
        # sklearn
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
        # pandas / openpyxl
        'openpyxl',
        'openpyxl.styles',
        'openpyxl.utils',
        'pandas._libs.tslibs.np_datetime',
        'pandas._libs.tslibs.nattype',
        'pandas._libs.tslibs.timedeltas',
        # other
        'fastdtw',
        'seaborn',
        'nibabel',
        'pydicom',
    ],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RadiobiologyAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    runtime_hooks=['pyi_rth_scipy.py'],
)
