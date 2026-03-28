# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SurfaceScope — RGB-D Surface Profiler & Analyzer.

Build command:
    pyinstaller build.spec

Output:
    dist/FASL_3D_Distance_Profiler/  (one-dir mode)
"""

import os

block_cipher = None

project_root = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(project_root, 'run_app.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        (os.path.join(project_root, 'app', 'static'), os.path.join('app', 'static')),
    ],
    hiddenimports=[
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'starlette',
        'pydantic',
        'numpy',
        'scipy',
        'scipy.ndimage',
        'PIL',
        'app',
        'app.main',
        'app.api',
        'app.api.routes',
        'app.simulation',
        'app.simulation.depth_generator',
        'app.simulation.depth_processing',
        'app.simulation.surface_reconstruction',
        'app.simulation.profile_analysis',
        'app.simulation.colormap',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'pandas',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SurfaceScope',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SurfaceScope',
)
