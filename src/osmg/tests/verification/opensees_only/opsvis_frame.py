"""
Portal frame analysis benchmark file.

Added: Mon Nov 25 04:41:46 AM PST 2024

Comes from the 2D portal frame example in `opsvis`:
https://opsvis.readthedocs.io/en/latest/ex_2d_portal_frame.html
"""

import openseespy.opensees as ops

ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)

column_length, girder_length = 4.0, 6.0

Acol, Agir = 2.0e-3, 6.0e-3
IzCol, IzGir = 1.6e-5, 5.4e-5

E = 200.0e9

Ep = {1: [E, Acol, IzCol], 2: [E, Acol, IzCol], 3: [E, Agir, IzGir]}

ops.node(1, 0.0, 0.0)
ops.node(2, 0.0, column_length)
ops.node(3, girder_length, 0.0)
ops.node(4, girder_length, column_length)

ops.fix(1, 1, 1, 1)
ops.fix(3, 1, 1, 0)

ops.geomTransf('Linear', 1)

# columns
ops.element('elasticBeamColumn', 1, 1, 2, Acol, E, IzCol, 1)
ops.element('elasticBeamColumn', 2, 3, 4, Acol, E, IzCol, 1)
# girder
ops.element('elasticBeamColumn', 3, 2, 4, Agir, E, IzGir, 1)

Px = 2.0e3
Wy = -10.0e3
Wx = 0.0

Ew = {3: ['-beamUniform', Wy, Wx]}

ops.timeSeries('Constant', 1)
ops.pattern('Plain', 1, 1)
ops.load(2, Px, 0.0, 0.0)

for etag in Ew:
    ops.eleLoad('-ele', etag, '-type', Ew[etag][0], Ew[etag][1], Ew[etag][2])

# recorder
ops.recorder(
    'Node', '-file', './disp.txt', '-time', '-node', 4, '-dof', 1, 2, 3, 'disp'
)

ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr', 1.0e-6, 6, 2)
ops.algorithm('Linear')
ops.integrator('LoadControl', 1)
ops.analysis('Static')
ops.analyze(1)
