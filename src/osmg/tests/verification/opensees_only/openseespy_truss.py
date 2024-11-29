"""
Simple OpenSees example of a plane truss.

Adapted from the OpenSeesPy documentation (examples).

Thu Nov 28 10:19:49 AM PST 2024

"""

import openseespy.opensees as ops

ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 2)

ops.node(1, 0.0, 0.0)
ops.node(2, 144.0, 0.0)
ops.node(3, 168.0, 0.0)
ops.node(4, 72.0, 96.0)

ops.fix(1, 1, 1)
ops.fix(2, 1, 1)
ops.fix(3, 1, 1)

ops.uniaxialMaterial('Elastic', 1, 3000.0)

ops.element('Truss', 1, 1, 4, 10.0, 1)
ops.element('Truss', 2, 2, 4, 5.0, 1)
ops.element('Truss', 3, 3, 4, 5.0, 1)

ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
ops.load(4, 100.0, -50.0)

# Adding a UDL on a truss member is not supported:
# ops.eleLoad('-ele', 1, '-type', '-beamUniform', 0.00, 0.00, 1.00)

# Add a recorder for the basic forces.
ops.recorder(
    'Element',
    '-file',
    '/tmp/truss_recorder.txt',  # noqa: S108
    '-time',
    '-ele',
    *(1, 2, 3),
    'localForce',
)

ops.system('BandSPD')
ops.numberer('RCM')
ops.constraints('Plain')
ops.integrator('LoadControl', 1.0)
ops.algorithm('Linear')
ops.analysis('Static')

ops.analyze(1)

ux = ops.nodeDisp(4, 1)
uy = ops.nodeDisp(4, 2)

assert abs(ux - 0.53009277713228375450) < 1e-12
assert abs(uy + 0.17789363846931768864) < 1e-12
