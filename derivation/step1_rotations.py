"""Step 1: rotation matrices (paper Eq. 8).

R = Rz(psi) * Ry(theta) * Rx(phi) rotates vectors from the body frame to the world frame.
"""
import sympy as sp

phi, theta, psi = sp.symbols('phi theta psi', real=True)  # roll, pitch, yaw


def Rx(a):
    """Rotation by angle a about the x axis."""
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])


def Ry(a):
    """Rotation by angle a about the y axis."""
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])


def Rz(a):
    """Rotation by angle a about the z axis."""
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])


R = Rz(psi) * Ry(theta) * Rx(phi)

if __name__ == '__main__':
    print("Rz(psi) =")
    sp.pprint(Rz(psi))
    print("\nR = Rz(psi) * Ry(theta) * Rx(phi) =")
    sp.pprint(R)
    # A rotation matrix must satisfy R * R^T = I
    print("\nR * R^T simplifies to identity:", sp.simplify(R * R.T) == sp.eye(3))
