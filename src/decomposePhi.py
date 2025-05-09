import numpy

# Value at which to consider no rotation to avoid numerical issues
rotationAngleDegThreshold = 0.0001

# Value at which to consider no stretch to avoid numerical issues
distanceFromIdentity = 0.00001


def decomposeF(F, twoD=False, verbose=False):
    """
    Get components out of a transformation gradient tensor F

    Parameters
    ----------
        F : 3x3 array
            The transformation gradient tensor F (I + du/dx)

        twoD : bool, optional
            is the F defined in 2D? This applies corrections to the strain outputs
            Default = False

        verbose : boolean, optional
            Print errors?
            Default = True

    Returns
    -------
        transformation : dictionary of arrays

                - r = 3x1 numpy array: Rotation in "rotation vector" format
                - z = 3x1 numpy array: Zoom in "zoom vector" format (z, y, x)
                - U = 3x3 numpy array: Right stretch tensor, U - numpy.eye(3) is the strain tensor in large strains
                - e = 3x3 numpy array: Strain tensor in small strains (symmetric)
                - vol   = float: First  invariant of the strain tensor ("Volumetric Strain"), det(F)-1
                - dev   = float: Second invariant of the strain tensor ("Deviatoric Strain")
                - volss = float: First  invariant of the strain tensor ("Volumetric Strain") in small strains, trace(e)/3
                - devss = float: Second invariant of the strain tensor ("Deviatoric Strain") in small strains

    """
    # - G = 3x3 numpy array: Eigen vectors * eigenvalues of strains, from which principal directions of strain can be obtained
    # Default non-success values to be over written if all goes well
    transformation = {
        "r": numpy.array([numpy.nan] * 3),
        "z": numpy.array([numpy.nan] * 3),
        "U": numpy.eye(3) * numpy.nan,
        "e": numpy.eye(3) * numpy.nan,
        "vol": numpy.nan,
        "dev": numpy.nan,
        "volss": numpy.nan,
        "devss": numpy.nan,
        # 'G': 3x3
    }

    # Check for NaNs if any quit
    if numpy.isnan(F).sum() > 0:
        if verbose:
            print("deformationFunction.decomposeF(): Nan value in F. Exiting")
        return transformation

    # Check for inf if any quit
    if numpy.isinf(F).sum() > 0:
        if verbose:
            print("deformationFunction.decomposeF(): Inf value in F. Exiting")
        return transformation

    # Check for singular F if yes quit
    try:
        numpy.linalg.inv(F)
    except numpy.linalg.linalg.LinAlgError:
        if verbose:
            print("deformationFunction.decomposeF(): F is singular. Exiting")
        return transformation

    ###########################################################
    # Polar decomposition of F = RU
    # U is the right stretch tensor
    # R is the rotation tensor
    ###########################################################

    # Compute the Right Cauchy tensor
    C = numpy.dot(F.T, F)

    # 2020-02-24 EA and OS (day of the fire in 3SR): catch the case when C is practically the identity matrix (i.e., a rigid body motion)
    # TODO: At least also catch the case when two eigenvales are very small
    if numpy.abs(numpy.subtract(C, numpy.eye(3))).sum() < distanceFromIdentity:
        # This forces the rest of the function to give trivial results
        C = numpy.eye(3)

    # Solve eigen problem
    CeigVal, CeigVec = numpy.linalg.eig(C)

    # 2018-06-29 OS & ER check for negative eigenvalues
    # test "really" negative eigenvalues
    if CeigVal.any() / CeigVal.mean() < -1:
        print(
            "deformationFunction.decomposeF(): negative eigenvalue in transpose(F). Exiting"
        )
        print("Eigenvalues are: {}".format(CeigVal))
        exit()
    # for negative eigen values but close to 0 we set it to 0
    CeigVal[CeigVal < 0] = 0

    # Diagonalise C --> which is U**2
    diagUsqr = numpy.array([[CeigVal[0], 0, 0], [0, CeigVal[1], 0], [0, 0, CeigVal[2]]])

    diagU = numpy.sqrt(diagUsqr)

    # 2018-02-16 check for both issues with negative (Udiag)**2 values and inverse errors
    try:
        U = numpy.dot(numpy.dot(CeigVec, diagU), CeigVec.T)
        R = numpy.dot(F, numpy.linalg.inv(U))
    except numpy.linalg.LinAlgError:
        return transformation

    # normalisation of rotation matrix in order to respect basic properties
    # otherwise it gives errors like trace(R) > 3
    # this issue might come from numerical noise.
    # ReigVal, ReigVec = numpy.linalg.eig(R)
    for i in range(3):
        R[i, :] /= numpy.linalg.norm(R[i, :])
    # print("traceR - sumEig = {}".format(R.trace() - ReigVal.sum()))
    # print("u.v = {}".format(numpy.dot(R[:, 0], R[:, 1])))
    # print("detR = {}".format(numpy.linalg.det(R)))

    # Calculate rotation angle
    # Detect an identity -- zero rotation
    # if numpy.allclose(R, numpy.eye(3),  atol=1e-03):
    #     rotationAngleRad = 0.0
    #     rotationAngleDeg = 0.0

    # Detect trace(R) > 3 (should not happen but happens)
    arccosArg = 0.5 * (R.trace() - 1.0)
    if arccosArg > 1.0:
        rotationAngleRad = 0.0
    else:
        # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_.E2.86.94_Euler_axis.2Fangle
        rotationAngleRad = numpy.arccos(arccosArg)
    rotationAngleDeg = numpy.rad2deg(float(rotationAngleRad))

    if rotationAngleDeg > rotationAngleDegThreshold:
        rotationAxis = numpy.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
        rotationAxis /= 2.0 * numpy.sin(rotationAngleRad)
        rot = rotationAngleDeg * rotationAxis
    else:
        rot = [0.0, 0.0, 0.0]
    ###########################################################

    # print "R is \n", R, "\n"
    # print "|R| is ", numpy.linalg.norm(R), "\n"
    # print "det(R) is ", numpy.linalg.det(R), "\n"
    # print "R.T - R-1 is \n", R.T - numpy.linalg.inv( R ), "\n\n"

    # print "U is \n", U, "\n"
    # print "U-1 is \n", numpy.linalg.inv( U ), "\n\n"

    # Also output eigenvectors * their eigenvalues as output:
    # G = []
    # for eigenvalue, eigenvector in zip(CeigVal, CeigVec):
    # G.append(numpy.multiply(eigenvalue, eigenvector))

    # Compute the volumetric strain from the determinant of F
    vol = numpy.linalg.det(F) - 1

    # Decompose U into an isotropic and deviatoric part
    # and compute the deviatoric strain as the norm of the deviatoric part
    if twoD:
        Udev = U[1:, 1:] * (numpy.linalg.det(F[1:, 1:]) ** (-1 / 2.0))
        dev = numpy.linalg.norm(Udev - numpy.eye(2))
    else:
        Udev = U * (numpy.linalg.det(F) ** (-1 / 3.0))
        dev = numpy.linalg.norm(Udev - numpy.eye(3))

    ###########################################################
    # Small strains bit
    ###########################################################
    # to get rid of numerical noise in 2D
    if twoD:
        F[0, :] = [1.0, 0.0, 0.0]
        F[:, 0] = [1.0, 0.0, 0.0]

    # In small strains: 0.5(F+F.T)
    e = 0.5 * (F + F.T) - numpy.eye(3)

    # The volumetric strain is the trace of the strain matrix
    volss = numpy.trace(e)

    # The deviatoric in the norm of the matrix
    if twoD:
        devss = numpy.linalg.norm(e[1:, 1:] - numpy.eye(2) * volss / 2.0)
    else:
        devss = numpy.linalg.norm(e - numpy.eye(3) * volss / 3.0)

    transformation = {
        "r": rot,
        "z": [U[i, i] for i in range(3)],
        "U": U,
        # 'G': G,
        "e": e,
        "vol": vol,
        "dev": dev,
        "volss": volss,
        "devss": devss,
    }

    return transformation


def decomposePhi(
    Phi, PhiCentre=[0.0, 0.0, 0.0], PhiPoint=[0.0, 0.0, 0.0], twoD=False, verbose=False
):
    """
    Get components out of a linear deformation function "Phi"

    Parameters
    ----------
        Phi : 4x4 array
            The deformation function operator "Phi"

        PhiCentre : 3x1 array, optional
            Point where Phi was calculated

        PhiPoint : 3x1 array, optional
            Point where Phi is going to be applied

        twoD : bool, optional
            is the Phi defined in 2D? This applies corrections to the strain outputs
            Default = False

        verbose : boolean, optional
            Print errors?
            Default = True

    Returns
    -------
        transformation : dictionary of arrays

                - t = 3x1 numpy array: Translation vector (z, y, x)
                - r = 3x1 numpy array: Rotation in "rotation vector" format
                - z = 3x1 numpy array: Zoom in "zoom vector" format (z, y, x)
                - U = 3x3 numpy array: Right stretch tensor, U - numpy.eye(3) is the strain tensor in large strains
                - e = 3x3 numpy array: Strain tensor in small strains (symmetric)
                - vol   = float: First  invariant of the strain tensor ("Volumetric Strain"), det(F)-1
                - dev   = float: Second invariant of the strain tensor ("Deviatoric Strain")
                - volss = float: First  invariant of the strain tensor ("Volumetric Strain") in small strains, trace(e)/3
                - devss = float: Second invariant of the strain tensor ("Deviatoric Strain") in small strains

    """
    # - G = 3x3 numpy array: Eigen vectors * eigenvalues of strains, from which principal directions of strain can be obtained
    # Default non-success values to be over written if all goes well
    transformation = {
        "t": numpy.array([numpy.nan] * 3),
        "r": numpy.array([numpy.nan] * 3),
        "z": numpy.array([numpy.nan] * 3),
        "U": numpy.eye(3) * numpy.nan,
        "e": numpy.eye(3) * numpy.nan,
        "vol": numpy.nan,
        "dev": numpy.nan,
        "volss": numpy.nan,
        "devss": numpy.nan,
        # 'G': 3x3
    }

    # Check for singular Phi if yes quit
    try:
        numpy.linalg.inv(Phi)
    except numpy.linalg.linalg.LinAlgError:
        return transformation

    # Check for NaNs if any quit
    if numpy.isnan(Phi).sum() > 0:
        return transformation

    # Check for NaNs if any quit
    if numpy.isinf(Phi).sum() > 0:
        return transformation

    ###########################################################
    # F, the inside 3x3 displacement gradient
    ###########################################################
    F = Phi[0:3, 0:3].copy()

    ###########################################################
    # Calculate transformation by undoing F on the PhiPoint
    ###########################################################
    tra = Phi[0:3, 3].copy()

    # compute distance between given point and the point where Phi was calculated
    dist = numpy.array(PhiPoint) - numpy.array(PhiCentre)

    # apply Phi to the given point and calculate its displacement
    tra -= dist - numpy.dot(F, dist)

    decomposedF = decomposeF(F, verbose=verbose)

    transformation = {
        "t": tra,
        "r": decomposedF["r"],
        "z": decomposedF["z"],
        "U": decomposedF["U"],
        # 'G': G,
        "e": decomposedF["e"],
        "vol": decomposedF["vol"],
        "dev": decomposedF["dev"],
        "volss": decomposedF["volss"],
        "devss": decomposedF["devss"],
    }

    return transformation
