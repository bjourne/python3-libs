import numpy as np
import matplotlib.pyplot as plt
import time

# Makes it so numpy always outputs the same random numbers which is
# useful during development. Comment out the line in production.
np.random.seed(1234)

def main():
    twopi = 2 * np.pi
    img = np.cdouble(0+1j)


    wavelength = 0.08
    k0  = twopi/wavelength

    z_noise = 0.05 * wavelength

    N, M = 100, 50
    x = np.arange(-N, N+1, dtype = 'f8')

    # Use random Z noise for now
    Z = z_noise * np.random.normal(size = (x.shape[0], x.shape[0])).astype('f8')

    A = np.linspace(0, 0.2, M, dtype = 'f8')

    tstart = time.time()
    answers = []

    # X and Y with shapes (x, 1) and (1, x) contains the same data.
    X = x[np.newaxis,:]
    Y = x[:,np.newaxis]

    # Compute squared Euclidean distances with shape (x, x).
    Rsq = X**2 + Y**2

    # Compute other distances with shape (M, M).
    a = A[:,np.newaxis]
    b = A[np.newaxis,:]
    SQ = 1 + np.sqrt(1 - a**2 - b**2)

    # Add noise thing (M, M, x, x).
    SQ_Z = SQ[:,:,np.newaxis,np.newaxis] * Z

    # Gets shape (M, 1, x) and (M, x, 1).
    A_X = A[:,np.newaxis,np.newaxis] * X
    A_Y = A[:,np.newaxis,np.newaxis] * Y

    # Calculates A*X + A*Y with shape (M, M, x, x).
    A_X_Y = A_X[:,np.newaxis] + A_Y[np.newaxis,:]

    # e^{-j*phases} with shape (M, M, x, x).
    A_stuff = np.exp(-img * k0 * (A_X_Y + SQ_Z))
    for w_beam in (1, 2, 4, 8):
        coeff = np.exp(-Rsq/w_beam**2)
        # Sums over the last two dimensions, shape becomes (M, M).
        E = np.sum(A_stuff * coeff, axis = (2, 3)) / w_beam**2
        answers.append(E)

    # Print checksum. Useful for debugging.
    print(sum(np.sum(E) for e in answers))
    print(f'Time: {time.time() - tstart:.3}')
    if True:
        plt.figure()
        for i, E in enumerate(answers):
            plt.subplot(2, 2, i+1)
            plt.imshow(np.log10(np.abs(E)), vmin=0.0001)
            plt.colorbar()
        plt.show()

if __name__ == '__main__':
    main()
