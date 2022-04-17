npoint = 400  # Number of particles
nframe = 500  # Number of frames
xmin, xmax, ymin, ymax = 0, 1, 0, 1  # Boundaries of Box
Dt = 0.00002  # time step
r = 0.0015  # Radius of particle
m = 2.672 * 10 ** -26  # Mass of Particle
kB = 1.38 * 10 ** -23  # Boltzmann Constant
ind = np.arange(0, npoint)  # Array of indices
# Array of all indices of combinations
indices = np.asarray(list(combinations(ind, 2)))


def MBSpeed(v, T):
    """
    Returns the Maxwell-Boltzman speed of a particle

    v is the final velocity of the particle
    T is the temperature of the system
    """
    return ((m * v) / (kB * T) * np.exp((-m * v ** 2)
                                        / (2 * kB * T)))


def BDist(E, T):
    """
    Returns the Boltzman distribution of energy

    v is the final velocity of the particle
    T is the temperature of the system
    """
    return (1 / (kB * T) * np.exp((-E)
                                  / (kB * T)))


def onCollision(v1, v2, r1, r2):
    """
    Returns the changed velocity of both particles after a collision

    v1 is the velocity vector of the first particle
    v2 is the velocity vector of the second particle
    r1 is the position vector of the first particle
    r2 is the position vector of the second particle
    """
    drx = r1[0] - r2[0]
    dry = r1[1] - r2[1]

    v1p = v1 - (np.dot(v1-v2, r1-r2)
                / (drx ** 2 + dry ** 2)) * (r1 - r2)
    v2p = v2 - (np.dot(v2-v1, r2-r1)
                / (drx ** 2 + dry ** 2)) * (r2 - r1)
    return [v1p, v2p]


def update_point(num):
    """
    Calculates the new position of each particle at each timestep.
    Also sets the bound of the particle's movement to the size
    of the box.

    num is the frame number
    """
    global x, y, vx, vy
    # provide some feedback so the user can see progress
    print('.', end='')
    for i in range(2):
        # Calculate the new position of the particles
        dx = Dt*vx
        dy = Dt*vy
        x = x + dx
        y = y + dy
        indx = np.where((x < xmin) | (x > xmax))
        indy = np.where((y < ymin) | (y > ymax))
        vx[indx] = -vx[indx]
        vy[indy] = -vy[indy]
        xx = np.asarray(list(combinations(x, 2)))
        yy = np.asarray(list(combinations(y, 2)))
        # Distances between particles:
        dd = (xx[:, 0]-xx[:, 1])**2+(yy[:, 0]-yy[:, 1])**2
        # Find which particles are colliding
        colls = indices[np.where(dd <= 4 * r**2)]
        for pair in colls:
            # Calculate the new velocities of the particles as vectors
            v1 = np.array([vx[pair[0]], vy[pair[0]]])
            v2 = np.array([vx[pair[1]], vy[pair[1]]])
            r1 = np.array([x[pair[0]], y[pair[0]]])
            r2 = np.array([x[pair[1]], y[pair[1]]])
            V1, V2 = onCollision(v1, v2, r1, r2)
            # Update the components of both particles' velocities
            vx[pair[0]] = V1[0]
            vx[pair[1]] = V2[0]
            vy[pair[0]] = V1[1]
            vy[pair[1]] = V2[1]
    data = np.stack((x, y), axis=-1)
    im.set_offsets(data)


# Create a box to bind the particles to
fig, ax = plt.subplots()
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# Create the particles and give their initial movement
x = np.random.random(npoint)
y = np.random.random(npoint)
vx = -500. * np.ones(npoint)
vy = np.zeros(npoint)

# "Bounce" the particle if it collides with the edges
vx[np.where(x <= 0.5)] = -vx[np.where(x <= 0.5)]

# Set the initial colour of the particles depending on position
col = np.where(x < 0.5, 'b', 'r')
im = ax.scatter(x, y, c=col)
im.set_sizes([20])

# Create the animation and display it
anim = animation.FuncAnimation(fig,
                               update_point, nframe, interval=30, repeat=False)
anim.save('collide.webm', extra_args=['-vcodec', 'libvpx'])
plt.close()
HTML('<video controls autoplay> <source src="collide.webm" '
     + 'type="video/webm"></video>')