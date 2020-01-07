import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Body():
    """
    This class contains adjustable parameters as attributes. These
    parameters include current and previous positions, velocities, and
    accelerations.
    """
    def __init__(self,
                 id, facecolor, pos,
                 mass=1, vel=None, acc=None):
        self.id = id
        self.facecolor = facecolor
        self.pos = np.array(pos, dtype=float)
        self.mass = mass
        self.vel = self.autocorrect_parameter(vel)
        self.acc = self.autocorrect_parameter(acc)
        self.vector_pos = [self.pos]
        self.vector_vel = [self.vel]
        self.vector_acc = [self.acc]

    def autocorrect_parameter(self, args):
        if args is None:
            return np.zeros(self.pos.shape, dtype=float)
        return np.array(args, dtype=float)

    def scalar_pos(self):
        return np.sqrt(np.sum(np.square(self.vector_pos), axis=1))

class GravitationalDynamics():
    """
    This class contains methods to run a simulation of N bodies that interact
    gravitationally over some time. Each body in the simulation keeps a record
    of parameters (pos, vel, and acc) as time progresses.
    """
    def __init__(self, bodies, t=0, gravitational_constant=6.67e-11):
        self.bodies = bodies
        self.nbodies = len(bodies)
        self.ndim = len(bodies[0].pos)
        self.t = t
        self.moments = [t]
        self.gravitational_constant = gravitational_constant

    def get_acc(self, ibody, jbody):
        dpos = ibody.pos - jbody.pos
        r = np.sum(dpos**2)
        acc = -1 * self.gravitational_constant * jbody.mass \
            * dpos / np.sqrt(r**3)
        return acc

    def update_accelerations(self):
        for ith_body, ibody in enumerate(self.bodies):
            ibody.acc *= 0
            for jth_body, jbody in enumerate(self.bodies):
                if ith_body != jth_body:
                    acc = self.get_acc(ibody, jbody)
                    ibody.acc += acc
            ibody.vector_acc.append(np.copy(ibody.acc))

    def update_velocities_and_positions(self, dt):
        for ibody in self.bodies:
            ibody.vel += ibody.acc * dt
            ibody.pos += ibody.vel * dt
            ibody.vector_vel.append(np.copy(ibody.vel))
            ibody.vector_pos.append(np.copy(ibody.pos))

    def simulate(self, dt, duration):
        nsteps = int(duration / dt)
        for ith_step in range(nsteps):
            self.update_accelerations()
            self.update_velocities_and_positions(dt)
            self.t += dt
            self.moments.append(self.t)

# Masses
SOLAR_MASS = 1.988e30
EARTH_MASS = 5.9722e24

# Distances
ASTRO_UNIT = 1.496e11
MILE = 1609

# Durations
HOUR = 3600
DAY = 24 * HOUR
YEAR = 365 * DAY

def main():
    m_sun = 1 * SOLAR_MASS

    sun = Body('Sun', 'yellow', [0, 0, 0], m_sun)

    m_mercury = 0.05227 * EARTH_MASS
    d_mercury = 0.4392 * ASTRO_UNIT
    v_mercury = (106_000 * MILE) / (1 * HOUR)
    mercury = Body('Mercury', 'gray',
                   [d_mercury, 0, 0], m_mercury,
                   [0, v_mercury, 0])

    m_earth = 1 * EARTH_MASS
    d_earth = 1 * ASTRO_UNIT
    v_earth = (66_600 * MILE) / (1 * HOUR)
    earth = Body('Earth', 'blue', [d_earth, 0, 0], m_earth, [0, v_earth, 0])

    m_mars = 0.1704 * EARTH_MASS
    d_mars = 1.653 * ASTRO_UNIT
    v_mars = (53_900 * MILE) / (1 * HOUR)
    mars = Body('Mars', 'darkred', [0, d_mars, 0], m_mars, [v_mars, 0, 0])

    bodies = [sun, mercury, earth, mars]
    dt = 2 * DAY
    duration = 10 * YEAR

    gd = GravitationalDynamics(bodies)
    gd.simulate(dt, duration)

    fig = plt.figure(figsize=(11, 7))
    ax_left = fig.add_subplot(1, 2, 1, projection='3d')
    ax_left.set_title('3-D Position')
    ax_right = fig.add_subplot(1, 2, 2)
    ax_right.set_title('Displacement vs Time')

    ts = np.array(gd.moments) / YEAR
    xticks = np.arange(max(ts)+1).astype(int)

    for body in gd.bodies:
        vector = np.array(body.vector_pos)
        vector_coordinates = vector / ASTRO_UNIT

        scalar = body.scalar_pos()
        scalar_coordinates = scalar / ASTRO_UNIT
        ax_left.scatter(*vector_coordinates.T, marker='.',
                        c=body.facecolor, label=body.id)
        ax_right.scatter(ts, scalar_coordinates, marker='.',
                         c=body.facecolor, label=body.id)
        ax_right.set_xticks(xticks)
        ax_right.grid(color='k', linestyle=':', alpha=0.3)
        fig.subplots_adjust(bottom=0.3)
        fig.legend(loc='lower center', mode='expand', ncol=len(gd.bodies))
    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    main()
