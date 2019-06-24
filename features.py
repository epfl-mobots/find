import numpy as np

class Velocities:
    def __init__(self, positions, timestep):
        self._velocities = []
        for p in positions:            
            rolled_p = np.roll(p, shift=1, axis=0)
            velocities = (rolled_p - p) / timestep
            sigma = np.std(velocities[:-1, :], axis=0)
            mu = np.mean(velocities[:-1, :], axis=0)

            x_rand = np.random.normal(mu[0], sigma[0], p.shape[1] // 2)
            y_rand = np.random.normal(mu[1], sigma[1], p.shape[1] // 2)
            for i in range(p.shape[1] // 2):
                velocities[-1, i * 2] = x_rand[i]
                velocities[-1, i * 2 + 1] = y_rand[i] 
            self._velocities.append(velocities)       


    def get(self):
        return self._velocities