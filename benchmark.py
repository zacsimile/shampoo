
from shampoo import Hologram
import numpy as n

if __name__ == '__main__':
    t = Hologram(n.random.rand(128, 128), wavelength = 800)
    t.reconstruct(1)