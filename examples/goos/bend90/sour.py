
'''
Import necessary python libraries and SPINS-B libraries.
'''

import datetime
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from spins import goos
from spins.goos import compat
from spins.goos_sim import maxwell
from spins.invdes.problem_graph import optplan, log_tools

'''
Set up Optimization Plan object.
'''
## Create folder where all spins-b results will be saved. ##

# Currently, the folder will be named "bend90_{current_timestamp}" and will
# be located in your Drive folder containing spins-b. To change this, change
# the following line and set `out_folder_name` somewhere else.
out_folder_name = "bend90_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
out_folder = os.path.join(os.getcwd(),out_folder_name)
if (not os.path.exists(out_folder)):
  os.makedirs(out_folder)

## Setup logging and Optimization Plan. ##
plan = goos.OptimizationPlan(save_path = out_folder)

'''
Define the constant background structures that will not be changed
during the design. In this case, these are the input and output waveguides.
'''
with plan:
  # Define input waveguide.
  wg_in = goos.Cuboid(pos=goos.Constant([-2000, 0, 0]),
                            extents=goos.Constant([3000, 400, 220]),
                            material=goos.material.Material(index=1))
  # Define output waveguide.
  wg_out = goos.Cuboid(pos=goos.Constant([0, 2000, 0]),
                             extents=goos.Constant([400, 3000, 220]),
                             material=goos.material.Material(index=1))
  
  # Group these background structures together.
  eps_background_structures = goos.GroupShape([wg_in, wg_out])

  '''
Visualize the constant background structures we just defined.
'''
with plan:
  eps_rendered = maxwell.RenderShape(
            eps_background_structures,
            region=goos.Box3d(center=[0, 0, 0], extents=[3000, 3000, 0]),
            mesh=maxwell.UniformMesh(dx=40),
            wavelength=635,
        )
  
  goos.util.visualize_eps(eps_rendered.get().array[2])

'''
Define and initialize the design region.
'''
with plan:
  # Use random initialization, where each pixel is randomly assigned
  # a value in the range [0.3,0.7].
  def initializer(size):
            # Set the seed immediately before calling `random` to ensure
            # reproducibility.
            np.random.seed(247)
            return np.random.random(size) * 0.2 + 0.5
  
  # Define the design region as a pixelated continuous shape, which is composed
  # of voxels whose permittivities can take on any value between `material` and
  # `material2`.
  var, design = goos.pixelated_cont_shape(
                initializer=initializer,
                pos=goos.Constant([0, 0, 0]),
                extents=[2000, 2000, 220],
                material=goos.material.Material(index=1),
                material2=goos.material.Material(index=1),
                pixel_size=[40, 40, 220],
                var_name="var_cont")
  
  eps = goos.GroupShape([design,])


"""
Visualize the design region as a sanity check.
"""
with plan:
  eps_rendered = maxwell.RenderShape(
            eps,
            region=goos.Box3d(center=[0, 0, 0], extents=[3000, 3000, 0]),
            mesh=maxwell.UniformMesh(dx=40),
            wavelength=1550,
        )
  goos.util.visualize_eps(eps_rendered.get().array[2])

'''
Setup EM solver - in this case, we use the FDFD solver that comes with 
spins-b, maxwell.
'''
with plan:
  # Define wavelength and solver.
  my_wavelength = 633
  my_solver = "maxwell_cg"

  # Define simulation space.
  my_simulation_space = maxwell.SimulationSpace(
            mesh=maxwell.UniformMesh(dx=40),
            sim_region=goos.Box3d(
                center=[0, 0, 0],
                extents=[6000, 6000, 2000],
            ),
            pml_thickness=[400, 400, 400, 400, 400, 400])
  
  # Define a waveguide mode source.
  my_sources = [maxwell.LaguerreGaussianSource(
        w0 = 1000,
        center = [0, 0, 0],
        beam_center =  [0, 0, 0],
        extents=[3000, 3000, 0],
        normal=[0, 0, -1],
        theta = 0,
        psi = 0,
        m = 1,
        p = 0,
        polarization_angle = 0,
        power=1),]
  
  
  # Define simulation outputs.
  my_outputs=[ maxwell.Epsilon(name="eps"),
               maxwell.ElectricField(name="field"),
               maxwell.WaveguideModeOverlap(name="overlap",
                                         center=[0, 1400, 0],
                                         extents=[2500, 0, 1000],
                                         normal=[0, 1, 0],
                                         mode_num=0,
                                         power=1)]

  # Setup the simulation object.
  sim = maxwell.fdfd_simulation(
        name="sim_cont",
        wavelength= my_wavelength,
        background=goos.material.Material(index=1.0),
        eps=eps,
        simulation_space = my_simulation_space,
        solver = my_solver,
        sources = my_sources,
        outputs= my_outputs
    )


'''
Visualize simulation of initial structure, as a sanity check.
'''
with plan:
  initial_structure = np.squeeze(sim['eps'].get().array)
  initial_field = np.squeeze(sim['field'].get().array)
  initial_overlap = np.squeeze(sim['overlap'].get().array)

  plt.figure()
  plt.subplot(1,2,1)
  plt.imshow(
    np.abs(
      initial_structure[2,:,:,25]))
  plt.colorbar()

  plt.subplot(1,2,2)
  plt.imshow(
    np.abs(
      initial_field[0,:,:,24]))
  plt.colorbar()
  plt.show()

  print("Initial overlap is {}.".format(np.abs(initial_overlap)))
  

