import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from spins import goos
from spins.goos import compat
from spins.goos_sim import maxwell
from spins.invdes.problem_graph import optplan


def main(save_folder: str,
         min_feature: float = 100,
         use_cubic: bool = True,
         visualize: bool = False) -> None:
    goos.util.setup_logging(save_folder)

    with goos.OptimizationPlan(save_path=save_folder) as plan:
        wg_1 = goos.Cuboid(pos=goos.Constant([1600, -4500, 0]),
                            extents=goos.Constant([600, 1000, 230]),
                            material=goos.material.Material(index=3.45))
        wg_2 = goos.Cuboid(pos=goos.Constant([-1600, -4500, 0]),
                            extents=goos.Constant([600, 1000, 230]),
                            material=goos.material.Material(index=3.45))
        wg_3 = goos.Cuboid(pos=goos.Constant([1600, 4500, 0]),
                            extents=goos.Constant([600, 1000, 230]),
                            material=goos.material.Material(index=3.45))
        wg_4 = goos.Cuboid(pos=goos.Constant([-1600, 4500, 0]),
                            extents=goos.Constant([600, 1000, 230]),
                            material=goos.material.Material(index=3.45))
        wg_5 = goos.Cuboid(pos=goos.Constant([4500, -1600, 0]),
                            extents=goos.Constant([1000, 600, 230]),
                            material=goos.material.Material(index=3.45))
        wg_6 = goos.Cuboid(pos=goos.Constant([4500, 1600, 0]),
                            extents=goos.Constant([1000, 600, 230]),
                            material=goos.material.Material(index=3.45))
        wg_7 = goos.Cuboid(pos=goos.Constant([-4500, 0, 0]),
                            extents=goos.Constant([1000, 600, 230]),
                            material=goos.material.Material(index=3.45))
        substrate = goos.Cuboid(pos=goos.Constant([0, 0, -500]),
                            extents=goos.Constant([11000, 11000, 770]),
                            material=goos.material.Material(index=1.4))

        def initializer(size):
            # Set the seed immediately before calling `random` to ensure
            # reproducibility.
            #np.random.seed(247)
            return np.ones(size)*0.5
            #return np.random.random(size) * 0.2 + 0.5

        # Continuous optimization.
        var, design = goos.pixelated_cont_shape(
            initializer=initializer,
            pos=goos.Constant([0, 0, 0]),
            extents=[8000, 8000, 230],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=3.45),
            pixel_size=[50, 50, 230],
            var_name="var_cont")

        sigmoid_factor = goos.Variable(4, parameter=True, name="discr_factor")
        design = goos.cast(goos.Sigmoid(sigmoid_factor * (2 * design - 1)),
                           goos.Shape)
        eps = goos.GroupShape([wg_1, wg_2, wg_3, wg_4, wg_5, wg_6, wg_7, design, substrate])

        # This node is purely for debugging purposes.
        eps_rendered = maxwell.RenderShape(
            eps,
            region=goos.Box3d(center=[0, 0, 0], extents=[10000, 10000, 0]),
            mesh=maxwell.UniformMesh(dx=50),
            wavelength=635,
        )
        if visualize:
            goos.util.visualize_eps(eps_rendered.get().array[2])

        obj, sim1, sim2, sim3, sim4, sim5, sim6, sim7 = make_objective(eps, "cont")

        for factor in [4,8,12]:
            sigmoid_factor.set(factor)
            goos.opt.scipy_minimize(
                obj,
                "L-BFGS-B",
                monitor_list=[sim1["eps"], sim1["field"], sim1["overlap1"], sim1["overlap2"], sim1["overlap3"], sim1["overlap4"], sim1["overlap5"], sim1["overlap6"], sim1["overlap7"],
                                            sim2["eps"], sim2["field"], sim2["overlap1"], sim2["overlap2"], sim2["overlap3"], sim2["overlap4"], sim2["overlap5"], sim2["overlap6"], sim2["overlap7"],
                                            sim3["eps"], sim3["field"], sim3["overlap1"], sim3["overlap2"], sim3["overlap3"], sim3["overlap4"], sim3["overlap5"], sim3["overlap6"], sim3["overlap7"],
                                            sim4["eps"], sim4["field"], sim4["overlap1"], sim4["overlap2"], sim4["overlap3"], sim4["overlap4"], sim4["overlap5"], sim4["overlap6"], sim4["overlap7"],
                                            sim5["eps"], sim5["field"], sim5["overlap1"], sim5["overlap2"], sim5["overlap3"], sim5["overlap4"], sim5["overlap5"], sim5["overlap6"], sim5["overlap7"],
                                            sim6["eps"], sim6["field"], sim6["overlap1"], sim6["overlap2"], sim6["overlap3"], sim6["overlap4"], sim6["overlap5"], sim6["overlap6"], sim6["overlap7"],
                                            sim7["eps"], sim7["field"], sim7["overlap1"], sim7["overlap2"], sim7["overlap3"], sim7["overlap4"], sim7["overlap5"], sim7["overlap6"], sim7["overlap7"],
                                            obj],
                max_iters=25,
                name="opt_cont{}".format(factor))

        plan.save()
        plan.run()

        if visualize:
            goos.util.visualize_eps(eps_rendered.get().array[2])


def make_objective(eps: goos.Shape, stage: str):
    sim_z_extent = 1800
    simcenter_positon_z = 300
    sources_wx = 8000
    sources_wy = 8000
    solver_info = maxwell.MaxwellSolver(solver="maxwell_cg", err_thresh=1e-2)
    pml_thickness = [400, 400, 400, 400, 400, 400]
    sources_position_z = 800
    wavelength=635
    background=goos.material.Material(index=1.0)
    simulation_space=maxwell.SimulationSpace(
        mesh=maxwell.UniformMesh(dx=50),
        sim_region=goos.Box3d(
            center=[0, 0, simcenter_positon_z],
            extents=[10000, 10000, sim_z_extent],),
        pml_thickness=pml_thickness)
    sources1 = [maxwell.LaguerreGaussianSource(
        w0 = 1400,
        center = [0, 0, sources_position_z],
        beam_center =  [0, 0, sources_position_z],
        extents=[sources_wx, sources_wy, 0],
        normal=[0, 0, -1],
        theta = 0,
        psi = 0,
        m = 2,
        p = 0,
        polarization_angle = 0,
        power=1),]
    sources2 = [maxwell.LaguerreGaussianSource(
        w0 = 1400,
        center = [0, 0, sources_position_z],
        beam_center =  [0, 0, sources_position_z],
        extents=[sources_wx, sources_wy, 0],
        normal=[0, 0, -1],
        theta = 0,
        psi = 0,
        m = 3,
        p = 0,
        polarization_angle = 0,
        power=1),]
    sources3 = [maxwell.LaguerreGaussianSource(
        w0 = 1400,
        center = [0, 0, sources_position_z],
        beam_center =  [0, 0, sources_position_z],
        extents=[sources_wx, sources_wy, 0],
        normal=[0, 0, -1],
        theta = 0,
        psi = 0,
        m = -3,
        p = 0,
        polarization_angle = 0,
        power=1),]
    sources4 = [maxwell.LaguerreGaussianSource(
        w0 = 1400,
        center = [0, 0, sources_position_z],
        beam_center =  [0, 0, sources_position_z],
        extents=[sources_wx, sources_wy, 0],
        normal=[0, 0, -1],
        theta = 0,
        psi = 0,
        m = -2,
        p = 0,
        polarization_angle = 0,
        power=1),]
    sources5 = [maxwell.LaguerreGaussianSource(
        w0 = 1400,
        center = [0, 0, sources_position_z],
        beam_center =  [0, 0, sources_position_z],
        extents=[sources_wx, sources_wy, 0],
        normal=[0, 0, -1],
        theta = 0,
        psi = 0,
        m = 1,
        p = 0,
        polarization_angle = 0,
        power=1),]
    sources6 = [maxwell.LaguerreGaussianSource(
        w0 = 1400,
        center = [0, 0, sources_position_z],
        beam_center =  [0, 0, sources_position_z],
        extents=[sources_wx, sources_wy, 0],
        normal=[0, 0, -1],
        theta = 0,
        psi = 0,
        m = -1,
        p = 0,
        polarization_angle = 0,
        power=1),]
    sources7 = [maxwell.LaguerreGaussianSource(
        w0 = 1400,
        center = [0, 0, sources_position_z],
        beam_center =  [0, 0, sources_position_z],
        extents=[sources_wx, sources_wy, 0],
        normal=[0, 0, -1],
        theta = 0,
        psi = 0,
        m = 0,
        p = 0,
        polarization_angle = 0,
        power=1),]
    outputs=[maxwell.Epsilon(name="eps"),
        maxwell.ElectricField(name="field"),
        maxwell.WaveguideModeOverlap(name="overlap1",
                                         center=[1600, -4500, 0],
                                         extents=[800, 0, 400],
                                         normal=[0, -1, 0],
                                         mode_num=0,
                                         power=1),
        maxwell.WaveguideModeOverlap(name="overlap2",
                                         center=[-1600, -4500, 0],
                                         extents=[800, 0, 400],
                                         normal=[0, -1, 0],
                                         mode_num=0,
                                         power=1),
        maxwell.WaveguideModeOverlap(name="overlap3",
                                         center=[1600, 4500, 0],
                                         extents=[800, 0, 400],
                                         normal=[0, 1, 0],
                                         mode_num=0,
                                         power=1),
        maxwell.WaveguideModeOverlap(name="overlap4",
                                         center=[-1600, 4500, 0],
                                         extents=[800, 0, 400],
                                         normal=[0, 1, 0],
                                         mode_num=0,
                                         power=1),
        maxwell.WaveguideModeOverlap(name="overlap5",
                                         center=[4500, -1600, 0],
                                         extents=[0, 800, 400],
                                         normal=[1, 0, 0],
                                         mode_num=0,
                                         power=1),
        maxwell.WaveguideModeOverlap(name="overlap6",
                                         center=[4500, 1600, 0],
                                         extents=[0, 800, 400],
                                         normal=[1, 0, 0],
                                         mode_num=0,
                                         power=1),
        maxwell.WaveguideModeOverlap(name="overlap7",
                                         center=[-4500, 0, 0],
                                         extents=[0, 800, 400],
                                         normal=[-1, 0, 0],
                                         mode_num=0,
                                         power=1),]
    sim1 = maxwell.fdfd_simulation(
        name="sim1_{}".format(stage),
        wavelength=wavelength,
        eps=eps,
        solver_info=solver_info,
        sources =  sources1,
        simulation_space=simulation_space,
        background=background,
        outputs=outputs,)
    sim2 = maxwell.fdfd_simulation(
        name="sim2_{}".format(stage),
        wavelength=wavelength,
        eps=eps,
        solver_info=solver_info,
        sources =  sources2,
        simulation_space=simulation_space,
        background=background,
        outputs=outputs,)
    sim3 = maxwell.fdfd_simulation(
        name="sim3_{}".format(stage),
        wavelength=wavelength,
        eps=eps,
        solver_info=solver_info,
        sources =  sources3,
        simulation_space=simulation_space,
        background=background,
        outputs=outputs,)
    sim4 = maxwell.fdfd_simulation(
        name="sim4_{}".format(stage),
        wavelength=wavelength,
        eps=eps,
        solver_info=solver_info,
        sources =  sources4,
        simulation_space=simulation_space,
        background=background,
        outputs=outputs,)
    sim5 = maxwell.fdfd_simulation(
        name="sim5_{}".format(stage),
        wavelength=wavelength,
        eps=eps,
        solver_info=solver_info,
        sources =  sources5,
        simulation_space=simulation_space,
        background=background,
        outputs=outputs,)
    sim6 = maxwell.fdfd_simulation(
        name="sim6_{}".format(stage),
        wavelength=wavelength,
        eps=eps,
        solver_info=solver_info,
        sources =  sources6,
        simulation_space=simulation_space,
        background=background,
        outputs=outputs,)
    sim7 = maxwell.fdfd_simulation(
        name="sim7_{}".format(stage),
        wavelength=wavelength,
        eps=eps,
        solver_info=solver_info,
        sources =  sources7,
        simulation_space=simulation_space,
        background=background,
        outputs=outputs,)

    obj1 = goos.rename(((goos.abs(sim1["overlap1"])-12)**2+goos.abs(sim1["overlap2"])**2+goos.abs(sim1["overlap3"])**2+goos.abs(sim1["overlap4"])**2+goos.abs(sim1["overlap5"])**2+goos.abs(sim1["overlap6"])**2+goos.abs(sim1["overlap7"])**2), name="obj1_{}".format(stage))
    obj2 = goos.rename((goos.abs(sim2["overlap1"])**2+(goos.abs(sim2["overlap2"])-12)**2+goos.abs(sim2["overlap3"])**2+goos.abs(sim2["overlap4"])**2+goos.abs(sim2["overlap5"])**2+goos.abs(sim2["overlap6"])**2+goos.abs(sim2["overlap7"])**2), name="obj2_{}".format(stage))
    obj3 = goos.rename((goos.abs(sim3["overlap1"])**2+goos.abs(sim3["overlap2"])**2+(goos.abs(sim3["overlap3"])-12)**2+goos.abs(sim3["overlap4"])**2+goos.abs(sim3["overlap5"])**2+goos.abs(sim3["overlap6"])**2+goos.abs(sim3["overlap7"])**2), name="obj3_{}".format(stage))
    obj4 = goos.rename((goos.abs(sim4["overlap1"])**2+goos.abs(sim4["overlap2"])**2+goos.abs(sim4["overlap3"])**2+(goos.abs(sim4["overlap4"])-12)**2+goos.abs(sim4["overlap5"])**2+goos.abs(sim4["overlap6"])**2+goos.abs(sim4["overlap7"])**2), name="obj4_{}".format(stage))
    obj5 = goos.rename((goos.abs(sim5["overlap1"])**2+goos.abs(sim5["overlap2"])**2+goos.abs(sim5["overlap3"])**2+goos.abs(sim5["overlap4"])**2+(goos.abs(sim5["overlap5"])-12)**2+goos.abs(sim5["overlap6"])**2+goos.abs(sim5["overlap7"])**2), name="obj5_{}".format(stage))
    obj6 = goos.rename((goos.abs(sim6["overlap1"])**2+goos.abs(sim6["overlap2"])**2+goos.abs(sim6["overlap3"])**2+goos.abs(sim6["overlap4"])**2+goos.abs(sim6["overlap5"])**2+(goos.abs(sim6["overlap6"])-12)**2+goos.abs(sim6["overlap7"])**2), name="obj5_{}".format(stage))
    obj7 = goos.rename((goos.abs(sim7["overlap1"])**2+goos.abs(sim7["overlap2"])**2+goos.abs(sim7["overlap3"])**2+goos.abs(sim7["overlap4"])**2+goos.abs(sim7["overlap5"])**2+goos.abs(sim7["overlap6"])**2+(goos.abs(sim7["overlap7"])-12)**2), name="obj5_{}".format(stage))
    obj = obj1+obj2+obj3+obj4+2*obj5+2*obj6+8*obj7
    return obj, sim1, sim2, sim3, sim4, sim5, sim6, sim7


def visualize(folder: str, step: int):
    """Visualizes result of the optimization.

    This is a quick visualization tool to plot the permittivity and electric
    field distribution at a particular save step. The function automatically
    determines whether the optimization is in continuous or discrete and
    plot the appropriate data.

    Args:
       folder: Save folder location.
       step: Save file step to load.
    """
    if step is None:
        step = goos.util.get_latest_log_step(folder)

    step = int(step)

    with open(os.path.join(folder, "step{0}.pkl".format(step)), "rb") as fp:
        data = pickle.load(fp)

    plt.figure()
    plt.subplot(1, 2, 1)
    eps = np.linalg.norm(data["monitor_data"]["sim1_cont.eps"], axis=0)
    plt.imshow(eps[:, :, 16].squeeze())
    plt.colorbar()
    plt.subplot(1, 2, 2)
    field_norm = np.linalg.norm(data["monitor_data"]["sim1_cont.field"], axis=0)
    plt.imshow(field_norm[:, :, 28].squeeze())
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "view"))
    parser.add_argument("save_folder")
    parser.add_argument("--step")

    args = parser.parse_args()
    if args.action == "run":
        main(args.save_folder,  visualize=True)
    elif args.action == "view":
        visualize(args.save_folder, args.step)
