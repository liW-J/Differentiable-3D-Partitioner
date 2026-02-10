'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2026-02-06 22:51:59
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-02-09 14:46:14
FilePath: /Differentiable-3D-Partitioner/partitioner/core/flow.py
Description: Flow for 3D Partitioner
'''
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from partitioner.core.partitioner import Partitioner
from partitioner.utils.visualize import visualize_z_single
from partitioner.utils.tensor2txt import tensor2txt
import yaml
from pathlib import Path


class Differentiable3DPartitionerFlow:

    def __init__(self,
                 num_nodes,
                 num_nets,
                 num_pins,
                 node_pos,
                 pin_pos,
                 flat_net2pin_map,
                 flat_net2pin_start_map,
                 pin2node_map,
                 node_size_x,
                 node_size_y,
                 config_path="configs/default.yaml"):
        self.num_nodes = num_nodes
        self.num_nets = num_nets
        self.num_pins = num_pins

        if isinstance(node_pos, torch.Tensor):
            self.device = node_pos.device
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.node_pos = node_pos.to(self.device)
        self.pin_pos = pin_pos.to(self.device)
        self.flat_net2pin_map = flat_net2pin_map.to(self.device).long()
        self.flat_net2pin_start_map = flat_net2pin_start_map.to(
            self.device).long()
        self.pin2node_map = pin2node_map.to(self.device).long()
        self.node_size_x = node_size_x[:num_nodes].to(self.device)
        self.node_size_y = node_size_y[:num_nodes].to(self.device)

        self.node_x = self.node_pos[:num_nodes]
        self.node_y = self.node_pos[self.node_pos.numel() // 2:self.
                                    node_pos.numel() // 2 + num_nodes]
        self.pin_pos_x = self.pin_pos[:self.pin2node_map.numel()]
        self.pin_pos_y = self.pin_pos[self.pin2node_map.numel():]

        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config = self.load_config(config_path=config_path)

    def load_config(self, config_path=None):
        """
        Load YAML configuration file
        """
        if config_path is None:
            config_path = self.project_root / "configs" / "default.yaml"

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Extract flow-related parameters from configuration
        flow_config = config.get('flow', {})
        # Training parameters
        self.num_iterations = flow_config.get('num_iterations', 5000)
        # Optimizer configuration
        optimizer_config = flow_config.get('optimizer', {})
        self.learning_rate = optimizer_config.get('learning_rate', 0.1)
        # LSE smoothing alpha scheduling
        alpha_config = flow_config.get('lse_smoothing_alpha', {})
        self.alpha_start = alpha_config.get('start', 20.0)
        self.alpha_end = alpha_config.get('end', 20.0)

        # Cutsize Loss configuration
        cutsize_config = flow_config.get('cutsize_loss', {})
        self.use_cutsize_loss = cutsize_config.get('enabled', True)
        lambda_cut_config = cutsize_config.get('lambda_cut', {})
        self.lambda_cut_start = lambda_cut_config.get('start', 10.0)
        self.lambda_cut_end = lambda_cut_config.get('end', 10000.0)
        self.lambda_cut_gamma = lambda_cut_config.get('gamma', 0.2)

        # Balance Loss configuration
        balance_config = flow_config.get('balance_loss', {})
        self.use_balance_loss = balance_config.get('enabled', True)
        lambda_balance_config = balance_config.get('lambda_balance', {})
        self.lambda_balance_start = lambda_balance_config.get('start', 0.0)
        self.lambda_balance_end = lambda_balance_config.get('end', 5.0)
        self.lambda_balance_gamma = lambda_balance_config.get('gamma', 10.0)

        # Wirelength Loss configuration
        wirelength_config = flow_config.get('wirelength_loss', {})
        self.lambda_wl = wirelength_config.get('lambda_wl', 1.0)

        # Visualization configuration
        viz_config = config.get('visualization', {})
        self.log_interval = viz_config.get('log_interval', 10)
        self.save_interval = viz_config.get('save_interval', 50)

        # Output configuration
        output_config = config.get('output', {})
        self.result_dir = output_config.get('result_dir')

        return config

    def run(self):
        """
        main function for differentiable partitioner
        """
        print("=" * 60)
        print("Differentiable 3D Partitioner")
        print("=" * 60)

        # load real circuit data
        print("\n1. Load real circuit data...")

        print(f"   - Number of cells: {self.num_nodes}")
        print(f"   - Number of nets: {self.num_nets}")
        print(f"   - Number of pins: {self.num_pins}")
        print(
            f"   - Coordinate range: x=[{self.pin_pos_x.min():.2f}, {self.pin_pos_x.max():.2f}], "
            f"y=[{self.pin_pos_y.min():.2f}, {self.pin_pos_y.max():.2f}]")

        # initialize partitioner
        print("\n2. Initialize partitioner...")
        model = Partitioner(
            num_nodes=self.num_nodes,
            flat_net2pin_map=self.flat_net2pin_map,
            flat_net2pin_start_map=self.flat_net2pin_start_map,
            pin2node_map=self.pin2node_map,
            pin_pos_x=self.pin_pos_x,
            pin_pos_y=self.pin_pos_y,
            node_x=self.node_x,
            node_y=self.node_y,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            alpha=1.0,
            config=self.config.get('partitioner'),
        )
        model = model.to(self.device)
        print(f"   - Initial alpha: {model.alpha}")
        print(
            f"   - Number of trainable parameters: {sum(p.numel() for p in model.parameters())}"
        )

        # print initial state
        print("\n3. Initial state:")
        use_debug_info = self.use_cutsize_loss or self.use_balance_loss
        if use_debug_info:
            initial_loss, debug_info = model(
                lambda_wl=self.lambda_wl,
                lambda_cut=self.lambda_cut_start
                if self.use_cutsize_loss else 0.0,
                lambda_balance=self.lambda_balance_start
                if self.use_balance_loss else 0.0,
                return_debug_info=True)
            print(f"   - Initial total loss: {initial_loss.item():.4f}")
            print(f"   - Initial HPWL: {debug_info['L_WL']:.4f}")
            if self.use_cutsize_loss:
                print(f"   - Initial cutsize: {debug_info['L_cut']:.4f}")
            if self.use_balance_loss:
                print(f"   - Initial balance: {debug_info['L_balance']:.4f}")
        else:
            initial_loss = model()
            print(f"   - Initial total HPWL: {initial_loss.item():.4f}")
        stats = model.get_assignment_stats()
        print(
            f"   - z statistics: mean={stats['z_mean']:.4f}, std={stats['z_std']:.4f}"
        )
        print(
            f"   - Number of top cells: {stats['top_cells']}, number of bottom cells: {stats['bottom_cells']}"
        )

        # set optimizer
        print("\n4. Set optimizer...")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        print(f"   - Optimizer: Adam")
        print(f"   - Learning rate: {self.learning_rate}")

        # training parameters
        alpha_schedule = np.linspace(self.alpha_start, self.alpha_end,
                                     self.num_iterations)

        # lambda_cut schedule (exponentially increase cutsize loss weight)
        if self.use_cutsize_loss:
            t = np.linspace(0.0, 1.0, self.num_iterations)
            lambda_cut_schedule = self.lambda_cut_start + \
                (self.lambda_cut_end - self.lambda_cut_start) * (t ** self.lambda_cut_gamma)
        else:
            lambda_cut_schedule = None

        # lambda_balance schedule (linear schedule for balance loss weight)
        if self.use_balance_loss:
            t = np.linspace(0.0, 1.0, self.num_iterations)
            lambda_balance_schedule = self.lambda_balance_start + \
                (self.lambda_balance_end - self.lambda_balance_start) * (t ** self.lambda_balance_gamma)
        else:
            lambda_balance_schedule = None

        # Plot schedule trends before breakpoint
        visualization_dir = os.path.join(
            self.project_root, self.config['visualization']['output_dir'], self.config['design']['name'])
        os.makedirs(visualization_dir, exist_ok=True)

        print(f"\n5. Start training ({self.num_iterations} iterations)...")
        print(f"   - Alpha schedule: {self.alpha_start} → {self.alpha_end}")
        if self.use_cutsize_loss:
            print(
                f"   - Cutsize loss enabled: λ_cut exponential schedule {self.lambda_cut_start} → {self.lambda_cut_end}, applied to {self.num_nets} nets"
            )
        else:
            print(f"   - Cutsize loss disabled")
        if self.use_balance_loss:
            print(
                f"   - Balance loss enabled: λ_balance linear schedule {self.lambda_balance_start} → {self.lambda_balance_end}"
            )
        else:
            print(f"   - Balance loss disabled")
        print("-" * 60)

        # initialize lists to store training history
        history_loss = []
        history_hpwl = []
        history_cut = []
        history_balance = []
        history_iterations = []

        # training loop
        for iteration in range(self.num_iterations):
            # update current iteration (used to switch between sigmoid and gumbel_softmax)
            model.current_iteration = iteration
            model.alpha = alpha_schedule[iteration]

            # update lambda_cut (gradually increase cutsize loss weight)
            if self.use_cutsize_loss:
                lambda_cut = lambda_cut_schedule[iteration]
            else:
                lambda_cut = 0.0

            # update lambda_balance (gradually change balance loss weight)
            if self.use_balance_loss:
                lambda_balance = lambda_balance_schedule[iteration]
            else:
                lambda_balance = 0.0

            if use_debug_info:
                loss, debug_info = model(lambda_wl=self.lambda_wl,
                                         lambda_cut=lambda_cut,
                                         lambda_balance=lambda_balance,
                                         return_debug_info=True)
                # record training history
                history_iterations.append(iteration + 1)
                history_loss.append(loss.item())
                history_hpwl.append(debug_info['L_WL'])
                history_cut.append(
                    debug_info.get('L_cut', 0.0) if self.
                    use_cutsize_loss else 0.0)
                history_balance.append(
                    debug_info.get('L_balance', 0.0) if self.
                    use_balance_loss else 0.0)
            else:
                loss = model(lambda_balance=lambda_balance)
                # record training history (only loss and hpwl available)
                history_iterations.append(iteration + 1)
                history_loss.append(loss.item())
                history_hpwl.append(
                    loss.item())  # when no debug_info, loss is HPWL
                history_cut.append(0.0)
                history_balance.append(0.0)

            # backward propagation
            optimizer.zero_grad()
            loss.backward()

            # calculate gradient statistics (for debugging)
            if model.t.grad is not None:
                t_grad_norm = model.t.grad.norm().item()
                # check if there are NaN or Inf gradients
                if torch.isnan(model.t.grad).any() or torch.isinf(
                        model.t.grad).any():
                    print(
                        f"Warning: iteration {iteration+1} detected NaN/Inf gradients"
                    )
            else:
                t_grad_norm = 0.0

            # update parameters
            optimizer.step()

            # print and visualize at specified intervals
            if (iteration + 1) % self.log_interval == 0 or iteration == 0:
                z = model.get_z()
                dz_dt_norm = (z * (1 - z)).norm().item()
                stats = model.get_assignment_stats()

                if use_debug_info:
                    log_str = f"Iter {iteration+1:4d} | Loss: {loss.item():8.2f} | HPWL: {debug_info['L_WL']:8.2f}"
                    if self.use_cutsize_loss:
                        log_str += f" | Cut: {debug_info['L_cut']:6.4f} | λ_cut: {lambda_cut:6.4f}"
                    if self.use_balance_loss:
                        log_str += f" | Balance: {debug_info['L_balance']:6.4f} | λ_balance: {lambda_balance:6.4f}"
                    log_str += f" | Alpha: {model.alpha:5.2f} | ||dt||: {t_grad_norm:6.4f} | Top: {stats['top_cells']:3d} | Bottom: {stats['bottom_cells']:3d}"
                    print(log_str)
                else:
                    print(f"Iter {iteration+1:4d} | "
                          f"Loss: {loss.item():8.2f} | "
                          f"Alpha: {model.alpha:5.2f} | "
                          f"||dt||: {t_grad_norm:6.4f} | "
                          f"||dz/dt||: {dz_dt_norm:6.4f} | "
                          f"Top: {stats['top_cells']:3d} | "
                          f"Bottom: {stats['bottom_cells']:3d}")

            if (iteration + 1) % self.save_interval == 0 or iteration == 0:
                save_path = os.path.join(
                    visualization_dir,
                    f'z_evolution_iter_{iteration+1:04d}.png')
                z = model.get_z()
                visualize_z_single(self.node_x,
                                   self.node_y,
                                   z,
                                   iteration + 1,
                                   save_path,
                                   node_size_x=self.node_size_x,
                                   node_size_y=self.node_size_y)

                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                # Plot Loss
                axes[0, 0].plot(history_iterations,
                                history_loss,
                                'b-',
                                linewidth=2,
                                label='Total Loss')
                axes[0, 0].set_xlabel('Iteration', fontsize=12)
                axes[0, 0].set_ylabel('Loss', fontsize=12)
                axes[0, 0].set_title('Training Loss',
                                     fontsize=14,
                                     fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend(fontsize=10)

                # Plot HPWL
                axes[0, 1].plot(history_iterations,
                                history_hpwl,
                                'g-',
                                linewidth=2,
                                label='HPWL')
                axes[0, 1].set_xlabel('Iteration', fontsize=12)
                axes[0, 1].set_ylabel('HPWL', fontsize=12)
                axes[0, 1].set_title('Half-Perimeter Wire Length',
                                     fontsize=14,
                                     fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend(fontsize=10)

                # Plot Cutsize
                if self.use_cutsize_loss and any(v > 0 for v in history_cut):
                    axes[1, 0].plot(history_iterations,
                                    history_cut,
                                    'r-',
                                    linewidth=2,
                                    label='Cutsize Loss')
                    axes[1, 0].set_xlabel('Iteration', fontsize=12)
                    axes[1, 0].set_ylabel('Cutsize Loss', fontsize=12)
                    axes[1, 0].set_title('Cutsize Loss',
                                         fontsize=14,
                                         fontweight='bold')
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].legend(fontsize=10)
                else:
                    axes[1, 0].text(0.5,
                                    0.5,
                                    'Cutsize Loss\nNot Enabled',
                                    ha='center',
                                    va='center',
                                    fontsize=12,
                                    transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Cutsize Loss',
                                         fontsize=14,
                                         fontweight='bold')

                # Plot Balance
                if self.use_balance_loss and any(v > 0
                                                 for v in history_balance):
                    axes[1, 1].plot(history_iterations,
                                    history_balance,
                                    'm-',
                                    linewidth=2,
                                    label='Balance Loss')
                    axes[1, 1].set_xlabel('Iteration', fontsize=12)
                    axes[1, 1].set_ylabel('Balance Loss', fontsize=12)
                    axes[1, 1].set_title('Balance Loss',
                                         fontsize=14,
                                         fontweight='bold')
                    axes[1, 1].grid(True, alpha=0.3)
                    axes[1, 1].legend(fontsize=10)
                else:
                    axes[1, 1].text(0.5,
                                    0.5,
                                    'Balance Loss\nNot Enabled',
                                    ha='center',
                                    va='center',
                                    fontsize=12,
                                    transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Balance Loss',
                                         fontsize=14,
                                         fontweight='bold')

                plt.tight_layout()
                curve_save_path = os.path.join(visualization_dir,
                                               'training_curves.png')
                plt.savefig(curve_save_path, dpi=150, bbox_inches='tight')
                print(f"   Training curves saved to: {curve_save_path}")
                plt.close()

        print("-" * 60)

        # final results
        print("\n6. Training completed, final results:")
        if use_debug_info:
            final_loss, final_debug_info = model(
                lambda_wl=self.lambda_wl,
                lambda_cut=self.lambda_cut_end
                if self.use_cutsize_loss else 0.0,
                lambda_balance=self.lambda_balance_end
                if self.use_balance_loss else 0.0,
                return_debug_info=True)
            print(f"   - Final total loss: {final_loss.item():.4f}")
            print(f"   - Final HPWL: {final_debug_info['L_WL']:.4f}")
            if self.use_cutsize_loss:
                print(f"   - Final cutsize: {final_debug_info['L_cut']:.4f}")
            if self.use_balance_loss:
                print(
                    f"   - Final balance: {final_debug_info['L_balance']:.4f}")
        else:
            final_loss = model(lambda_balance=self.lambda_balance_end if self.
                               use_balance_loss else 0.0)
            print(f"   - Final total HPWL: {final_loss.item():.4f}")

        stats = model.get_assignment_stats()
        print(
            f"   - z statistics: mean={stats['z_mean']:.4f}, std={stats['z_std']:.4f}, "
            f"min={stats['z_min']:.4f}, max={stats['z_max']:.4f}")
        print(
            f"   - Number of top cells: {stats['top_cells']}, number of bottom cells: {stats['bottom_cells']}"
        )

        # get binary assignment
        binary_z = model.get_binary_assignment()
        print(f"\n7. Binary assignment (threshold=0.5):")
        print(f"   - Number of top cells: {binary_z.sum().item()}")
        print(f"   - Number of bottom cells: {(1 - binary_z).sum().item()}")

        z = model.get_z()
        print(f"\n8. Example soft assignment values (first 10 cells):")
        for i in range(min(10, self.num_nodes)):
            print(
                f"   Cell {i:3d} (x={self.node_x[i].item():.2f}, y={self.node_y[i].item():.2f}): z={z[i].item():.4f} → {'Top' if z[i] > 0.5 else 'Bottom'}"
            )

        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)

        # Save results
        result_path = os.path.join(self.project_root, self.result_dir,
                                   self.config['design']['name'],
                                   'binary_assignment.pt')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        torch.save(binary_z, result_path)
        print(f"   Binary assignment saved to: {result_path}")
        tensor2txt(binary_z, os.path.join(self.project_root, self.result_dir,
                                   self.config['design']['name'],
                                   'binary_assignment.txt'))


if __name__ == "__main__":
    print("Starting 3D Partitioner Flow...")
    differentiable_3d_partitioner_flow = Differentiable3DPartitionerFlow(
        num_nodes=2735,
        num_nets=2644,
        num_pins=8110,
        node_pos=torch.load(
            "benchmarks/tensor/iccad2022/case2_hidden/placement.pt").detach(),
        pin_pos=torch.load(
            "benchmarks/tensor/iccad2022/case2_hidden/pinpos.pt").detach(),
        flat_net2pin_map=torch.load(
            "benchmarks/tensor/iccad2022/case2_hidden/flat_net2pin_map.pt").
        detach(),
        flat_net2pin_start_map=torch.load(
            "benchmarks/tensor/iccad2022/case2_hidden/flat_net2pin_start_map.pt"
        ).detach(),
        pin2node_map=torch.load(
            "benchmarks/tensor/iccad2022/case2_hidden/pin2node_map.pt").detach(
            ),
        node_size_x=torch.load(
            "benchmarks/tensor/iccad2022/case2_hidden/node_size_x.pt").detach(
            ),
        node_size_y=torch.load(
            "benchmarks/tensor/iccad2022/case2_hidden/node_size_y.pt").detach(
            ),
    )
    differentiable_3d_partitioner_flow.run()
