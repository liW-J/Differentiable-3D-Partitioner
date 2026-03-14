'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2026-02-05 16:56:01
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-02-28 18:29:24
FilePath: /Differentiable-3D-Partitioner/partitioner/core/partitioner.py
Description: 3D Partitioner Core Implementation
Implements differentiable partitioning with placement and terminal awareness
for optimized pseudo-3D placement.
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

COORD_EPSILON = 1e-2


class Partitioner(nn.Module):
    """
    Differentiable 3D Partitioner based on LogSumExp soft bounding box
    
    Features:
    1. Mapping trainable pre-activation variable t_i to soft assignment z_i = sigmoid(t_i)
    2. Using LSE soft bounding box to compute HPWL of top and bottom layers
    3. Computing gradient of total HPWL with respect to t, supporting end-to-end training
    4. Support differentiable cutsize loss based on Snake-3D, used to reduce cross-layer connections
    """

    def __init__(self,
                 num_nodes,
                 flat_net2pin_map,
                 flat_net2pin_start_map,
                 pin2node_map,
                 pin_pos_x,
                 pin_pos_y,
                 node_x,
                 node_y,
                 node_pos,
                 node_size_x,
                 node_size_y,
                 alpha=1.0,
                 net_weights=None,
                 gumbel_tau=0.1,
                 dreamplace_basic=None,
                 config=None):
        """
        initialize LSE partitioner
        
        Args:
            num_nodes: number of cells (nodes)
            flat_net2pin_map: flat net to pin mapping, shape [num_pins] tensor
            flat_net2pin_start_map: start index of each net in flat_net2pin_map, shape [num_nets+1] tensor
            pin2node_map: pin to node mapping, shape [num_pins] tensor
            pin_pos_x: x coordinates of pins, shape [num_pins] tensor
            pin_pos_y: y coordinates of pins, shape [num_pins] tensor
            alpha: LSE smoothing parameter, larger means harder to be 0 or 1
            net_weights: optional net weights, shape [num_nets] tensor
            gumbel_tau: Gumbel Softmax temperature parameter, controlling the smoothness of softmax (default 0.1)
                        smaller tau means results closer to discrete distribution; larger tau means smoother distribution
                        if config is provided, this will be overridden by config['partitioner'].get('gumbel_tau', 0.1)
            config: optional configuration dictionary or path to YAML config file
                    if provided, will override alpha and gumbel_tau from config
        """
        super(Partitioner, self).__init__()

        # Load configuration if provided
        if config is None:
            raise ValueError("config is required")

        self.config = config

        self.num_nodes = num_nodes
        self.num_nets = flat_net2pin_start_map.numel() - 1

        # register fixed data structures (no gradient)
        self.register_buffer('flat_net2pin_map',
                             flat_net2pin_map.detach().clone().long())
        self.register_buffer('flat_net2pin_start_map',
                             flat_net2pin_start_map.detach().clone().long())
        self.register_buffer('pin2node_map',
                             pin2node_map.detach().clone().long())
        # Pin position = node position + pin_offset; store offset for x,y,z co-optimization
        node_x_clone = node_x.detach().clone()
        node_y_clone = node_y.detach().clone()
        pin_offset_x = pin_pos_x.detach().clone() - node_x_clone[pin2node_map]
        pin_offset_y = pin_pos_y.detach().clone() - node_y_clone[pin2node_map]
        self.register_buffer('pin_offset_x', pin_offset_x)
        self.register_buffer('pin_offset_y', pin_offset_y)
        # node_x, node_y as optimization variables (no sigmoid), co-optimized with z
        self.node_x = nn.Parameter(node_x_clone)
        self.node_y = nn.Parameter(node_y_clone)
        self.register_buffer('node_size_x', node_size_x.detach().clone())
        self.register_buffer('node_size_y', node_size_y.detach().clone())

        self.dreamplace_basic = dreamplace_basic
        self.node_pos = node_pos

        x_tail_clone = node_pos[num_nodes:node_pos.numel() // 2]
        y_tail_clone = node_pos[node_pos.numel() // 2 + num_nodes:]
        self.register_buffer('x_tail_clone', x_tail_clone)
        self.register_buffer('y_tail_clone', y_tail_clone)
        self.x_tail = nn.Parameter(x_tail_clone)
        self.y_tail = nn.Parameter(y_tail_clone)

        # trainable pre-activation variable t_i (one for each cell)
        # use small random initialization to avoid all z being 0.5 (symmetric point)
        t = torch.randn(num_nodes) * 0.1
        # t = torch.ones(num_nodes) * 10
        self.t = nn.Parameter(t)

        # initial_values
        # LSE smoothing parameter
        self.alpha = alpha

        # Gumbel Softmax temperature parameter
        self.gumbel_tau = config['gumbel_tau']
        self.gumbel_switch_iteration = config['gumbel_switch_iteration']

        # Current iteration counter (used to switch between sigmoid and gumbel_softmax)
        self.current_iteration = 0

        # net weights (if not provided, default to all 1)
        if net_weights is None:
            self.register_buffer('net_weights', torch.ones(self.num_nets))
        else:
            self.register_buffer('net_weights', net_weights.detach().clone())

    def get_z(self):
        """
        map pre-activation variable t to soft assignment z
        Returns:
            z: shape [num_nodes] tensor, representing the probability of each cell being assigned to the top layer
        """
        # switch to gumbel_softmax_z after gumbel_switch_iteration iterations
        if self.current_iteration < self.gumbel_switch_iteration:
            return torch.sigmoid(self.t)
        else:
            return self.gumbel_softmax_z(self.t, tau=self.gumbel_tau)

    def get_pin_pos_x(self):
        """Current pin x = node_x[pin2node] + pin_offset_x (differentiable w.r.t. node_x)."""
        return self.node_x[self.pin2node_map] + self.pin_offset_x

    def get_pin_pos_y(self):
        """Current pin y = node_y[pin2node] + pin_offset_y (differentiable w.r.t. node_y)."""
        return self.node_y[self.pin2node_map] + self.pin_offset_y

    def gumbel_softmax_z(self, pi_logits, tau=0.1):
        """
        use Gumbel Softmax to convert logits to soft assignment probabilities   
        Gumbel Softmax is a differentiable sampling method for discrete variables.

        Args:
            pi_logits: shape [num_nodes] tensor, logit values for each cell
            tau: temperature parameter, controlling the smoothness of the softmax
                 smaller tau means more discrete distribution; larger tau means more smooth distribution
        
        Returns:
            z: shape [num_nodes] tensor, probability of each cell being assigned to the top layer
        """
        # convert single logit t to two logits: [t, 0]
        # the first logit corresponds to top layer, the second logit corresponds to bottom layer
        # use [t, 0] instead of [t, -t] to keep the semantic consistent with the original sigmoid
        # sigmoid(t) = exp(t) / (exp(t) + exp(0)) = exp(t) / (exp(t) + 1)
        logits = torch.stack(
            [pi_logits, torch.zeros_like(pi_logits)], dim=-1)  # [num_nodes, 2]

        # generate Gumbel noise: G = -log(-log(U)), where U ~ Uniform(0,1)
        # use numerically stable implementation
        uniform = torch.rand_like(logits)
        # avoid log(0) and log(1), use clamp
        uniform = torch.clamp(uniform, min=1e-8, max=1.0 - 1e-8)
        gumbel_noise = -torch.log(-torch.log(uniform))

        # add Gumbel noise and divide by temperature parameter
        gumbel_logits = (logits + gumbel_noise) / tau  # [num_nodes, 2]
        softmax_probs = torch.softmax(gumbel_logits, dim=-1)  # [num_nodes, 2]

        # return the first element (probability of top layer)
        z = softmax_probs[..., 0]  # [num_nodes]

        return z

    def lse_max(self, weighted_vals):
        """
        numerically stable LSE maximum calculation
        
        formula: max(x) ≈ (1/α) * log_sum_exp(α * x)
        
        numerically stable implementation: log_sum_exp(a_j) = m + log Σ_j exp(a_j - m)
        where m = max_j a_j
        
        Args:
            weighted_vals: weighted values, shape [..., ...] tensor
            
        Returns:
            soft maximum, shape same as weighted_vals (except the aggregated dimension)
        """
        # use torch.logsumexp to ensure numerical stability
        # logsumexp(α * x) = log(Σ exp(α * x))
        # then divide by α to get soft maximum
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        # calculate log_sum_exp(α * weighted_vals)
        lse = torch.logsumexp(self.alpha * weighted_vals, dim=-1)

        # return (1/α) * log_sum_exp(α * weighted_vals)
        return lse / self.alpha

    def lse_min(self, weighted_vals):
        """
        numerically stable LSE minimum calculation
        formula: min(x) ≈ -(1/α) * log_sum_exp(-α * x)
        
        Args:
            weighted_vals: weighted values, shape [..., ...] tensor
            
        Returns:
            soft minimum, shape same as weighted_vals (except the aggregated dimension)
        """
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        # calculate log_sum_exp(-α * weighted_vals)
        lse = torch.logsumexp(-self.alpha * weighted_vals, dim=-1)

        # return -(1/α) * log_sum_exp(-α * weighted_vals)
        return -lse / self.alpha

    def segment_logsumexp(self, values, segment_ids, num_segments):
        """
        numerically stable segment-based logsumexp calculation
        
        Args:
            values: input values, shape [total_elements] tensor
            segment_ids: segment id for each element, shape [total_elements] tensor
            num_segments: number of segments
            
        Returns:
            logsumexp result for each segment, shape [num_segments] tensor
        """
        # find max value in each segment for numerical stability
        # use scatter_reduce if available (PyTorch 1.12+), otherwise use fallback
        if hasattr(torch.Tensor, 'scatter_reduce_'):
            segment_max = torch.full((num_segments, ),
                                     float('-inf'),
                                     device=values.device,
                                     dtype=values.dtype)
            segment_max = segment_max.scatter_reduce_(0,
                                                      segment_ids,
                                                      values,
                                                      reduce='amax',
                                                      include_self=False)
        else:
            # fallback: compute max for each segment (segment count is usually small)
            segment_max = torch.zeros(num_segments,
                                      device=values.device,
                                      dtype=values.dtype)
            for seg_id in range(num_segments):
                mask = segment_ids == seg_id
                if mask.any():
                    segment_max[seg_id] = values[mask].max()
                else:
                    segment_max[seg_id] = float('-inf')

        # subtract max from values for numerical stability
        values_shifted = values - segment_max[segment_ids]  # [total_elements]

        # compute exp of shifted values
        exp_values = torch.exp(values_shifted)  # [total_elements]

        # sum exp values for each segment
        segment_sum = torch.zeros(num_segments,
                                  device=values.device,
                                  dtype=values.dtype)
        segment_sum = segment_sum.scatter_add_(0, segment_ids, exp_values)

        # compute logsumexp: log(sum(exp)) = max + log(sum(exp(values - max)))
        logsumexp_per_segment = segment_max + torch.log(segment_sum + 1e-10)

        return logsumexp_per_segment

    def compute_cutsize(self, net_idx):
        """
        calculate differentiable cutsize of the specified net (based on Snake-3D formula)
        
        formula:
        - cutsize(n) = (1 - LSE-min(z_i for i∈n)) * LSE-max(z_i for i∈n)
        
        This formula penalizes cross-layer connections: when all nodes are in the same layer (z_i are close to 0 or 1),
        cutsize is close to 0; when nodes are distributed between two layers, cutsize increases.
        
        Args:
            net_idx: index of the net
            
        Returns:
            differentiable cutsize of the specified net, scalar tensor
        """
        # get all pin indices for this net
        start_idx = self.flat_net2pin_start_map[net_idx]
        end_idx = self.flat_net2pin_start_map[net_idx + 1]
        pin_indices = self.flat_net2pin_map[start_idx:end_idx]

        if pin_indices.numel() < 2:
            return torch.tensor(0.0, device=self.node_x.device)

        z = self.get_z()
        node_indices = self.pin2node_map[
            pin_indices]  # shape: [num_pins_in_net]
        z_net = z[node_indices]  # shape: [num_pins_in_net]

        lse_max_z = self.lse_max(z_net)
        lse_min_z = self.lse_min(z_net)

        # cutsize(n) = (1 - LSE-min(z_i)) * LSE-max(z_i)
        cutsize = (1.0 - lse_min_z) * lse_max_z

        return cutsize

    def compute_hpwl_batch(self, net_indices, layer='both'):
        """
        batch calculation of HPWL for multiple nets (vectorized version)
        
        Args:
            net_indices: network indices, tensor of shape [num_nets]
            layer: 'top', 'bottom', or 'both' (default 'both')
        
        Returns:
            if layer == 'both': tuple of (hpwl_top, hpwl_bottom), each shape [num_nets]
            else: hpwl values, tensor of shape [num_nets]
        """
        if net_indices.numel() == 0:
            empty = torch.tensor([], device=self.node_x.device)
            if layer == 'both':
                return empty, empty
            return empty

        num_nets = net_indices.numel()
        z = self.get_z()

        # get start and end indices for all nets
        start_indices = self.flat_net2pin_start_map[net_indices]  # [num_nets]
        end_indices = self.flat_net2pin_start_map[net_indices +
                                                  1]  # [num_nets]

        # calculate pin count for each net
        pin_counts = end_indices - start_indices  # [num_nets]
        valid_mask = pin_counts >= 2  # [num_nets]

        if not valid_mask.any():
            zeros = torch.zeros(num_nets, device=self.node_x.device)
            if layer == 'both':
                return zeros, zeros
            return zeros

        # only process valid nets
        valid_start_indices = start_indices[valid_mask]  # [num_valid_nets]
        valid_end_indices = end_indices[valid_mask]  # [num_valid_nets]
        valid_pin_counts = pin_counts[valid_mask]  # [num_valid_nets]
        valid_net_indices = net_indices[valid_mask]  # [num_valid_nets]
        num_valid_nets = valid_pin_counts.numel()

        # collect all valid net's pin indices
        if num_valid_nets > 0:
            repeated_start_indices = torch.repeat_interleave(
                valid_start_indices, valid_pin_counts)
            cumsum_pin_counts = torch.cumsum(valid_pin_counts,
                                             dim=0)  # [num_valid_nets]
            net_start_positions = torch.cat([
                torch.tensor([0], device=valid_pin_counts.device),
                cumsum_pin_counts[:-1]
            ])  # [num_valid_nets]

            repeated_net_starts = torch.repeat_interleave(
                net_start_positions, valid_pin_counts)
            total_pins = cumsum_pin_counts[-1].item()
            global_indices = torch.arange(total_pins,
                                          device=valid_start_indices.device,
                                          dtype=torch.long)
            pin_offsets_per_net = global_indices - repeated_net_starts

            all_pin_indices = self.flat_net2pin_map[repeated_start_indices +
                                                    pin_offsets_per_net]
        else:
            all_pin_indices = torch.empty(0,
                                          dtype=torch.long,
                                          device=self.node_x.device)

        # get all pin corresponding node indices and z values
        all_node_indices = self.pin2node_map[all_pin_indices]  # [total_pins]
        all_z_net = z[all_node_indices]  # [total_pins]

        # get position of each pin with random selection
        # randomly choose between original value or max_val - value for each pin
        all_x_net = self.get_pin_pos_x()[all_pin_indices]  # [total_pins]
        all_y_net = self.get_pin_pos_y()[all_pin_indices]  # [total_pins]
        all_x_net = all_x_net - all_x_net.min().detach() + COORD_EPSILON
        all_y_net = all_y_net - all_y_net.min().detach() + COORD_EPSILON
        all_x_net_rev = all_x_net.max().detach() - all_x_net + COORD_EPSILON
        all_y_net_rev = all_y_net.max().detach() - all_y_net + COORD_EPSILON

        # compute weighted values for top layer: z_node * x_pin and z_node * y_pin
        weighted_x_top_max = all_z_net * all_x_net  # [total_pins]
        weighted_y_top_max = all_z_net * all_y_net  # [total_pins]
        weighted_x_top_min = all_z_net * all_x_net_rev  # [total_pins]
        weighted_y_top_min = all_z_net * all_y_net_rev  # [total_pins]

        # compute weighted values for bottom layer: (1 - z_node) * x_pin and (1 - z_node) * y_pin
        z_bottom = 1.0 - all_z_net  # [total_pins]
        weighted_x_bottom_max = z_bottom * all_x_net_rev  # [total_pins]
        weighted_y_bottom_max = z_bottom * all_y_net_rev  # [total_pins]
        weighted_x_bottom_min = z_bottom * all_x_net  # [total_pins]
        weighted_y_bottom_min = z_bottom * all_y_net  # [total_pins]

        # vectorized calculation of HPWL for each net
        hpwl_top_per_net = torch.zeros(num_valid_nets,
                                       device=self.node_x.device)
        hpwl_bottom_per_net = torch.zeros(num_valid_nets,
                                          device=self.node_x.device)

        # calculate HPWL for each net using vectorized approach

        # precompute pin offsets for all nets
        pin_offsets = torch.cumsum(torch.cat([
            torch.tensor([0], device=valid_pin_counts.device),
            valid_pin_counts[:-1]
        ]),
                                   dim=0)  # [num_valid_nets]

        # vectorized HPWL calculation by grouping nets with same pin count
        # Strategy: group nets by pin count, then process each group in parallel
        unique_pin_counts, inverse_indices, counts = torch.unique(
            valid_pin_counts, return_inverse=True, return_counts=True)

        # process each group of nets with same pin count
        for group_idx, pin_count in enumerate(unique_pin_counts):
            # get indices of nets in this group
            group_mask = inverse_indices == group_idx
            group_net_indices = torch.where(group_mask)[
                0]  # [num_nets_in_group]
            num_nets_in_group = group_net_indices.numel()

            if num_nets_in_group == 0:
                continue

            pin_count_int = pin_count.item()

            # get pin offsets for this group
            group_pin_offsets = pin_offsets[
                group_net_indices]  # [num_nets_in_group]

            # create index tensor for batch slicing: [num_nets_in_group, pin_count]
            # each row corresponds to one net's pins
            batch_indices = (group_pin_offsets.unsqueeze(1) + torch.arange(
                pin_count_int, device=group_pin_offsets.device).unsqueeze(0))

            # batch extract data for all nets in this group
            # shape: [num_nets_in_group, pin_count]
            if layer in ['top', 'both']:
                wx_top_max_batch = weighted_x_top_max[batch_indices]
                wy_top_max_batch = weighted_y_top_max[batch_indices]
                wx_top_min_batch = weighted_x_top_min[batch_indices]
                wy_top_min_batch = weighted_y_top_min[batch_indices]

                # batch LSE computation: apply lse_max along pin dimension
                # lse_max uses logsumexp which supports batch dimension
                x_top_max_batch = self.lse_max(wx_top_max_batch)
                y_top_max_batch = self.lse_max(wy_top_max_batch)
                x_top_min_batch = self.lse_max(wx_top_min_batch)
                y_top_min_batch = self.lse_max(wy_top_min_batch)

                # batch max computation
                wx_top_max_max = wx_top_max_batch.max(dim=1)[0]
                wx_top_min_max = wx_top_min_batch.max(dim=1)[0]
                wy_top_max_max = wy_top_max_batch.max(dim=1)[0]
                wy_top_min_max = wy_top_min_batch.max(dim=1)[0]

                # # vectorized HPWL calculation for this group
                hpwl_top_group = (
                    x_top_max_batch + y_top_max_batch + x_top_min_batch +
                    y_top_min_batch -
                    torch.maximum(wx_top_max_max, wx_top_min_max).detach() -
                    torch.maximum(wy_top_max_max, wy_top_min_max).detach())

                # assign results back
                hpwl_top_per_net[group_net_indices] = hpwl_top_group

            if layer in ['bottom', 'both']:
                wx_bottom_max_batch = weighted_x_bottom_max[batch_indices]
                wy_bottom_max_batch = weighted_y_bottom_max[batch_indices]
                wx_bottom_min_batch = weighted_x_bottom_min[batch_indices]
                wy_bottom_min_batch = weighted_y_bottom_min[batch_indices]

                # batch LSE computation
                x_bottom_max_batch = self.lse_max(wx_bottom_max_batch)
                y_bottom_max_batch = self.lse_max(wy_bottom_max_batch)
                x_bottom_min_batch = self.lse_max(wx_bottom_min_batch)
                y_bottom_min_batch = self.lse_max(wy_bottom_min_batch)

                # batch max computation
                wx_bottom_max_max = wx_bottom_max_batch.max(dim=1)[0]
                wx_bottom_min_max = wx_bottom_min_batch.max(dim=1)[0]
                wy_bottom_max_max = wy_bottom_max_batch.max(dim=1)[0]
                wy_bottom_min_max = wy_bottom_min_batch.max(dim=1)[0]

                # vectorized HPWL calculation for this group
                hpwl_bottom_group = (
                    x_bottom_max_batch + y_bottom_max_batch +
                    x_bottom_min_batch + y_bottom_min_batch - torch.maximum(
                        wx_bottom_max_max, wx_bottom_min_max).detach() -
                    torch.maximum(wy_bottom_max_max,
                                  wy_bottom_min_max).detach())

                # assign results back
                hpwl_bottom_per_net[group_net_indices] = hpwl_bottom_group

        # create complete result arrays (including invalid nets)
        if layer == 'both':
            result_top = torch.zeros(num_nets, device=self.node_x.device)
            result_bottom = torch.zeros(num_nets, device=self.node_x.device)
            result_top[valid_mask] = hpwl_top_per_net
            result_bottom[valid_mask] = hpwl_bottom_per_net
            return result_top, result_bottom
        elif layer == 'top':
            result = torch.zeros(num_nets, device=self.node_x.device)
            result[valid_mask] = hpwl_top_per_net
            return result
        else:  # layer == 'bottom'
            result = torch.zeros(num_nets, device=self.node_x.device)
            result[valid_mask] = hpwl_bottom_per_net
            return result

    def compute_cutsize_batch(self, net_indices):
        """
        batch calculation of differentiable cutsize for multiple nets (vectorized version)
        
        Args:
            net_indices: network indices, tensor of shape [num_nets]
        
        Returns:
            cutsize values, tensor of shape [num_nets]
        """
        if net_indices.numel() == 0:
            return torch.tensor([], device=self.node_x.device)

        num_nets = net_indices.numel()
        z = self.get_z()

        # get start and end indices for all nets
        start_indices = self.flat_net2pin_start_map[net_indices]  # [num_nets]
        end_indices = self.flat_net2pin_start_map[net_indices +
                                                  1]  # [num_nets]

        # calculate pin count for each net
        pin_counts = end_indices - start_indices  # [num_nets]

        # filter out nets with less than 2 pins (these nets have cutsize=0)
        valid_mask = pin_counts >= 2  # [num_nets]

        if not valid_mask.any():
            return torch.zeros(num_nets, device=self.node_x.device)

        # only process valid nets
        valid_start_indices = start_indices[valid_mask]  # [num_valid_nets]
        valid_end_indices = end_indices[valid_mask]  # [num_valid_nets]
        valid_pin_counts = pin_counts[valid_mask]  # [num_valid_nets]
        num_valid_nets = valid_pin_counts.numel()

        # vectorized collection of all valid net's pin indices
        total_pins = valid_pin_counts.sum().item()
        if total_pins == 0:
            return torch.zeros(num_nets, device=self.node_x.device)

        # create segment indices for grouping pins by net (used later for grouped operations)
        segment_ids = torch.repeat_interleave(
            torch.arange(num_valid_nets, device=self.node_x.device),
            valid_pin_counts)  # [total_pins]

        # optimized extraction of pin indices using list comprehension (more memory efficient than broadcasting)
        # this avoids creating a large max_pins x num_valid_nets matrix
        all_pin_indices = torch.cat([
            self.flat_net2pin_map[valid_start_indices[i]:valid_end_indices[i]]
            for i in range(num_valid_nets)
        ])  # [total_pins]

        all_node_indices = self.pin2node_map[all_pin_indices]  # [total_pins]
        all_z_net = z[all_node_indices]  # [total_pins]
        # using segment-based logsumexp operations
        alpha_z = self.alpha * all_z_net  # [total_pins]
        lse_max_per_net = self.segment_logsumexp(alpha_z, segment_ids,
                                                 num_valid_nets) / self.alpha
        neg_alpha_z = -self.alpha * all_z_net  # [total_pins]
        lse_min_per_net = -self.segment_logsumexp(neg_alpha_z, segment_ids,
                                                  num_valid_nets) / self.alpha

        # calculate cutsize: (1 - lse_min) * lse_max
        valid_cutsizes = (
            1.0 - lse_min_per_net) * lse_max_per_net  # [num_valid_nets]

        # create complete result array (including invalid nets)
        result = torch.zeros(num_nets, device=self.node_x.device)
        result[valid_mask] = valid_cutsizes

        return result

    def compute_terminal_positions(self, net_indices):
        """
        compute terminal positions at the center of optimal region for cut nets (vectorized version)
        
        Args:
            net_indices: network indices, tensor of shape [num_nets]
        
        Returns:
            terminal_positions: tensor of shape [num_nets, 2] (x, y coordinates)
                               terminal positions for non-cut nets are NaN
            cut_mask: boolean tensor of shape [num_nets], True indicates this net generates a terminal
        """
        if net_indices.numel() == 0:
            return torch.empty((0, 2), device=self.node_x.device), torch.empty(
                0, dtype=torch.bool, device=self.node_x.device)

        num_nets = net_indices.numel()
        terminal_positions = torch.full((num_nets, 2),
                                        float('nan'),
                                        device=self.node_x.device)

        # get the probability of each cell being assigned to top layer
        z = self.get_z()  # [num_nodes]

        # get start and end indices for all nets
        start_indices = self.flat_net2pin_start_map[net_indices]  # [num_nets]
        end_indices = self.flat_net2pin_start_map[net_indices +
                                                  1]  # [num_nets]

        # calculate pin count for each net
        pin_counts = end_indices - start_indices  # [num_nets]
        valid_mask = pin_counts >= 2  # [num_nets]

        if not valid_mask.any():
            cut_mask = torch.zeros(num_nets,
                                   dtype=torch.bool,
                                   device=self.node_x.device)
            return terminal_positions, cut_mask

        # only process valid nets
        valid_start_indices = start_indices[valid_mask]  # [num_valid_nets]
        valid_end_indices = end_indices[valid_mask]  # [num_valid_nets]
        valid_pin_counts = pin_counts[valid_mask]  # [num_valid_nets]
        valid_net_indices = net_indices[valid_mask]  # [num_valid_nets]
        num_valid_nets = valid_pin_counts.numel()

        all_pin_indices = torch.cat([
            self.flat_net2pin_map[valid_start_indices[i]:valid_end_indices[i]]
            for i in range(num_valid_nets)
        ])  # [total_pins]

        # get all pin corresponding node indices and z values
        all_node_indices = self.pin2node_map[all_pin_indices]  # [total_pins]
        all_z_net = z[all_node_indices]  # [total_pins]

        # vectorized cut detection: determine whether each net is cut
        # a net is cut if its pins are not all in top (z > 0.5) or all in bottom (z < 0.5)
        cut_mask = torch.zeros(num_nets,
                               dtype=torch.bool,
                               device=self.node_x.device)

        # precompute pin offsets for all valid nets
        pin_offsets = torch.cumsum(torch.cat([
            torch.tensor([0], device=valid_pin_counts.device),
            valid_pin_counts[:-1]
        ]),
                                   dim=0)  # [num_valid_nets]

        # vectorized cut detection by grouping nets with same pin count
        unique_pin_counts, inverse_indices, counts = torch.unique(
            valid_pin_counts, return_inverse=True, return_counts=True)

        # process each group of nets with same pin count
        for group_idx, pin_count in enumerate(unique_pin_counts):
            # find all nets in this group
            group_mask = inverse_indices == group_idx
            group_net_indices_in_valid = group_mask.nonzero(
                as_tuple=True)[0]  # indices in valid_nets
            num_group_nets = group_net_indices_in_valid.numel()

            if num_group_nets == 0:
                continue

            # get pin offsets for this group
            group_pin_offsets = pin_offsets[
                group_net_indices_in_valid]  # [num_group_nets]

            # create batch indices for all pins in this group
            # shape: [num_group_nets, pin_count]
            batch_indices = group_pin_offsets.unsqueeze(1) + torch.arange(
                pin_count, device=group_pin_offsets.device).unsqueeze(0)

            # get z values for all pins in this group
            z_net_batch = all_z_net[
                batch_indices]  # [num_group_nets, pin_count]

            # determine if all in top (z > 0.5) or all in bottom (z < 0.5) for each net
            all_in_top = (z_net_batch > 0.5).all(dim=1)  # [num_group_nets]
            all_in_bottom = (z_net_batch < 0.5).all(dim=1)  # [num_group_nets]

            # if not all in top and not all in bottom, then it is cut
            group_cut_mask = ~(all_in_top | all_in_bottom)  # [num_group_nets]

            # map back to original net indices
            valid_net_positions = group_net_indices_in_valid[group_cut_mask]
            if valid_net_positions.numel() > 0:
                # find corresponding positions in original net_indices
                original_positions = valid_mask.nonzero(
                    as_tuple=True)[0][valid_net_positions]
                cut_mask[original_positions] = True

        if not cut_mask.any():
            return terminal_positions, cut_mask

        # for cut nets, compute their terminal positions (center of optimal region)
        # optimal region is defined as the center of the bounding box of all pin positions for this net
        cut_valid_mask = valid_mask & cut_mask  # [num_nets]
        cut_valid_positions = cut_valid_mask.nonzero(
            as_tuple=True)[0]  # positions in net_indices

        if cut_valid_positions.numel() == 0:
            return terminal_positions, cut_mask

        # get cut nets' start and end indices
        cut_start_indices = start_indices[cut_valid_mask]  # [num_cut_nets]
        cut_end_indices = end_indices[cut_valid_mask]  # [num_cut_nets]
        cut_pin_counts = pin_counts[cut_valid_mask]  # [num_cut_nets]
        num_cut_nets = cut_pin_counts.numel()

        # use list comprehension for better performance than explicit loop
        all_cut_pin_indices = torch.cat([
            self.flat_net2pin_map[cut_start_indices[i]:cut_end_indices[i]]
            for i in range(num_cut_nets)
        ])  # [total_cut_pins]

        # get all cut pin positions
        all_cut_pin_x = self.get_pin_pos_x()[
            all_cut_pin_indices]  # [total_cut_pins]
        all_cut_pin_y = self.get_pin_pos_y()[
            all_cut_pin_indices]  # [total_cut_pins]

        # vectorized terminal position calculation by grouping nets with same pin count
        cut_pin_offsets = torch.cumsum(torch.cat([
            torch.tensor([0], device=cut_pin_counts.device),
            cut_pin_counts[:-1]
        ]),
                                       dim=0)  # [num_cut_nets]

        # group cut nets by pin count
        cut_unique_pin_counts, cut_inverse_indices, cut_counts = torch.unique(
            cut_pin_counts, return_inverse=True, return_counts=True)

        # process each group of cut nets with same pin count
        for group_idx, pin_count in enumerate(cut_unique_pin_counts):
            # find all cut nets in this group
            group_mask = cut_inverse_indices == group_idx
            group_net_indices_in_cut = group_mask.nonzero(
                as_tuple=True)[0]  # indices in cut_nets
            num_group_nets = group_net_indices_in_cut.numel()

            if num_group_nets == 0:
                continue

            # get pin offsets for this group
            group_pin_offsets = cut_pin_offsets[
                group_net_indices_in_cut]  # [num_group_nets]

            # create batch indices for all pins in this group
            batch_indices = group_pin_offsets.unsqueeze(1) + torch.arange(
                pin_count, device=group_pin_offsets.device).unsqueeze(0)

            # get pin positions for all pins in this group
            pin_x_batch = all_cut_pin_x[
                batch_indices]  # [num_group_nets, pin_count]
            pin_y_batch = all_cut_pin_y[
                batch_indices]  # [num_group_nets, pin_count]

            # compute center of bounding box for each net (vectorized)
            center_x_batch = (pin_x_batch.max(dim=1)[0] + pin_x_batch.min(
                dim=1)[0]) / 2.0  # [num_group_nets]
            center_y_batch = (pin_y_batch.max(dim=1)[0] + pin_y_batch.min(
                dim=1)[0]) / 2.0  # [num_group_nets]

            # map back to original net positions
            original_positions = cut_valid_positions[group_net_indices_in_cut]
            terminal_positions[original_positions, 0] = center_x_batch
            terminal_positions[original_positions, 1] = center_y_batch

        return terminal_positions, cut_mask

    def detect_terminal_overlaps(self,
                                 terminal_positions,
                                 cut_mask,
                                 overlap_threshold=1500):
        """
        detect whether terminals overlap
        
        Args:
            terminal_positions: tensor of shape [num_nets, 2] (x, y coordinates)
            cut_mask: boolean tensor of shape [num_nets], True indicates this net generates a terminal
            overlap_threshold: overlap threshold, terminals with distance less than this value are considered overlapping
        
        Returns:
            overlap_groups: list of lists, each sublist contains overlapping net indices
            overlap_mask: boolean tensor of shape [num_nets], True indicates this net's terminal overlaps with other nets' terminals
        """
        num_nets = terminal_positions.shape[0]
        overlap_mask = torch.zeros(num_nets,
                                   dtype=torch.bool,
                                   device=terminal_positions.device)
        overlap_groups = []

        # only consider nets that generate terminals
        valid_indices = torch.where(cut_mask)[0]  # [num_valid_nets]

        if valid_indices.numel() < 2:
            return overlap_groups, overlap_mask

        # get valid terminal positions
        valid_positions = terminal_positions[
            valid_indices]  # [num_valid_nets, 2]

        # compute distances between all terminals
        # use Euclidean distance
        positions_expanded_1 = valid_positions.unsqueeze(
            1)  # [num_valid_nets, 1, 2]
        positions_expanded_2 = valid_positions.unsqueeze(
            0)  # [1, num_valid_nets, 2]
        distances = torch.norm(positions_expanded_1 - positions_expanded_2,
                               dim=2)  # [num_valid_nets, num_valid_nets]

        # find overlapping terminals (distance less than threshold, and not itself)
        num_valid_nets = valid_indices.numel()
        overlap_matrix = (distances < overlap_threshold) & (
            distances > 0)  # [num_valid_nets, num_valid_nets]

        # use union-find or simple method to find overlap groups
        visited = torch.zeros(num_valid_nets,
                              dtype=torch.bool,
                              device=terminal_positions.device)

        for i in range(num_valid_nets):
            if visited[i]:
                continue

            # find all terminals overlapping with i
            overlaps_with_i = overlap_matrix[i] | overlap_matrix[:, i]
            if overlaps_with_i.any():
                # create an overlap group
                group_indices = torch.where(overlaps_with_i)[0]
                group_original_indices = valid_indices[group_indices].tolist()
                overlap_groups.append(group_original_indices)

                # mark as visited
                visited[group_indices] = True

                # update overlap_mask
                overlap_mask[valid_indices[group_indices]] = True

        return overlap_groups, overlap_mask

    def compute_cutsize_loss(self,
                             selected_nets=None,
                             cutsize_net_weights=None,
                             handle_terminal_overlap=True,
                             overlap_threshold=500,
                             overlap_weight_penalty=2.0):
        """
        calculate total cutsize loss (only for selected nets)
        
        L_cut = Σ_{n∈selected_nets} w(n) * cutsize(n)
        
        if handle_terminal_overlap=True, it will detect terminal overlaps and increase the weight of one of the overlapping terminals to eliminate cutsize
        
        Args:
            selected_nets: network indices to apply cutsize constraint, tensor or list
                           if None, calculate for all nets
            cutsize_net_weights: weights for each selected net, tensor or list
                                 if None, use self.net_weights corresponding to the selected nets
                                 if selected_nets is None, use self.net_weights
            handle_terminal_overlap: whether to handle terminal overlap, default True
            overlap_threshold: distance threshold for terminal overlap
            overlap_weight_penalty: multiplier to increase net weight when terminals overlap
        
        Returns:
            total_cutsize: total cutsize loss, scalar tensor
        """
        if selected_nets is None:
            # if not specified, calculate for all nets (may be slow)
            selected_nets = list(range(self.num_nets))

        if isinstance(selected_nets, list):
            selected_nets = torch.tensor(selected_nets,
                                         dtype=torch.long,
                                         device=self.node_x.device)

        # ensure selected_nets is in valid range
        if selected_nets.max() >= self.num_nets or selected_nets.min() < 0:
            raise ValueError(
                f"selected_nets index out of range [0, {self.num_nets-1}]")

        # get weights
        if cutsize_net_weights is None:
            # use weights from self.net_weights corresponding to the selected nets
            weights = self.net_weights[selected_nets].clone()
        else:
            # use user-provided weights
            if isinstance(cutsize_net_weights, list):
                cutsize_net_weights = torch.tensor(cutsize_net_weights,
                                                   dtype=torch.float32,
                                                   device=self.node_x.device)
            if cutsize_net_weights.numel() != selected_nets.numel():
                raise ValueError(
                    f"cutsize_net_weights length ({cutsize_net_weights.numel()}) "
                    f"must match selected_nets length ({selected_nets.numel()})"
                )
            weights = cutsize_net_weights.clone()

        weights.fill_(0.0)

        # handle terminal overlap
        if handle_terminal_overlap:
            # compute terminal positions
            terminal_positions, cut_mask = self.compute_terminal_positions(
                selected_nets)
            overlap_groups, overlap_mask = self.detect_terminal_overlaps(
                terminal_positions,
                cut_mask,
                overlap_threshold=overlap_threshold)

            if overlap_groups:
                # concatenate all groups into a single tensor
                all_overlapping_indices = torch.cat([
                    torch.tensor(group,
                                 dtype=torch.long,
                                 device=selected_nets.device)
                    for group in overlap_groups if len(group) > 0
                ])
                # remove duplicates using torch.unique
                overlapping_net_indices = torch.unique(all_overlapping_indices)

                # expand selected_nets to [num_selected_nets, 1] and overlapping_net_indices to [1, num_overlapping_nets]
                # then compare to find matches
                selected_nets_expanded = selected_nets.unsqueeze(
                    1)  # [num_selected_nets, 1]
                overlapping_expanded = overlapping_net_indices.unsqueeze(
                    0)  # [1, num_overlapping_nets]

                # find matches: [num_selected_nets, num_overlapping_nets]
                matches = (selected_nets_expanded == overlapping_expanded)
                matching_positions = matches.any(dim=1)  # [num_selected_nets]
                weights[matching_positions] += overlap_weight_penalty

        # vectorized calculation of cutsize for all selected nets
        cutsizes = self.compute_cutsize_batch(
            selected_nets)  # [num_selected_nets]

        total_cutsize = (weights * cutsizes).sum()

        return total_cutsize

    def compute_balance_loss(self):
        """
        calculate balance loss
        if the density of a bin exceeds half of the bin area, add relu penalty
        
        Returns:
            balance_loss: balance loss, scalar tensor
        """

        z = self.get_z()
        top_z = z
        bottom_z = 1 - z

        top_threshold_factor = self.config['balance_loss']['top_threshold_factor']
        bottom_threshold_factor = self.config['balance_loss']['bottom_threshold_factor']
        num_bins_x = self.config['balance_loss']['num_bins_x']
        num_bins_y = self.config['balance_loss']['num_bins_y']

        def compute_density_map(partition_z, num_bin_x, num_bin_y):
            node_x = self.node_x.detach()
            node_y = self.node_y.detach()
            x_range = node_x.max() - node_x.min()
            y_range = node_y.max() - node_y.min()
            bin_size_x = x_range / num_bin_x
            bin_size_y = y_range / num_bin_y

            # calculate the area of each bin
            node_area_map = torch.zeros(num_bin_x, num_bin_y, device=z.device)
            density_map = torch.zeros(num_bin_x, num_bin_y, device=z.device)
            node_x_min = node_x.min()
            node_y_min = node_y.min()

            # compute bin indices for all nodes at once (vectorized)
            x_idx = ((node_x - node_x_min) / bin_size_x).long()
            y_idx = ((node_y - node_y_min) / bin_size_y).long()
            x_idx = torch.clamp(x_idx, 0, num_bin_x - 1)
            y_idx = torch.clamp(y_idx, 0, num_bin_y - 1)

            # compute node areas and weighted areas (vectorized)
            node_areas = self.node_size_x * self.node_size_y  # [num_nodes]
            weighted_areas = node_areas * partition_z  # [num_nodes]

            # use index_add_ to accumulate values (vectorized)
            # index_add_ requires 1D indices, so we flatten the 2D indices
            # Convert 2D indices (x_idx, y_idx) to 1D linear indices
            linear_indices = x_idx * num_bin_y + y_idx  # [num_nodes]

            # flatten density maps for index_add_
            density_map_flat = density_map.flatten()  # [num_bin_x * num_bin_y]
            node_area_map_flat = node_area_map.flatten(
            )  # [num_bin_x * num_bin_y]
            density_map_flat.index_add_(0, linear_indices, weighted_areas)
            node_area_map_flat.index_add_(0, linear_indices, node_areas)
            density_map = density_map_flat.view(num_bin_x, num_bin_y)
            node_area_map = node_area_map_flat.view(num_bin_x, num_bin_y)

            return density_map, node_area_map

        top_density_map, node_area_map = compute_density_map(
            top_z, num_bins_x, num_bins_y)
        bottom_density_map, _ = compute_density_map(bottom_z, num_bins_x,
                                                    num_bins_y)

        balance_loss = torch.relu(top_density_map - node_area_map*top_threshold_factor).sum() + \
                       torch.relu(bottom_density_map - node_area_map*bottom_threshold_factor).sum()
        # balance_loss = torch.relu(top_density_map - node_area_map*0.329).sum() + \
        #                torch.relu(bottom_density_map - node_area_map*0.671).sum()

        return balance_loss

    def compute_density_loss(self):
        """
        Calculate density loss using DREAMPlace density op.
        Sum of density overflow on top and bottom layer (same bin grid as placement).

        Returns:
            density_loss: density loss, scalar tensor
        """

        def get_density(pos):
            return self.dreamplace_basic.op_collections.density_op(pos)

        density_pos = torch.cat(
            [self.node_x, self.x_tail, self.node_y, self.y_tail], dim=0)
        density_loss = get_density(density_pos)
        return density_loss

    def forward(self,
                lambda_wl=1.0,
                lambda_cut=0.0,
                lambda_balance=0.0,
                lambda_density=0.0,
                selected_nets=None,
                cutsize_net_weights=None,
                return_debug_info=False):
        """
        calculate total loss (HPWL + Cutsize + Balance + Density)
        L_WL = Σ_e (HPWL_top_e + HPWL_bottom_e) * weight_e
        L_cut = Σ_{n∈selected_nets} w(n) * cutsize(n)
        L_balance = balance loss (penalizes density exceeding half bin area)
        L_density = DREAMPlace density overflow loss (top + bottom layer)
        L_total = λ_WL * L_WL + λ_cut * L_cut + λ_balance * L_balance + λ_density * L_density

        Args:
            lambda_wl: weight of HPWL loss, default 1.0
            lambda_cut: weight of cutsize loss, default 0.0
            lambda_balance: weight of balance loss, default 0.0
            lambda_density: weight of density loss, default 0.0
            selected_nets: indices of nets to apply cutsize constraint, tensor or list
                           only used when lambda_cut > 0
            cutsize_net_weights: weights of each selected net, tensor or list
                                 if None, use self.net_weights corresponding to the selected nets
            return_debug_info: whether to return debug information, default False

        Returns:
            total_loss: total loss, scalar tensor
            if return_debug_info:
                debug_info: dict containing:
                    - 'L_WL': total HPWL loss
                    - 'L_cut': total cutsize loss
                    - 'L_balance': total balance loss
                    - 'L_density': total density loss
                    - 'L_total': total loss
                return (total_loss, debug_info) tuple
            else:
                return total_loss
        """
        # batch calculation of HPWL for all nets (vectorized, much faster)
        all_net_indices = torch.arange(self.num_nets,
                                       device=self.node_x.device)
        hpwl_top_all, hpwl_bottom_all = self.compute_hpwl_batch(
            all_net_indices, layer='both')

        # weighted accumulate: Σ_e (HPWL_top_e + HPWL_bottom_e) * weight_e
        total_hpwl = (self.net_weights *
                      (hpwl_top_all + hpwl_bottom_all)).sum()

        cutsize_loss = torch.tensor(0.0, device=self.node_x.device)
        if lambda_cut > 0:
            cutsize_loss = self.compute_cutsize_loss(
                selected_nets=selected_nets,
                cutsize_net_weights=cutsize_net_weights)

        balance_loss = torch.tensor(0.0, device=self.node_x.device)
        if lambda_balance > 0:
            balance_loss = self.compute_balance_loss()

        density_loss = torch.tensor(0.0, device=self.node_x.device)
        if lambda_density > 0:
            density_loss = self.compute_density_loss()

        # total_loss = (lambda_wl * total_hpwl + lambda_cut * cutsize_loss +
        #               lambda_balance * balance_loss +
        #               lambda_density * density_loss)

        total_loss = (lambda_cut * cutsize_loss + lambda_density * density_loss)

        # if not return debug information, return total loss
        if not return_debug_info:
            return total_loss

        # return total loss and debug information
        debug_info = {
            'L_WL': total_hpwl.item(),
            'L_cut': cutsize_loss.item() if lambda_cut > 0 else 0.0,
            'L_balance': balance_loss.item() if lambda_balance > 0 else 0.0,
            'L_density': density_loss.item() if lambda_density > 0 else 0.0,
            'L_total': total_loss.item()
        }
        return total_loss, debug_info

    def get_binary_assignment(self, threshold=0.5):
        """
        get binary assignment
        threshold for binary assignment, default 0.5
            
        Returns:
            binary_z: binary assignment, 1 for top, 0 for bottom
        """
        z = self.get_z()
        return (z > threshold).to(torch.int32)

    def get_assignment_stats(self):
        """
        get assignment statistics for debugging
        
        Returns:
            dict: contains z statistics
        """
        z = self.get_z()
        return {
            'z_mean': z.mean().item(),
            'z_std': z.std().item(),
            'z_min': z.min().item(),
            'z_max': z.max().item(),
            'top_cells': (z > 0.5).sum().item(),
            'bottom_cells': (z <= 0.5).sum().item()
        }
