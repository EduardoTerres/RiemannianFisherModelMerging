from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional

import torch
from safetensors.torch import load_file
from torch import Tensor

from src.geometry import Manifold, SOnManifold


MergeMode = Literal[
    "standard",
    "standard_rescaled",
    "diagonal_fisher",
    "diagonal_fisher_avg",
    "diagonal_fisher_rescaled",
    "diagonal_fisher_max_rescaled",
    "diagonal_fisher_std_rescaled",
    "diagonal_fisher_kl_rescaled",
    "fisher",
]

class RiemannianMerging(ABC):
    """
    Abstract Riemannian merging interface.

    Subclasses implement three concerns separately:
      1. load_adapters  - I/O: open files and return raw weight dicts.
      2. to_task_vectors - geometry: convert weights to Lie-algebra task vectors.
      3. merge_formula  - math: combine task vectors (with optional curvature).

    The public entry-point `merge` wires them together.
    """

    def __init__(
        self,
        manifold: Manifold,
        lam: float = 1.0,
        alphas: Optional[List[float]] = None,
    ):
        self.manifold = manifold
        self.lam = lam
        self.alphas = alphas  # per-task weights; ones if None

    def _resolve_alphas(self, T: int) -> Tensor:
        if self.alphas is not None:
            return torch.tensor(self.alphas, dtype=torch.float32)
        return torch.ones(T, dtype=torch.float32)

    @abstractmethod
    def load_weights(self, adapter_paths: List[str]) -> List[Dict[str, Tensor]]:
        """Load adapter weights from disk. Returns one state-dict per path."""

    @abstractmethod
    def load_fishers(self, paths: List[str]) -> List[Dict[str, Tensor]]:
        """Load diagonal Fisher dicts from disk. Returns one dict per path."""

    @abstractmethod
    def merge_formula(
        self,
        weights_list: List[Tensor],
        fisher_list: Optional[List[Dict[str, Tensor]]] = None,
        mode: MergeMode = "standard",
        alphas: Optional[Tensor] = None,
    ) -> Tensor:
        """Merge task vectors into one (B, n, n) Omega.

        Modes: "standard" or "diagonal_fisher"
        Fisher data is required for Fisher-based modes.
        """

    @abstractmethod
    def merge(
        self,
        adapter_paths: List[str],
        fisher_paths: Optional[List[str]] = None,
        mode: MergeMode = "standard",
    ) -> Dict[str, Tensor]:
        """
        Full merging pipeline
        """


class OFTMerging(RiemannianMerging):
    """
    Riemannian merging for OFT adapters on SO(n).

    Three merge modes
    -----------------
    standard           Weighted average of Lie-algebra task vectors. No Fisher.
    diagonal_fisher Element-wise Fisher weighting in the vectorised so(n) basis.
                    Each component j of the merged vector is
                        Omega*_j = (sum_t alpha_t f_(t,j) Omega_(t,j)) / (lambda + sum_t alpha_t f_(t,j))
    """

    def __init__(
        self,
        lam: float = 0.0,
        alphas: Optional[List[float]] = None,
        device: str = "cpu",
    ):
        super().__init__(manifold=SOnManifold(), lam=lam, alphas=alphas)
        self.manifold: SOnManifold
        self.device = device

    def oft_params_to_skew_matrix(
            self, oft_params: torch.Tensor, son_dimension: int,
        ) -> torch.Tensor:
        """
        Convert OFT parameters (num_blocks, num_params_per_block)
        to skew-symmetric matrices (num_blocks, block_size, block_size).

        Taken from OrthoMerge_OFT_models.py.
        """
        num_blocks, num_params = oft_params.shape
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)
        if num_params != son_dimension:
            raise ValueError(f"num_params_per_block={num_params}  block_size={block_size} ")

        indices = torch.triu_indices(block_size, block_size, offset=1, device=oft_params.device)
        rows, cols = indices[0], indices[1]
        S = torch.zeros(
            num_blocks, block_size, block_size, dtype=oft_params.dtype, device=oft_params.device
        )

        S[:, rows, cols] = oft_params
        S = S - S.transpose(-2, -1)

        return S


    def skew_matrix_to_oft_params(self, S: torch.Tensor) -> torch.Tensor:
        """
        Convert skew-symmetric matrices (num_blocks, block_size, block_size)
        to OFT parameters (num_blocks, num_params_per_block).

        Taken from OrthoMerge_OFT_models.py.
        """

        num_blocks, block_size, _ = S.shape

        indices = torch.triu_indices(block_size, block_size, offset=1, device=S.device)
        rows, cols = indices[0], indices[1]
        oft_params = S[:, rows, cols]  # (num_blocks, num_params_per_block)

        return oft_params

    def load_weights(self, adapter_paths: List[str]) -> List[Dict[str, Tensor]]:
        """
        Load OFT adapters

        Code taken from OrthoMerge_OFT_models.py
        """
        all_weights = []
        for adapter_path in adapter_paths:
            model_path = os.path.join(adapter_path, "adapter_model.safetensors")
            if not os.path.exists(model_path):
                model_path = os.path.join(adapter_path, "adapter_model.bin")
                if os.path.exists(model_path):
                    weights = torch.load(model_path, map_location=self.device)
                else:
                    print(f"[WARNING]: No adapter weights found at {adapter_path}, skipping...")
                    continue
            else:
                weights = load_file(model_path, device=self.device)

            all_weights.append(weights)
            print(f"Loaded adapters: {adapter_path}")

        return all_weights

    def load_fishers(
        self,
        paths: List[str],
        remove_default_ettiquete: bool = True,
    ) -> List[Dict[str, Tensor]]:
        all_fishers = []
        for path in paths:
            if not os.path.exists(path):
                print(f"[WARNING]: No Fisher file found at {path}, skipping...")
                continue
            fisher = load_file(path, device=self.device)
            all_fishers.append(fisher)
            print(f"Loaded Fisher: {path}")

        # Remove .default from all keys
        if remove_default_ettiquete:
            for fisher in all_fishers:
                for key in list(fisher.keys()):
                    if ".default" in key:
                        new_key = key.replace(".default", "")
                        fisher[new_key] = fisher.pop(key)
        return all_fishers

    def merge_formula(
        self,
        weights_list: List[Tensor],
        fisher_list: Optional[List[Dict[str, Tensor]]] = None,
        mode: MergeMode = "standard",
        alphas: Optional[Tensor] = None,
    ) -> Tensor:
        """Merge OFT task vectors using the specified mode.

        Args:
            weights_list: T tensors of shape (num_blocks, d), skew-symmetric so(n) parameters per task.
            fisher_list: Per-task Fisher info — diagonal (num_blocks, d) for ``"diagonal_fisher"``,
                full (num_blocks, d, d) for ``"fisher"``. Unused in ``"standard"`` mode.
            mode: ``"standard"`` (alpha-weighted avg), ``"diagonal_fisher"``, or ``"fisher"``
                (both Fisher modes transport to a common tangent space and solve a linear system).
            alphas: Optional (T,) tensor overriding self.alphas. Kept differentiable when provided.

        Returns:
            Merged OFT parameters of shape (num_blocks, d).

        Raises:
            ValueError: If ``fisher_list`` is None for a Fisher mode, or ``mode`` is unsupported.
        """
        T = len(weights_list)
        if alphas is None:
            alphas = self._resolve_alphas(T).to(self.device)

        if mode == "standard":
            return self._standard_merging(weights_list, alphas)

        if mode == "standard_rescaled":
            # assumes alpha_t = 1 and then applies orthomerge correction
            merged = self._standard_merging(weights_list, alphas)
            return self._orthomerge_rescale(weights_list, merged, alphas)

        if mode == "diagonal_fisher":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            return self._diagonal_fisher_merging(weights_list, fisher_list, alphas)

        if mode == "diagonal_fisher_avg":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            avg_alphas = torch.full_like(alphas, 1.0 / len(weights_list))
            return self._diagonal_fisher_merging(
                weights_list,
                fisher_list,
                avg_alphas,
            )

        if mode == "diagonal_fisher_rescaled":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            merged = self._diagonal_fisher_merging(weights_list, fisher_list, alphas)
            return self._fisher_norm_rescale(weights_list, fisher_list, merged, alphas)

        if mode == "diagonal_fisher_max_rescaled":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            merged = self._diagonal_fisher_merging(weights_list, fisher_list, alphas)
            return self._fisher_norm_max_rescale(weights_list, fisher_list, merged, alphas)

        if mode == "diagonal_fisher_std_rescaled":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            merged = self._diagonal_fisher_merging(weights_list, fisher_list, alphas)
            return self._standard_norm_rescale(weights_list, merged, alphas)

        if mode == "diagonal_fisher_kl_rescaled":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            return self._diagonal_fisher_merging_kl_rescaled(
                weights_list=weights_list,
                fisher_list=fisher_list,
                alphas=alphas,
            )

        if mode == "fisher":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            return self._fisher_merging(weights_list, fisher_list, alphas)

        raise ValueError(f"Unsupported merge mode: {mode!r}")

    def apply_alphas(self, weights_list: List[Tensor], alphas: Tensor) -> Tensor:
        """Differentiable weighted sum of task vectors; alphas is (T,) or (T, L) broadcast to (T,).
        """
        stacked = torch.stack(weights_list, dim=0).float()  # (T, ...)
        a = alphas.to(stacked)
        if a.dim() == 0:
            a = a.unsqueeze(0).expand(stacked.shape[0])
        return torch.einsum("t,t...->...", a, stacked)

    def _standard_merging(self, weights_list: List[Tensor], alphas) -> Tensor:
        """Compute a weighted average of OFT task vectors in so(n).

        Args:
            weights_list: T tensors of shape (num_blocks, d), the skew-symmetric parameters per task.
            alphas: T scalars or a (T,) tensor, one mixing coefficient per task.

        Returns:
            Merged parameters of shape (num_blocks, d).
        """
        num_models = len(weights_list)
        stacked = torch.stack(weights_list, dim=0)  # (T, num_blocks, n, n)
        if isinstance(alphas, torch.Tensor):
            a = alphas.to(dtype=stacked.dtype, device=stacked.device)
        else:
            a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)
        return torch.einsum("t,t...->...", a, stacked) / num_models

    def _orthomerge_rescale(
        self,
        weights_list: List[Tensor],
        merged: Tensor,
        alphas,
    ) -> Tensor:
        """Apply the OrthoMerge Frobenius-norm correction.

        Args:
            weights_list: T tensors of shape (num_blocks, d), OFT parameters per task.
            merged: Merged OFT parameters of shape (num_blocks, d).
            alphas: T scalars or a (T,) tensor, one mixing coefficient per task.

        Returns:
            Rescaled merged parameters of shape (num_blocks, d).
        """
        num_models = len(weights_list)
        stacked = torch.stack(weights_list, dim=0).float().to(merged.device)
        if isinstance(alphas, torch.Tensor):
            a = alphas.to(dtype=stacked.dtype, device=stacked.device)
        else:
            a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)

        while a.dim() < stacked.dim():
            a = a.unsqueeze(-1)

        weighted = a * stacked
        sum_of_norms = torch.norm(weighted.flatten(1), p="fro", dim=1).sum()
        norm_of_merged = torch.norm(num_models * merged.float(), p="fro")
        correction = sum_of_norms / norm_of_merged.clamp(min=1e-8)
        print("[orthomerge_rescale] " f"sum_of_norms={sum_of_norms.item():.6g}, " f"norm_of_sum={norm_of_merged.item():.6g}, " f"correction={correction.item():.6g}")
        return (correction * merged).to(dtype=weights_list[0].dtype)

    def _fisher_norm_rescale(
            self,
            weights_list: List[Tensor],
            fisher_list: List[Tensor],
            merged: Tensor,
            alphas,
        ) -> Tensor:
            """Apply the OrthoMerge norm correction with diagonal Fisher metrics.

            Args:
                weights_list: T tensors of shape (num_blocks, d), OFT parameters per task.
                fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates per task.
                merged: Merged OFT parameters of shape (num_blocks, d).
                alphas: T scalars or a (T,) tensor, one mixing coefficient per task.

            Returns:
                Rescaled merged parameters of shape (num_blocks, d).
            """
            stacked = torch.stack(weights_list, dim=0).float().to(merged.device)
            fishers = torch.stack(
                [f.float().to(merged.device).clamp(min=0.0) for f in fisher_list],
                dim=0,
            )

            if isinstance(alphas, torch.Tensor):
                a = alphas.to(dtype=stacked.dtype, device=stacked.device)
            else:
                a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)

            while a.dim() < stacked.dim():
                a = a.unsqueeze(-1)

            individual_norms = torch.sqrt(
                (fishers * stacked.pow(2)).flatten(1).sum(dim=1).clamp(min=0.0)
            )

            a_flat = a.reshape(a.shape[0], -1)[:, 0]
            sum_of_norms = (a_flat.abs() * individual_norms).sum()

            fisher_sum = (a.abs() * fishers).sum(dim=0)

            norm_of_merged = torch.sqrt(
                (fisher_sum * merged.float().pow(2)).sum().clamp(min=0.0)
            )

            correction = sum_of_norms / norm_of_merged.clamp(min=1e-8)

            print(
                "[fisher_norm_rescale] "
                f"sum_of_norms={sum_of_norms.item():.6g}, "
                f"norm_of_sum={norm_of_merged.item():.6g}, "
                f"correction={correction.item():.6g}"
            )

            new_merged = (correction * merged).to(dtype=weights_list[0].dtype)
            return new_merged

    def _fisher_norm_max_rescale(
            self,
            weights_list: List[Tensor],
            fisher_list: List[Tensor],
            merged: Tensor,
            alphas,
        ) -> Tensor:
            """Apply Fisher-energy lower-bound correction with diagonal Fisher metrics.

            Args:
                weights_list: T tensors of shape (num_blocks, d), OFT parameters per task.
                fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates per task.
                merged: Merged OFT parameters of shape (num_blocks, d).
                alphas: T scalars or a (T,) tensor, one mixing coefficient per task.

            Returns:
                Rescaled merged parameters of shape (num_blocks, d).
            """
            stacked = torch.stack(weights_list, dim=0).float().to(merged.device)
            fishers = torch.stack(
                [f.float().to(merged.device).clamp(min=0.0) for f in fisher_list],
                dim=0,
            )

            if isinstance(alphas, torch.Tensor):
                a = alphas.to(dtype=stacked.dtype, device=stacked.device)
            else:
                a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)

            while a.dim() < stacked.dim():
                a = a.unsqueeze(-1)

            a_flat = a.reshape(a.shape[0], -1)[:, 0].abs()

            target_energies = (fishers * stacked.pow(2)).flatten(1).sum(dim=1)
            target_energies = a_flat * target_energies

            merged_energies = (fishers * merged.float().pow(2)).flatten(1).sum(dim=1)

            ratios = target_energies / merged_energies.clamp(min=1e-8)
            correction = torch.sqrt(ratios.clamp(min=0.0)).max()

            print(
                "[fisher_norm_max_rescale] "
                f"max_target={target_energies.max().item():.6g}, "
                f"min_merged={merged_energies.min().item():.6g}, "
                f"correction={correction.item():.6g}"
            )

            new_merged = (correction * merged).to(dtype=weights_list[0].dtype)
            return new_merged

    def _standard_norm_rescale(
        self,
        weights_list: List[Tensor],
        merged: Tensor,
        alphas,
    ) -> Tensor:
        stacked = torch.stack(weights_list, dim=0).float().to(merged.device)

        if isinstance(alphas, torch.Tensor):
            a = alphas.to(dtype=stacked.dtype, device=stacked.device)
        else:
            a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)

        while a.dim() < stacked.dim():
            a = a.unsqueeze(-1)

        individual_norms = torch.norm(stacked.flatten(1), p="fro", dim=1)
        a_flat = a.reshape(a.shape[0], -1)[:, 0].abs()
        alpha_bar_t = a_flat / a_flat.sum().clamp(min=1e-8)  # normalized alphas
        target_norm = (alpha_bar_t * individual_norms).sum()
        norm_of_merged = torch.norm(merged.float(), p="fro")
        correction = target_norm / norm_of_merged.clamp(min=1e-8)

        print(
            "[standard_norm_rescale] "
            f"target_norm={target_norm.item():.6g}, "
            f"norm_of_merged={norm_of_merged.item():.6g}, "
            f"correction={correction.item():.6g}"
        )

        return (correction * merged).to(dtype=weights_list[0].dtype)

    def _fisher_diagonal_alphas(self, fisher_list: List[Tensor]) -> Tensor:
        """Compute per-task alphas as alpha_t = sum_s <f_t, f_s> / Z.

        For diagonal Fishers, trace(F_t F_s) = dot(f_t, f_s) (summed over all
        blocks and coordinates). Tasks whose Fisher is most aligned with the
        others receive higher weight. Alphas are L1-normalised to sum to 1.

        Args:
            fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates.

        Returns:
            Normalised alpha tensor of shape (T,).
        """
        flat = [f.flatten().float() for f in fisher_list]        # T x (num_blocks*d,)
        F = torch.stack(flat)                                     # (T, D)
        F = F / F.norm(dim=1, keepdim=True).clamp(min=1e-8)      # normalise by Frobenius norm
        scores = (F @ F.T).sum(dim=1)                            # (T,), alpha_t = sum_s <f_t, f_s>
        alphas = scores / scores.sum().clamp(min=1e-8)
        return 1 - alphas.to(fisher_list[0].device)

    def _diagonal_fisher_merging(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Tensor],
        alphas,
    ) -> Tensor:
        """Merge OFT adapters using diagonal Fisher information as per-parameter weights.

        Since F_tilde = diag(fisher_t), the full matrix system A @ omega = b reduces to
        an element-wise division: omega_i = b_i / (lam + A_i), where
            A_i = sum_t alpha_t * fisher_t_i
            b_i = sum_t alpha_t * (lam + fisher_t_i) * omega_t_i

        Args:
            weights_list: T tensors of shape (num_blocks, d).
            fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates.
            alphas: T scalars or (T,) tensor.

        Returns:
            Merged parameters of shape (num_blocks, d).
        """
        ref = weights_list[0]
        A = torch.zeros_like(ref, dtype=torch.float32)  # sum_t alpha_t * f_t
        b = torch.zeros_like(ref, dtype=torch.float32)  # sum_t alpha_t * (lam + f_t) * omega_t

        for alpha_t, omega_t, f_t in zip(alphas, weights_list, fisher_list):
            ft = f_t.float().to(ref.device)
            A = A + alpha_t * ft
            b = b + alpha_t * (self.lam + ft) * omega_t.float().to(ref.device)

        return (b / (self.lam + A).clamp(min=1e-8)).to(ref.dtype)

    def _diagonal_fisher_merging_kl_rescaled(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Tensor],
        alphas,
        base_fisher_list: Optional[List[Tensor]] = None,
        eps: float = 1e-8,
        tol: float = 1e-6,
        max_outer: int = 20,
        max_newton: int = 50,
        damping: float = 1e-4,
    ) -> Tensor:
        """
        Diagonal Fisher OFT merge with task-wise KL/Fisher lower-bound constraints.

        Solves, on the positive-definite KKT branch,

            min_w  1/2 w^T diag(D) w - b^T w

        subject to, for each dataset/task k,

            1/2 w^T diag(H_k) w >= 1/2 xi_k^T diag(H_k) xi_k.

        where

            D = lambda + sum_t alpha_t f_t
            b = sum_t alpha_t (lambda + f_t) * xi_t

        and H_k is the pretrained/base Fisher on dataset k. If base_fisher_list is
        not provided, fisher_list is used as a surrogate.

        Args:
            weights_list: T tensors of shape (num_blocks, d), OFT coordinates xi_t.
            fisher_list: T tensors of shape (num_blocks, d), diagonal transported Fishers f_t.
            alphas: T coefficients.
            base_fisher_list: optional T tensors of shape (num_blocks, d), base Fishers H_k.
            eps: numerical floor.
            tol: feasibility/KKT tolerance.
            max_outer: active-set iterations.
            max_newton: Newton iterations per active set.
            damping: diagonal damping added to the small Newton system.

        Returns:
            Merged OFT parameters of shape (num_blocks, d).
        """
        ref = weights_list[0]
        device = ref.device

        if isinstance(alphas, torch.Tensor):
            alphas_t = alphas.to(device=device, dtype=torch.float32)
        else:
            alphas_t = torch.tensor(alphas, device=device, dtype=torch.float32)

        weights = [
            w.to(device=device, dtype=torch.float32)
            for w in weights_list
        ]

        fishers = [
            f.to(device=device, dtype=torch.float32).clamp(min=0.0)
            for f in fisher_list
        ]

        if base_fisher_list is None:
            # Surrogate: use the task Fisher as the dataset-wise KL metric.
            H_list = fishers
        else:
            # Preferred: H_k = Fisher of pretrained/base model on dataset k.
            H_list = [
                h.to(device=device, dtype=torch.float32).clamp(min=0.0)
                for h in base_fisher_list
            ]

        T = len(weights)

        # Build the diagonal linear system:
        #   D * omega = b
        # where D = lambda + sum_t alpha_t f_t.
        A_diag = torch.zeros_like(ref, dtype=torch.float32, device=device)
        b = torch.zeros_like(ref, dtype=torch.float32, device=device)

        for alpha_t, omega_t, f_t in zip(alphas_t, weights, fishers):
            A_diag = A_diag + alpha_t * f_t
            b = b + alpha_t * (self.lam + f_t) * omega_t

        D = (self.lam + A_diag).clamp(min=eps)

        # Unconstrained diagonal Fisher merge.
        omega_hat = b / D

        H_stack = torch.stack(H_list, dim=0)  # (T, num_blocks, d)

        # Dataset-wise thresholds:
        #   delta_k = 1/2 xi_k^T H_k xi_k.
        deltas = torch.stack([
            0.5 * (H_k * omega_k.pow(2)).sum()
            for H_k, omega_k in zip(H_list, weights)
        ])

        # Check whether the unconstrained merge already satisfies every constraint.
        infos_hat = 0.5 * (
            H_stack * omega_hat.pow(2).unsqueeze(0)
        ).flatten(1).sum(dim=1)

        if torch.all(infos_hat >= deltas - tol):
            return omega_hat.to(dtype=ref.dtype)

        # Active-set initialization: constraints violated by the unconstrained merge.
        active = infos_hat < deltas - tol
        mu_full = torch.zeros(T, device=device, dtype=torch.float32)

        omega = omega_hat

        for _ in range(max_outer):
            active_idx = torch.nonzero(active, as_tuple=False).flatten()

            if active_idx.numel() == 0:
                omega = b / D
                break

            mu = mu_full[active_idx].clone()
            H_A = H_stack[active_idx]  # (m, num_blocks, d)

            # Newton solve for active multipliers.
            for _ in range(max_newton):
                denom = D - torch.einsum("m,m...->...", mu, H_A)

                # Stay on the positive-definite branch.
                if denom.min() <= eps:
                    mu = 0.5 * mu
                    continue

                omega = b / denom

                infos_A = 0.5 * (
                    H_A * omega.pow(2).unsqueeze(0)
                ).flatten(1).sum(dim=1)

                r = infos_A - deltas[active_idx]

                if r.abs().max() < tol:
                    break

                m = active_idx.numel()
                J = torch.empty((m, m), device=device, dtype=torch.float32)

                # Jacobian:
                #   J[k,j] = sum_i h_k_i h_j_i b_i^2 / denom_i^3.
                denom3 = denom.pow(3).clamp(min=eps)
                b2 = b.pow(2)

                for a in range(m):
                    for c in range(m):
                        J[a, c] = (H_A[a] * H_A[c] * b2 / denom3).sum()

                J = J + damping * torch.eye(m, device=device, dtype=torch.float32)

                try:
                    step = torch.linalg.solve(J, -r)
                except RuntimeError:
                    step = torch.linalg.lstsq(J, -r.unsqueeze(-1)).solution.squeeze(-1)

                # Backtracking to preserve mu >= 0 and denom > 0.
                step_scale = 1.0
                accepted = False

                for _bt in range(30):
                    mu_new = (mu + step_scale * step).clamp(min=0.0)
                    denom_new = D - torch.einsum("m,m...->...", mu_new, H_A)

                    if denom_new.min() > eps:
                        omega_new = b / denom_new

                        infos_new = 0.5 * (
                            H_A * omega_new.pow(2).unsqueeze(0)
                        ).flatten(1).sum(dim=1)

                        r_new = infos_new - deltas[active_idx]

                        if r_new.abs().sum() <= r.abs().sum() or step_scale < 1e-4:
                            mu = mu_new
                            accepted = True
                            break

                    step_scale *= 0.5

                if not accepted:
                    break

            # Write active multipliers back into the full vector.
            mu_full[active_idx] = mu

            denom = D - torch.einsum("t,t...->...", mu_full, H_stack)

            if denom.min() <= eps:
                denom = denom.clamp(min=eps)

            omega = b / denom

            all_infos = 0.5 * (
                H_stack * omega.pow(2).unsqueeze(0)
            ).flatten(1).sum(dim=1)

            slack = all_infos - deltas

            feasible = torch.all(slack >= -tol)
            comp_ok = torch.all((mu_full <= tol) | (slack.abs() <= tol))

            if feasible and comp_ok:
                return omega.to(dtype=ref.dtype)

            # Add newly violated constraints.
            active = active | (slack < -tol)

            # Remove constraints with nearly zero multiplier and positive slack.
            active = active & ~((mu_full <= tol) & (slack > tol))

        # Final feasibility check.
        final_infos = 0.5 * (
            H_stack * omega.pow(2).unsqueeze(0)
        ).flatten(1).sum(dim=1)

        if torch.any(final_infos < deltas - tol):
            print("[WARNING] Multi-KL Lagrange solver returned an infeasible point.")

        return omega.to(dtype=ref.dtype)

    def _fisher_merging(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Tensor],
        alphas: List[float],
    ) -> Tensor:
        """Merge OFT adapters using full Fisher information matrices as per-parameter weights.

        Args:
            weights_list: T tensors of shape (num_blocks, d), skew-symmetric parameters per task.
            fisher_list: T tensors of shape (num_blocks, d, d), full Fisher matrices per task.
            alphas: T scalars, one mixing coefficient per task.

        Returns:
            Merged parameters of shape (num_blocks, d).
        """
        raise NotImplementedError("Full Fisher merging is not implemented yet.")
        num_blocks, son_dimension = weights_list[0].shape
        dtype, device = weights_list[0].dtype, weights_list[0].device
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)

        Id = torch.eye(son_dimension, dtype=dtype, device=device).unsqueeze(0)

        A = self.lam * Id.expand(num_blocks, -1, -1).clone()
        b = torch.zeros(num_blocks, son_dimension, dtype=dtype, device=device)

        for alpha_t, oft_params_t, fisher_t in zip(alphas, weights_list, fisher_list):
            # Reconstruct skew-symmetric matrices
            skew_matrix = self.oft_params_to_skew_matrix(oft_params_t, son_dimension)  # (num_blocks, n, n)
            Pt = self.manifold.compute_Pt(skew_matrix, block_size)  # (num_blocks, d, d)

            F_tilde = Pt @ fisher_t @ Pt.transpose(-2, -1)

            A += alpha_t * F_tilde
            rhs = (self.lam * Id.squeeze(0) + F_tilde) @ oft_params_t.unsqueeze(-1)
            b += alpha_t * rhs.squeeze(-1)

        return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    def fisher_full_task_vectors(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Tensor],
        alphas: List[float],
    ) -> List[Tensor]:
        """Return the Fisher-corrected task vectors

            v_t = (lambda I_d + sum_j alpha_j F_tilde_j)^(-1) (lambda I_d + F_tilde_t) xi_t

        for each task t.

        Args:
            weights_list: T tensors of shape (num_blocks, d), skew-symmetric parameters per task.
            fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates per task.
            alphas: T scalars, one mixing coefficient per task.

        Returns:
            List of T tensors, each of shape (num_blocks, d).
        """
        num_blocks, son_dimension = weights_list[0].shape
        dtype, device = weights_list[0].dtype, weights_list[0].device
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)

        Id = torch.eye(son_dimension, dtype=dtype, device=device).unsqueeze(0)  # (1, d, d)

        # First build all transported Fishers
        F_tilde_list = []
        for oft_params_t, fisher_t in zip(weights_list, fisher_list):
            skew_matrix = self.oft_params_to_skew_matrix(oft_params_t, son_dimension)  # (num_blocks, n, n)
            # Pt = self.manifold.compute_Pt(skew_matrix=skew_matrix, block_size=block_size)  # (num_blocks, d, d)
            # F_tilde = Pt @ torch.diag_embed(fisher_t) @ Pt.transpose(-2, -1)  # (num_blocks, d, d)
            F_tilde = torch.diag_embed(fisher_t)  # (num_blocks, d, d), no transport for simplicity
            # F_tilde = torch.diag_embed(fisher_t)  # (num_blocks, d, d), no transport for simplicity
            F_tilde_list.append(F_tilde)

        # Shared inverse term: (lambda I_d + sum_t alpha_t F_tilde_t)^(-1)
        A = self.lam * Id.expand(num_blocks, -1, -1).clone()  # (num_blocks, d, d)
        for alpha_t, F_tilde in zip(alphas, F_tilde_list):
            A += alpha_t * F_tilde

        fisher_task_vectors = []
        for oft_params_t, F_tilde_t in zip(weights_list, F_tilde_list):
            rhs = ((self.lam * Id.squeeze(0) + F_tilde_t) @ oft_params_t.unsqueeze(-1)).squeeze(-1)
            v_t = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
            fisher_task_vectors.append(v_t)

        return fisher_task_vectors

    def fisher_task_vectors(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Tensor],
        alphas: List[float] = None,
    ) -> List[Tensor]:
        """Return the Fisher-weighted task vectors

            v_t = (lambda I_d + F_tilde_t) xi_t

        for each task t.

        Args:
            weights_list: T tensors of shape (num_blocks, d), skew-symmetric parameters per task.
            fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates per task.

        Returns:
            List of T tensors, each of shape (num_blocks, d).
        """
        num_blocks, son_dimension = weights_list[0].shape
        dtype, device = weights_list[0].dtype, weights_list[0].device
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)

        Id = torch.eye(son_dimension, dtype=dtype, device=device).unsqueeze(0)  # (1, d, d)

        fisher_task_vectors = []
        for oft_params_t, fisher_t in zip(weights_list, fisher_list):
            skew_matrix = self.oft_params_to_skew_matrix(oft_params_t, son_dimension)  # (num_blocks, n, n)
            Pt = self.manifold.compute_Pt(skew_matrix=skew_matrix, block_size=block_size)  # (num_blocks, d, d)

            F_tilde_t = Pt @ torch.diag_embed(fisher_t) @ Pt.transpose(-2, -1)  # (num_blocks, d, d)

            v_t = ((self.lam * Id.squeeze(0) + F_tilde_t) @ oft_params_t.unsqueeze(-1)).squeeze(-1)
            fisher_task_vectors.append(v_t)

        return fisher_task_vectors

    def merge(
        self,
        adapter_paths: List[str],
        fisher_paths: Optional[List[str]] = None,
        mode: MergeMode = "standard",
    ):
        print(f"\nMerging {len(adapter_paths)} OFT adapters with {mode} mode...")

        # Load weights
        all_weights = self.load_weights(adapter_paths)
        if not all_weights:
            raise ValueError("No valid adapter weights found!")

        # Load Fishers
        all_fishers = self.load_fishers(fisher_paths) if fisher_paths else None

        merged_weights = {}

        # Merge per key
        for key in all_weights[0].keys():
            if "oft_r" in key or ("oft_" in key.lower() and "classifier" not in key.lower()):
                print(f"  Processing key: {key}")

                weights_layer = [weights[key] for weights in all_weights]

                fishers_layer = None
                if all_fishers:
                    fishers_layer = [fisher[key] for fisher in all_fishers]

                avg_weight = self.merge_formula(
                    weights_list=weights_layer,
                    fisher_list=fishers_layer,
                    mode=mode,
                )

                merged_weights[key] = avg_weight

            else:
                print("[WARNING] Non-OFT weight detected.")

        print(f"  Merged {len(merged_weights)} weight tensors")
        return merged_weights
