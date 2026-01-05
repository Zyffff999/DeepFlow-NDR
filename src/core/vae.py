"""
Optimized CorrelatedGaussianVAE
===============================

Refactored for stability and simplicity:
1. Basic Cholesky construction (Softplus diagonal + Raw lower triangle).
2. Standard analytical KL divergence for Multivariate Normal.
3. Removed aggressive condition number clamping (let the loss guide the gradients).
4. Preserved all interfaces and other training modes (diagonal/gmm/ae).
"""

from src.core.packet import PacketEncoder, FactorizedDecoder, SessionEncoder
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def _numel(t: torch.Tensor) -> int:
    return int(t.numel())


def ce_loss_chunked(
        pos_feat: Optional[torch.Tensor],
        targets: torch.Tensor,
        classifier: Optional[nn.Linear] = None,
        logits: Optional[torch.Tensor] = None,
        chunk: int = 262144
) -> torch.Tensor:
    assert (pos_feat is not None) ^ (logits is not None), "提供 pos_feat+classifier 或直接 logits（二选一）。"
    if logits is not None:
        B, P, L, V = logits.shape
        y = targets.reshape(-1).long()
        x = logits.reshape(-1, V)
        return F.cross_entropy(x, y, reduction='mean')

    # pos_feat 路线
    assert classifier is not None, "使用 pos_feat 时必须提供共享分类头 classifier。"
    B, P, L, d = pos_feat.shape
    N = B * P * L
    x = pos_feat.reshape(N, d)
    y = targets.reshape(N).long()

    total = 0.0
    i = 0
    while i < N:
        j = min(i + chunk, N)
        logits_i = classifier(x[i:j])  # [chunk, vocab]
        total += F.cross_entropy(logits_i, y[i:j], reduction='sum')
        i = j
    return total / N


class CorrelatedGaussianVAE(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            training_mode: str = "diagonal",
            n_components: int = 3,
            min_logvar: float = -10.0, # Adjusted defaults to be less aggressive
            max_logvar: float = 10.0,
            hidden_dim: int = 96,
            dropout: float = 0.1,
            encoder_cfg: Optional[Dict[str, Any]] = None,
            decoder_cfg: Optional[Dict[str, Any]] = None,
            global_dim: int = 0,
            global_mlp_hidden: Optional[int] = None,
            global_mlp_out: Optional[int] = None,
            global_fuse: str = "concat",
            # Kept for compatibility but unused in optimized version
            **kwargs
    ):
        super().__init__()

        assert training_mode in ["diagonal", "correlated", "gmm", "ae"], \
            "training_mode must be 'diagonal', 'correlated', 'gmm', or 'ae'"

        self.latent_dim = latent_dim
        self.training_mode = training_mode
        self.n_components = n_components
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.global_dim = int(global_dim or 0)
        self.global_fuse = str(global_fuse or "concat").lower()
        if self.global_fuse != "concat":
            raise ValueError("global_fuse must be 'concat'.")

        # ---------------- Encoder / Decoder Config ----------------
        encoder_defaults: Dict[str, Any] = {
            "packet_latent_dim": 128,
            "d_model": 128,
            "latent_dim": latent_dim,
            "spatial_reduction_size": 8
        }
        if encoder_cfg is not None:
            encoder_defaults.update(encoder_cfg)

        decoder_defaults: Dict[str, Any] = {
            "latent_dim": latent_dim,
            "output_size": 256 * 16,  # 16 packets * 256 bytes
            "rank": 192,
        }
        if decoder_cfg is not None:
            decoder_defaults.update(decoder_cfg)

        self.encoder = SessionEncoder(**encoder_defaults)
        self.decoder = FactorizedDecoder(**decoder_defaults)

        # ---------------- Global Encoder + Fusion ----------------
        self.byte_feat_dim = int(getattr(self.encoder, "final_C", latent_dim))
        self.global_mlp_out = int(global_mlp_out or latent_dim)
        global_hidden = int(global_mlp_hidden or max(64, self.global_mlp_out))

        if self.global_dim > 0:
            self.global_encoder = nn.Sequential(
                nn.BatchNorm1d(self.global_dim),
                nn.Linear(self.global_dim, global_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(global_hidden, global_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(global_hidden, self.global_mlp_out),
                nn.GELU(),
                nn.LayerNorm(self.global_mlp_out),
            )
        else:
            self.global_encoder = None

        if self.global_dim > 0 and self.global_fuse == "concat":
            fuse_in = self.byte_feat_dim + self.global_mlp_out
        else:
            fuse_in = self.byte_feat_dim

        # ---------------- Bottleneck + Heads ----------------
        self.bottleneck_fc = nn.Sequential(
            nn.Linear(fuse_in, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
        )

        self.fc_mu = nn.Linear(latent_dim, latent_dim)

        if training_mode == "diagonal":
            self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        elif training_mode == "correlated":
            # 1. Diagonal elements of Cholesky L
            self.fc_L_diag = nn.Linear(latent_dim, latent_dim)
            # 2. Lower triangular elements (excluding diagonal)
            self.num_lower_elements = latent_dim * (latent_dim - 1) // 2
            self.fc_L_lower = (
                nn.Linear(latent_dim, self.num_lower_elements)
                if self.num_lower_elements > 0
                else None
            )

        elif training_mode == "gmm":
            self.fc_logvar = nn.Linear(latent_dim, latent_dim)
            self.register_parameter("gmm_logits", nn.Parameter(torch.zeros(n_components)))
            self.register_parameter(
                "gmm_means",
                nn.Parameter(torch.randn(n_components, latent_dim) * 0.01),
            )
            self.register_parameter(
                "gmm_logvars",
                nn.Parameter(torch.full((n_components, latent_dim), -2.0)),
            )

        elif training_mode == "ae":
            pass # No variance head needed

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.bottleneck_fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.zeros_(self.fc_mu.bias)

        if self.training_mode == "diagonal":
            nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.01)
            nn.init.constant_(self.fc_logvar.bias, -2.0)

        elif self.training_mode == "correlated":
            # Initialize L_diag to produce values close to 1.0 (Identity matrix)
            # Softplus(0.55) approx 1.0
            nn.init.xavier_normal_(self.fc_L_diag.weight, gain=0.01)
            nn.init.constant_(self.fc_L_diag.bias, 0.55)

            if self.fc_L_lower is not None:
                # Initialize off-diagonals to 0 (Independent start)
                nn.init.zeros_(self.fc_L_lower.weight)
                nn.init.zeros_(self.fc_L_lower.bias)

        elif self.training_mode == "gmm":
            nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.01)

    # --------------------------------------------------------------------- #
    # GMM Properties
    # --------------------------------------------------------------------- #
    @property
    def gmm_weights(self):
        if self.training_mode != "gmm": return None
        return F.softmax(self.gmm_logits, dim=0)

    @property
    def gmm_vars(self):
        if self.training_mode != "gmm": return None
        return torch.exp(torch.clamp(self.gmm_logvars, self.min_logvar, self.max_logvar))

    # --------------------------------------------------------------------- #
    # Basic Cholesky Construction (Optimized)
    # --------------------------------------------------------------------- #
    def construct_cholesky_matrix(self, L_diag_raw: torch.Tensor, L_lower: torch.Tensor) -> torch.Tensor:
        """
        Constructs Lower Triangular Matrix L where Σ = L @ L.T
        Uses Softplus for diagonal positivity. No complex eigenvalues clamping.
        """
        B, D = L_diag_raw.shape
        device = L_diag_raw.device

        # 1. Diagonal: Softplus ensures positivity. Adding epsilon prevents collapse to 0.
        L_diag = F.softplus(L_diag_raw) + 1e-5

        # 2. Create the matrix container
        L = torch.zeros(B, D, D, device=device, dtype=L_diag_raw.dtype)

        # 3. Fill Diagonal
        idx = torch.arange(D, device=device)
        L[:, idx, idx] = L_diag

        # 4. Fill Off-Diagonal (Lower Triangle)
        if L_lower is not None and L_lower.shape[-1] > 0:
            # We map the flat vector L_lower into the lower triangle indices
            row_idx, col_idx = torch.tril_indices(D, D, offset=-1, device=device)
            L[:, row_idx, col_idx] = L_lower

        return L

    # --------------------------------------------------------------------- #
    # Forward Pass
    # --------------------------------------------------------------------- #
    def forward(self, data: torch.Tensor, global_stats: Optional[torch.Tensor] = None) -> Tuple:

        byte_features = self.encoder(data)
        if self.global_encoder is not None and global_stats is not None:
            global_vec = self.global_encoder(global_stats)
            features = torch.cat([byte_features, global_vec], dim=-1)
        else:
            features = byte_features
        bottleneck = self.bottleneck_fc(features)
        mu = self.fc_mu(bottleneck)

        if self.training_mode == "diagonal":
            logvar = self.fc_logvar(bottleneck)
            logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu
            latent_params = logvar

        elif self.training_mode == "correlated":
            # Predict raw params
            L_diag_raw = self.fc_L_diag(bottleneck)
            L_lower = self.fc_L_lower(bottleneck) if self.fc_L_lower else None

            # Construct L
            L = self.construct_cholesky_matrix(L_diag_raw, L_lower)

            if self.training:
                # Reparameterization: z = mu + L @ epsilon
                eps = torch.randn(mu.shape[0], self.latent_dim, 1, device=mu.device)
                # bmm: [B, D, D] x [B, D, 1] -> [B, D, 1]
                noise = torch.bmm(L, eps).squeeze(-1)
                z = mu + noise
            else:
                z = mu
            latent_params = L

        elif self.training_mode == "gmm":
            # (Standard GMM logic kept same)
            logvar = self.fc_logvar(bottleneck)
            logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu
            latent_params = logvar

        elif self.training_mode == "ae":
            z = mu
            latent_params = None

        else:
            raise NotImplementedError

        reconstructed_output = self.decoder(z)
        return z, mu, latent_params, reconstructed_output

    def compute_kl_divergence(
            self,
            mu: torch.Tensor,
            latent_params: torch.Tensor,
            correlation_penalty_weight: float = 1.0  # NEW PARAMETER
        ) -> torch.Tensor:

        if self.training_mode == "diagonal":
            logvar = latent_params
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            return kl

        elif self.training_mode == "correlated":
            L = latent_params  # [B, D, D]
            D = self.latent_dim

            # --- Decomposed Trace Calculation ---
            # Tr(LL^T) = sum(L_ij^2). We split this into diagonal and off-diagonal.

            # 1. Diagonal Cost (Variance magnitude)
            # L_diag are the elements on the diagonal
            L_diag = torch.diagonal(L, dim1=-2, dim2=-1)
            trace_diag = torch.sum(L_diag.pow(2), dim=1)

            # 2. Off-Diagonal Cost (Correlation strength)
            # Total sum of squares minus diagonal sum of squares
            total_sq = torch.sum(L.pow(2), dim=(1, 2))
            trace_off_diag = total_sq - trace_diag

            # Apply the relaxation factor to the correlation part only
            weighted_trace = trace_diag + (trace_off_diag * correlation_penalty_weight)

            # --- Mean and LogDet Terms (Standard) ---
            mu_sq = torch.sum(mu.pow(2), dim=1)
            log_det_cov = 2 * torch.sum(torch.log(L_diag + 1e-10), dim=1)

            # Modified KL formula
            kl = 0.5 * (weighted_trace + mu_sq - D - log_det_cov)
            return kl

        elif self.training_mode == "gmm":
            return self.compute_gmm_kl_stable(mu, latent_params)

        elif self.training_mode == "ae":
            return torch.zeros(mu.shape[0], device=mu.device)

        else:
            raise NotImplementedError

    def compute_gmm_kl_stable(self, mu, logvar, n_samples=5):
        """
        Computes KL(q(z|x) || p(z)) using pure Monte Carlo sampling.
        Ensures consistency and non-negativity.
        """
        # Constants
        batch_size, latent_dim = mu.shape
        log_2pi = 1.837877

        kl_samples = []

        for _ in range(n_samples):
            # 1. Sample z from q(z|x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # [B, D]

            # 2. Log q(z|x) (Log-density of the
            #
            # posterior at sampled z)
            # log N(z; mu, var) = -0.5 * (log(2pi) + logvar + (z-mu)^2/var)
            # Since z = mu + eps*std, (z-mu)^2/var = eps^2
            log_q_z = -0.5 * torch.sum(log_2pi + logvar + eps.pow(2), dim=1)  # [B]

            # 3. Log p(z) (Log-density of the GMM prior at sampled z)
            z_expanded = z.unsqueeze(1)  # [B, 1, D]
            means_expanded = self.gmm_means.unsqueeze(0)  # [1, K, D]
            logvars_expanded = self.gmm_logvars.unsqueeze(0)  # [1, K, D]

            # Log-prob for each component k
            # -0.5 * sum(log(2pi) + logvar_k + (z - mu_k)^2 / var_k)
            log_p_k = -0.5 * torch.sum(
                log_2pi + logvars_expanded +
                (z_expanded - means_expanded).pow(2) / torch.exp(logvars_expanded),
                dim=2
            )  # [B, K]

            # Mixture log-prob: log sum (w_k * exp(log_p_k))
            # = log sum exp(log_w_k + log_p_k)
            log_w = F.log_softmax(self.gmm_logits, dim=0)  # [K]
            log_p_z = torch.logsumexp(log_w.unsqueeze(0) + log_p_k, dim=1)  # [B]

            # 4. KL Sample = log q(z) - log p(z)
            kl_samples.append(log_q_z - log_p_z)

        # Average over samples
        kl_mean = torch.mean(torch.stack(kl_samples), dim=0)  # [B]

        # Clamp for safety (floating point errors can cause -0.00001)
        return torch.clamp(kl_mean, min=0.0)

        # --------------------------------------------------------------------- #
        # Loss Function (Updated Interface)
        # --------------------------------------------------------------------- #
    def vae_loss_full(
            self,
            recon_logits: torch.Tensor,
            x: torch.Tensor,
            mu: torch.Tensor,
            latent_param: torch.Tensor,
            kld_weight: float = 1.0,
            free_bits: float = 0.0,
            correlation_penalty_weight: float = 0.05,  # DEFAULT TO 0.1 TO ENHANCE CORRELATION
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = x.shape[0]

        # 1. Reconstruction Loss
        recon_flat = recon_logits.reshape(batch_size, -1)
        x_flat = x.reshape(batch_size, -1)
        # recon_loss = F.mse_loss(recon_flat, x_flat, reduction="mean")

        recon_normalized = F.mse_loss(recon_flat, x_flat.float() / 255.0, reduction="mean")
        recon_loss = recon_normalized * 4096.0

        if self.training_mode == "ae":
            return recon_loss, recon_loss, torch.zeros_like(recon_loss), torch.zeros_like(recon_loss)

        # 2. KL Divergence (With correlation control)
        # We pass the weight into the KL computer
        kl_per_sample = self.compute_kl_divergence(
            mu,
            latent_param,
            correlation_penalty_weight=correlation_penalty_weight
        )

        # 3. Free Bits Logic
        if free_bits > 0.0:
            kl_loss = F.relu(kl_per_sample - free_bits).mean()
        else:
            kl_loss = kl_per_sample.mean()

        # 4. Total Loss
        total_loss = recon_loss + (kld_weight * kl_loss)

        return total_loss, recon_loss, kl_loss, torch.tensor(0.0, device=x.device)

    # --------------------------------------------------------------------- #
    # Diagnostic / Inference Tools
    # --------------------------------------------------------------------- #
    def get_features(self, x: torch.Tensor, global_stats: Optional[torch.Tensor] = None):
        with torch.no_grad():
            x = x.float()
            byte_features = self.encoder(x)
            if self.global_encoder is not None and global_stats is not None:
                global_vec = self.global_encoder(global_stats)
                features = torch.cat([byte_features, global_vec], dim=-1)
            else:
                features = byte_features
            bottleneck_features = self.bottleneck_fc(features)
            mu = self.fc_mu(bottleneck_features)
            return features, mu

    def compute_mahalanobis_distance(
            self,
            x: torch.Tensor,
            reference_mu: torch.Tensor,
            reference_cov_inv: torch.Tensor,
            global_stats: Optional[torch.Tensor] = None):
        _, mu = self.get_features(x, global_stats)
        centered = mu - reference_mu.unsqueeze(0)
        mahal_squared = torch.sum(centered * torch.matmul(centered, reference_cov_inv), dim=1)
        return torch.sqrt(torch.clamp(mahal_squared, min=0.0) + 1e-6)
