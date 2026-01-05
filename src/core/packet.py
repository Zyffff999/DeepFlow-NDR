import torch
import torch.nn as nn
import torch.nn.functional as F

def _pick_gn_groups(C: int) -> int:
    # GroupNorm 分组数的简易策略（batch 小/混合精度更稳）
    for g in (16, 8, 4, 2, 1):
        if C % g == 0:
            return g
    return 1


class TCNBlock(nn.Module):
    """
    Depthwise-Conv(因果左填充) + GroupNorm + SiLU
    -> Pointwise 1x1 + GroupNorm + SiLU
    -> 可选 SE 通道注意力
    -> Dropout
    -> 残差(支持 stride/通道对齐) + GELU

    说明：
    - 训练更稳：用 GroupNorm 替代 BN，batch 小/AMP 友好
    - 速度更快：去掉"大核 full conv"，改为 DW+PW 结构
    - 因果：仅对 DW 做左填充，保持时间一致性

    Ablation Parameters:
    - use_dilation: If False, forces dilation=1 (A2-no-dilation)
    - depthwise_only: If True, removes pointwise conv (A2-depthwise-only)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 dropout=0.1, use_se=True, se_reduction=8, causal=False, residual_scale=1.0,
                 use_dilation=False, depthwise_only=False):
        super().__init__()
        self.causal = causal
        self.residual_scale = residual_scale
        self.depthwise_only = depthwise_only

        # ABLATION: A2-no-dilation
        effective_dilation = dilation if use_dilation else 1

        # 因果左填充仅作用于 DW
        pad_left = (kernel_size - 1) * effective_dilation if causal else (kernel_size // 2) * effective_dilation
        self.pad = nn.ConstantPad1d((pad_left, 0), 0) if pad_left > 0 else nn.Identity()

        # Depthwise 1D
        self.dw = nn.Conv1d(in_channels, in_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            dilation=effective_dilation,
                            groups=in_channels,
                            bias=False)
        self.gn1 = nn.GroupNorm(_pick_gn_groups(in_channels), in_channels)
        self.act1 = nn.SiLU(inplace=True)

        # ABLATION: A2-depthwise-only
        # If depthwise_only=True, we skip pointwise and use a simple projection if channels differ
        if not depthwise_only:
            # Pointwise 1x1
            self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.gn2 = nn.GroupNorm(_pick_gn_groups(out_channels), out_channels)
            self.act2 = nn.SiLU(inplace=True)
        else:
            # Simple channel alignment if needed
            if in_channels != out_channels:
                self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
                self.gn2 = nn.GroupNorm(_pick_gn_groups(out_channels), out_channels)
            else:
                self.pw = None
                self.gn2 = None
            self.act2 = None

        # SE 通道注意力（轻量可选）
        self.use_se = use_se
        if use_se:
            hidden = max(1, out_channels // se_reduction)
            self.se_fc1 = nn.Conv1d(out_channels, hidden, kernel_size=1)
            self.se_fc2 = nn.Conv1d(hidden, out_channels, kernel_size=1)

        self.drop = nn.Dropout(dropout)

        # 残差对齐：通道或步幅不一致时 1x1
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        self.final_act = nn.GELU()

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        y = self.pad(x)
        y = self.dw(y)
        y = self.gn1(y)
        y = self.act1(y)

        # Apply pointwise if not depthwise_only
        if not self.depthwise_only:
            y = self.pw(y)
            y = self.gn2(y)
            y = self.act2(y)
        else:
            # Only apply projection if channels differ
            if self.pw is not None:
                y = self.pw(y)
                y = self.gn2(y)

        if self.use_se:
            s = F.adaptive_avg_pool1d(y, 1)
            s = self.se_fc2(F.silu(self.se_fc1(s)))
            y = y * torch.sigmoid(s)

        y = self.drop(y)
        out = residual + self.residual_scale * y
        return self.final_act(out)


class PacketEncoder(nn.Module):
    """
    Ablation Parameters:
    - header_only: If True, only uses header bytes and ignores payload (A1-header-only)
    - use_dilation: If False, all TCN blocks use dilation=1 (A2-no-dilation)
    - depthwise_only: If True, removes pointwise convs from TCN blocks (A2-depthwise-only)
    """

    def __init__(self, vocab_size=257, emb_dim=128, latent_dim=64,
                 header_tcn_channels=[192, 156, 128],
                 payload_tcn_channels=[192, 156, 128],
                 spatial_reduction_size=64,
                 header_len=None,
                 header_only=True,
                 use_dilation=False,
                 depthwise_only=False):
        super(PacketEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.spatial_reduction_size = spatial_reduction_size
        self.header_len = header_len

        # ABLATION FLAGS
        self.header_only = header_only
        self.use_dilation = use_dilation
        self.depthwise_only = depthwise_only

        self.header_tcn = self._build_tcn_stack(
            in_channels=emb_dim,
            channel_list=header_tcn_channels,
            strides=[2] + [1] * (len(header_tcn_channels) - 1),
            causal=True
        )

        # ABLATION: A1-header-only
        # Only build payload TCN if not header_only
        if not header_only:
            self.payload_tcn = self._build_tcn_stack(
                in_channels=emb_dim,
                channel_list=payload_tcn_channels,
                strides=[2] + [1] * (len(payload_tcn_channels) - 1),
                causal=True
            )
        else:
            self.payload_tcn = None

        C = header_tcn_channels[-1]

        # ABLATION: A1-header-only
        # Only create cross-domain components if using both header and payload
        if not header_only:
            self.cross_gate = nn.Sequential(
                nn.LayerNorm(C * 2),
                nn.Linear(C * 2, C),
                nn.SiLU(),
                nn.Linear(C, C * 2),
                nn.Sigmoid()
            )

            # 双向 cross attention（序列长度缩短后会更快）
            self.cross_attn_header_to_payload = nn.MultiheadAttention(
                embed_dim=C, num_heads=4, batch_first=True, dropout=0.1
            )
            self.cross_attn_payload_to_header = nn.MultiheadAttention(
                embed_dim=C, num_heads=4, batch_first=True, dropout=0.1
            )

            fused_dim = (C * spatial_reduction_size) * 2
        else:
            self.cross_gate = None
            self.cross_attn_header_to_payload = None
            self.cross_attn_payload_to_header = None
            fused_dim = C * spatial_reduction_size

        self.fc_out = nn.Linear(fused_dim, latent_dim, bias=False)
        self.output_norm = nn.LayerNorm(latent_dim)

    def _build_tcn_stack(self, in_channels, channel_list, strides=None, causal=True):
        layers = []
        if strides is None:
            strides = [1] * len(channel_list)
        for i, out_ch in enumerate(channel_list):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else channel_list[i - 1]
            layers.append(
                TCNBlock(in_ch, out_ch, kernel_size=3, stride=strides[i], dilation=dilation,
                         dropout=0.1, use_se=True, causal=causal,
                         use_dilation=self.use_dilation,
                         depthwise_only=self.depthwise_only)
            )
        return nn.Sequential(*layers)

    def forward(self, packets):
        # packets: [B, packet_len]
        packets = packets.long()
        header_len = self.header_len
        if self.header_only:
            if header_len is None:
                header_bytes = packets
            else:
                header_bytes = packets[:, :header_len]
        else:
            if header_len is None:
                header_len = packets.shape[1] // 2
            header_bytes = packets[:, :header_len]

        # [B,L] -> [B,C,L]
        header = self.embedding(header_bytes).permute(0, 2, 1)

        # TCN 提取后：序列长度已因 stride=2 从 128 -> 64
        header_feat = self.header_tcn(header)  # [B, C, Lh=~64]

        # ABLATION: A1-header-only
        if self.header_only:
            # Only use header features
            h_red = F.adaptive_avg_pool1d(header_feat, self.spatial_reduction_size)  # [B,C,S]
            h_vec = h_red.flatten(1)  # [B, C*S]
            z = self.output_norm(self.fc_out(h_vec))  # [B, latent_dim]
            return z

        # Original dual-path processing
        payload_bytes = packets[:, header_len:]
        payload = self.embedding(payload_bytes).permute(0, 2, 1)
        payload_feat = self.payload_tcn(payload)  # [B, C, Lp=~64]

        # -------- 跨域门控（在注意力前做通道重标定）--------
        ch = header_feat.mean(dim=2)  # [B,C]
        cp = payload_feat.mean(dim=2)  # [B,C]
        gates = self.cross_gate(torch.cat([ch, cp], dim=1))  # [B, 2C]
        gh, gp = gates.chunk(2, dim=1)  # 两个 [B,C]
        header_feat = header_feat * gh.unsqueeze(-1)
        payload_feat = payload_feat * gp.unsqueeze(-1)

        # -------- 双向 cross-attn --------
        # conv: [B,C,L] -> attn: [B,L,C]
        h_q = header_feat.permute(0, 2, 1)
        p_kv = payload_feat.permute(0, 2, 1)
        h_out, _ = self.cross_attn_header_to_payload(query=h_q, key=p_kv, value=p_kv)

        p_q = payload_feat.permute(0, 2, 1)
        h_kv = header_feat.permute(0, 2, 1)
        p_out, _ = self.cross_attn_payload_to_header(query=p_q, key=h_kv, value=h_kv)

        # -------- 空间压缩 + 融合 --------
        h_attn = h_out.permute(0, 2, 1)  # [B,C,L]
        p_attn = p_out.permute(0, 2, 1)  # [B,C,L]

        h_red = F.adaptive_avg_pool1d(h_attn, self.spatial_reduction_size)  # [B,C,S]
        p_red = F.adaptive_avg_pool1d(p_attn, self.spatial_reduction_size)  # [B,C,S]

        h_vec = h_red.flatten(1)  # [B, C*S]
        p_vec = p_red.flatten(1)  # [B, C*S]

        fused = torch.cat([h_vec, p_vec], dim=1)
        z = self.output_norm(self.fc_out(fused))  # [B, latent_dim]
        return z


class SessionEncoder(nn.Module):
    """
    Ablation Parameters:
    - header_only: If True, only uses header bytes (A1-header-only)
    - use_dilation: If False, all TCN blocks use dilation=1 (A2-no-dilation)
    - depthwise_only: If True, removes pointwise convs from TCN blocks (A2-depthwise-only)
    """

    def __init__(
        self,
        packet_latent_dim=256,
        d_model=256,
        latent_dim=32,
        spatial_reduction_size=8,
        header_len=None,
        vocab_size=257,
        emb_dim=128,
        header_tcn_channels=None,
        payload_tcn_channels=None,
        session_tcn_channels=None,
        header_only=False,
        use_dilation=True,
        depthwise_only=False,
    ):
        super(SessionEncoder, self).__init__()

        # ABLATION FLAGS
        self.header_only = header_only
        self.use_dilation = use_dilation
        self.depthwise_only = depthwise_only

        if header_tcn_channels is None:
            header_tcn_channels = [192, 156, 128]
        if payload_tcn_channels is None:
            payload_tcn_channels = [192, 156, 128]
        if session_tcn_channels is None:
            session_tcn_channels = [128, 64, latent_dim]
        if d_model != packet_latent_dim:
            # Force match to keep session TCN input consistent with packet encoder output.
            d_model = packet_latent_dim

        self.packet_encoder = PacketEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            latent_dim=packet_latent_dim,
            header_tcn_channels=header_tcn_channels,
            payload_tcn_channels=payload_tcn_channels,
            header_len=header_len,
            header_only=header_only,
            use_dilation=use_dilation,
            depthwise_only=depthwise_only,
            spatial_reduction_size=spatial_reduction_size,
        )

        self.final_C = session_tcn_channels[-1]
        self.avgmax_proj = nn.Linear(self.final_C * 2, self.final_C, bias=False)
        self.session_tcn = self._build_tcn_stack(d_model, session_tcn_channels)
        self.final_norm = nn.LayerNorm(self.final_C)

    def _build_tcn_stack(self, in_channels, channel_list):
        layers = []
        for i, out_ch in enumerate(channel_list):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else channel_list[i - 1]
            layers.append(
                TCNBlock(in_ch, out_ch, kernel_size=3, stride=1, dilation=dilation,
                         dropout=0.1, use_se=True, causal=True,
                         use_dilation=self.use_dilation,
                         depthwise_only=self.depthwise_only)
            )
        return nn.Sequential(*layers)

    def forward(self, sessions):
        # sessions: [B, num_packets, packet_len]
        B, N, L = sessions.shape
        packets_flat = sessions.reshape(-1, L)  # [B*N, packet_len]
        z_flat = self.packet_encoder(packets_flat)  # [B*N, d_model]

        z_seq = z_flat.view(B, N, -1).permute(0, 2, 1)  # [B, d_model, num_packets]
        session_feat = self.session_tcn(z_seq)  # [B, C, num_packets]
        session_vec = session_feat.mean(dim=2)  # [B, 64]

        return self.final_norm(session_vec)


class ResidualBlock1(nn.Module):
    def __init__(self, hidden_dim, dropout_p=0.15):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class FactorizedDecoder(nn.Module):
    def __init__(self, latent_dim=96, output_size=16 * 256, output_shape=None, rank=256, num_deep_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        if output_shape is None:
            if output_size == 16 * 256:
                output_shape = (16, 256)
            else:
                output_shape = (1, int(output_size))
        self.output_shape = output_shape
        self.output_size = int(self.output_shape[0] * self.output_shape[1])
        self.rank = rank

        self.U = nn.Linear(latent_dim, rank, bias=False)
        self.deep_layers = nn.ModuleList([ResidualBlock1(rank) for _ in range(num_deep_layers)])
        self.final_norm = nn.LayerNorm(rank)
        self.V = nn.Linear(rank, output_size, bias=True)

    def forward(self, z):
        B = z.size(0)
        h = self.U(z)
        for layer in self.deep_layers:
            h = layer(h)
        h = self.final_norm(h)
        out = self.V(h).view(B, self.output_shape[0], self.output_shape[1])
        return out


# ============================================================================
# ABLATION STUDY CONFIGURATION
# ============================================================================

class AblationConfig:
    """
    Simple configuration class for ablation studies.

    Usage:
        config = AblationConfig(
            header_only=True,      # A1-header-only
            use_dilation=False,    # A2-no-dilation
            depthwise_only=True    # A2-depthwise-only
        )

        model = create_model(config)
    """

    def __init__(self,
                 header_only=False,
                 use_dilation=True,
                 depthwise_only=False):
        self.header_only = header_only
        self.use_dilation = use_dilation
        self.depthwise_only = depthwise_only

    def __repr__(self):
        return (f"AblationConfig(\n"
                f"  A1-header-only: {self.header_only}\n"
                f"  A2-no-dilation: {not self.use_dilation}\n"
                f"  A2-depthwise-only: {self.depthwise_only}\n"
                f")")

    def get_name(self):
        """Generate a descriptive name for logging/saving models"""
        parts = []
        if self.header_only:
            parts.append("header-only")
        if not self.use_dilation:
            parts.append("no-dilation")
        if self.depthwise_only:
            parts.append("depthwise-only")

        if not parts:
            return "baseline"
        return "_".join(parts)


def create_model(config, packet_latent_dim=128, d_model=128, latent_dim=32,
                 decoder_rank=256, decoder_layers=1):
    """
    Factory function to create encoder-decoder pair with ablation settings.

    Args:
        config: AblationConfig instance
        packet_latent_dim: Packet encoder output dimension
        d_model: Session encoder model dimension
        latent_dim: Final latent dimension
        decoder_rank: Decoder intermediate dimension
        decoder_layers: Number of decoder residual blocks

    Returns:
        encoder, decoder: Tuple of models
    """
    encoder = SessionEncoder(
        packet_latent_dim=packet_latent_dim,
        d_model=d_model,
        latent_dim=latent_dim,
        header_only=config.header_only,
        use_dilation=config.use_dilation,
        depthwise_only=config.depthwise_only
    )

    decoder = FactorizedDecoder(
        latent_dim=latent_dim,
        output_size=16 * 256,
        rank=decoder_rank,
        num_deep_layers=decoder_layers
    )

    return encoder, decoder


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    def count_params(m):
        return sum(p.numel() for p in m.parameters())


    print("=" * 80)
    print("ABLATION STUDY DEMO")
    print("=" * 80)

    # Define all ablation configurations to test
    configs = [
        AblationConfig(header_only=False, use_dilation=True, depthwise_only=False),  # Baseline
        AblationConfig(header_only=True, use_dilation=True, depthwise_only=False),  # A1
        AblationConfig(header_only=False, use_dilation=False, depthwise_only=False),  # A2-no-dilation
        AblationConfig(header_only=False, use_dilation=True, depthwise_only=True),  # A2-depthwise
        # You can also combine ablations:
        # AblationConfig(header_only=True, use_dilation=False, depthwise_only=False),
    ]

    # Test input
    dummy_sessions = torch.randint(0, 256, (2, 16, 256))

    print("\nTesting all configurations:\n")

    for config in configs:
        print("-" * 80)
        print(f"Configuration: {config.get_name()}")
        print(config)

        # Create models
        enc, dec = create_model(config)

        # Count parameters
        enc_params = count_params(enc)
        dec_params = count_params(dec)

        print(f"Encoder params: {enc_params:,}")
        print(f"Decoder params: {dec_params:,}")
        print(f"Total params:   {enc_params + dec_params:,}")

        # Test forward pass
        try:
            z = enc(dummy_sessions)
            recon = dec(z)
            print(f"✓ Forward pass successful")
            print(f"  Latent shape: {z.shape}")
            print(f"  Output shape: {recon.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")

        print()

    print("=" * 80)
    print("SINGLE MODEL EXAMPLE")
    print("=" * 80)

    # Example: Create a specific ablation model
    my_config = AblationConfig(
        header_only=True,  # Enable A1-header-only
        use_dilation=False,  # Enable A2-no-dilation
        depthwise_only=False  # Disable A2-depthwise-only
    )

    print(f"\nCreating model with configuration: {my_config.get_name()}")
    print(my_config)

    encoder, decoder = create_model(my_config)

    # Test
    z = encoder(dummy_sessions)
    output = decoder(z)

    print(f"\nModel created successfully!")
    print(f"Input shape:  {dummy_sessions.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total params: {count_params(encoder) + count_params(decoder):,}")
