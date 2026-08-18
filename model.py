import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


FastLayerNorm2d = LayerNorm2d


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FastDWT_Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("FastDWT_Down requires even image height and width.")

        x_unshuffled = self.unshuffle(x)
        batch, channels_times_four, height, width = x_unshuffled.shape
        channels = channels_times_four // 4
        x1, x2, x3, x4 = x_unshuffled.reshape(batch, channels, 4, height, width).unbind(dim=2)

        ll = (x1 + x2 + x3 + x4) * 0.5
        hl = (-x1 + x2 - x3 + x4) * 0.5
        lh = (-x1 - x2 + x3 + x4) * 0.5
        hh = (x1 - x2 - x3 + x4) * 0.5

        # Stack, rather than concatenate, so each input channel owns four
        # adjacent sub-band channels: [LL, HL, LH, HH].
        return torch.stack((ll, hl, lh, hh), dim=2).reshape(
            batch, channels_times_four, height, width
        )


class FastIDWT_Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels_times_four, height, width = x.shape
        if channels_times_four % 4:
            raise ValueError("FastIDWT_Up requires a channel count divisible by four.")

        channels = channels_times_four // 4
        ll, hl, lh, hh = x.reshape(batch, channels, 4, height, width).unbind(dim=2)

        x1 = (ll - hl - lh + hh) * 0.5
        x2 = (ll + hl - lh - hh) * 0.5
        x3 = (ll - hl + lh - hh) * 0.5
        x4 = (ll + hl + lh + hh) * 0.5

        coefficients = torch.stack((x1, x2, x3, x4), dim=2).reshape(
            batch, channels_times_four, height, width
        )
        return self.shuffle(coefficients)


class DWTDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dwt = FastDWT_Down()
        self.reduce = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.dwt(x))


class IDWTUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.expand = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False)
        self.idwt = FastIDWT_Up()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.idwt(self.expand(x))


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        if dw_channels % 2 or ffn_channels % 2:
            raise ValueError("NAFBlock expansion widths must be even for SimpleGate.")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

        # NAFNet residual scaling: blocks start as identity mappings.
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        residual = inp + x * self.beta

        x = self.norm2(residual)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return residual + x * self.gamma


class NAFNetDWT(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_dim: int = 16,
        enc_blks=(1, 1, 1, 2),
        middle_blk: int = 2,
        dec_blks=(1, 1, 1, 1),
        upscale: int = 2,
    ):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("Residual learning requires in_channels to equal out_channels.")
        if upscale < 1 or int(upscale) != upscale:
            raise ValueError("upscale must be a positive integer.")
        if len(enc_blks) != len(dec_blks):
            raise ValueError("enc_blks and dec_blks must have the same number of stages.")

        self.upscale = int(upscale)
        self.required_multiple = 2 ** len(enc_blks)
        self.intro = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = base_dim
        for num_blocks in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)]))
            self.downsamples.append(DWTDownsample(channels, channels * 2))
            channels *= 2

        self.middle = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blk)])

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num_blocks in dec_blks:
            self.upsamples.append(IDWTUpsample(channels, channels // 2))
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)]))

        if self.upscale > 1:
            self.sr_up = nn.Sequential(
                nn.Conv2d(base_dim, base_dim * self.upscale**2, kernel_size=3, padding=1),
                nn.PixelShuffle(self.upscale),
            )
        else:
            self.sr_up = nn.Identity()

        self.ending = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)
        # Start as an exact bilinear-upsample baseline, not random residuals.
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def _pad_to_valid_size(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.required_multiple
        pad_w = (-width) % self.required_multiple
        if pad_h == 0 and pad_w == 0:
            return x

        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.ndim != 4:
            raise ValueError("Expected an NCHW tensor with shape [batch, channels, height, width].")
        if inp.shape[1] != self.intro.in_channels:
            raise ValueError(
                f"Expected {self.intro.in_channels} input channels, got {inp.shape[1]}."
            )

        original_height, original_width = inp.shape[-2:]
        baseline = F.interpolate(inp, scale_factor=self.upscale, mode="bilinear", align_corners=False)

        x = self._pad_to_valid_size(inp)
        x = self.intro(x)
        skips = []
        for encoder, downsample in zip(self.encoders, self.downsamples):
            x = encoder(x)
            skips.append(x)
            x = downsample(x)

        x = self.middle(x)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            x = decoder(x + skip)

        x = self.sr_up(x)
        residual = self.ending(x)
        residual = residual[..., : original_height * self.upscale, : original_width * self.upscale]
        return baseline + residual
