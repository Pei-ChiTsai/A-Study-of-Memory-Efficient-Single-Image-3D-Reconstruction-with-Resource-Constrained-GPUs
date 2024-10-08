import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange, repeat

from ...utils import BaseModule


class Triplane1DTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        plane_size: int
        num_channels: int

    cfg: Config

    def configure(self) -> None:
        """
            Adjust the values of the embedding, with the default set to a random normal distribution.         
        """
        self.embeddings = nn.Parameter(
            #     torch.zeros(
            #             (3, self.cfg.num_channels, self.cfg.plane_size, self.cfg.plane_size),
            #         dtype=torch.float32,
            #     )
            #  * 1
            # / math.sqrt(self.cfg.num_channels)  

            # torch.full(
            #             (3, self.cfg.num_channels, self.cfg.plane_size, self.cfg.plane_size),
            #             1000,
            #         dtype=torch.float32,
            #     )
            #  * 1
            # / math.sqrt(self.cfg.num_channels)  
            
            # num_channel : 1024, plane_size : 32
            torch.randn(
                (3, self.cfg.num_channels, self.cfg.plane_size, self.cfg.plane_size),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(self.cfg.num_channels)  
        )


    def forward(self, batch_size: int) -> torch.Tensor:
        return rearrange(      
            repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size),
            "B Np Ct Hp Wp -> B Ct (Np Hp Wp)",
        )

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, Ct, Nt = tokens.shape
        assert Nt == self.cfg.plane_size**2 * 3 #3072
        assert Ct == self.cfg.num_channels      #40
        return rearrange(
            tokens,
            "B Ct (Np Hp Wp) -> B Np Ct Hp Wp",
            Np=3,
            Hp=self.cfg.plane_size,
            Wp=self.cfg.plane_size,
        )
