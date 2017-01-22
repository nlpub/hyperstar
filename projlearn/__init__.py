from .data import Data
from .baseline             import Baseline
from .regularized_hyponym  import RegularizedHyponym
from .regularized_synonym  import RegularizedSynonym
from .regularized_hypernym import RegularizedHypernym
from .frobenius_loss       import FrobeniusLoss
from .mlp                  import MLP

MODELS = {
    'baseline':              Baseline,
    'regularized_hyponym':   RegularizedHyponym,
    'regularized_synonym':   RegularizedSynonym,
    'regularized_hypernym':  RegularizedHypernym,
    'frobenius_loss':        FrobeniusLoss,
    'mlp':                   MLP
}
