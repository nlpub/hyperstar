__author__ = 'Dmitry Ustalov'

from .baseline import Baseline
from .data import Data
from .frobenius_loss import FrobeniusLoss
from .mlp import MLP
from .regularized_hypernym import RegularizedHypernym
from .regularized_hyponym import RegularizedHyponym
from .regularized_hyponym_phi import RegularizedHyponymPhi
from .regularized_synonym import RegularizedSynonym
from .regularized_synonym_phi import RegularizedSynonymPhi

MODELS = {
    'baseline': Baseline,
    'regularized_hyponym': RegularizedHyponym,
    'regularized_synonym': RegularizedSynonym,
    'regularized_hyponym_phi': RegularizedHyponymPhi,
    'regularized_synonym_phi': RegularizedSynonymPhi,
    'regularized_hypernym': RegularizedHypernym,
    'frobenius_loss': FrobeniusLoss,
    'mlp': MLP
}
