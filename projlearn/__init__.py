from .data import Data
from .baseline                import Baseline
from .regularized_hyponym     import RegularizedHyponym
from .regularized_synonym     import RegularizedSynonym
from .regularized_hyponym_phi import RegularizedHyponymPhi
from .regularized_synonym_phi import RegularizedSynonymPhi
from .regularized_hypernym    import RegularizedHypernym
from .frobenius_loss          import FrobeniusLoss
from .mlp                     import MLP
from .toyota import Toyota

MODELS = {
    'baseline':                Baseline,
    'regularized_hyponym':     RegularizedHyponym,
    'regularized_synonym':     RegularizedSynonym,
    'regularized_hyponym_phi': RegularizedHyponymPhi,
    'regularized_synonym_phi': RegularizedSynonymPhi,
    'regularized_hypernym':    RegularizedHypernym,
    'frobenius_loss':          FrobeniusLoss,
    'mlp':                     MLP,
    'toyota': Toyota
}
