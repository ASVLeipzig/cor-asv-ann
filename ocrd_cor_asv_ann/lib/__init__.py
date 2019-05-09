'''backend library interface

Sequence2Sequence - encapsulates ANN model definition and application
Node - tree data type for beam search
Alignment - encapsulates global sequence alignment and distance metrics
'''

from .alignment import Alignment
from .seq2seq import Sequence2Sequence, Node, GAP
