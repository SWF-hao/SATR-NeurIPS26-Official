from .conn_snn import ConnSNN
from .conn_snn_optim import ConnSNN_Optim

from .dense_snn import DenseSNN

from .dense_mlp import DenseMLP
from .dense_gru import DenseGRU
from .dense_lstm import DenseLSTM
from .dense_rnn import DenseRNN


NETWORKS = {
    "ConnSNN": ConnSNN,
    "ConnSNN_Optim": ConnSNN_Optim,

    # Dense
    "DenseSNN": DenseSNN,

    "DenseMLP": DenseMLP,
    "DenseGRU": DenseGRU,
    "DenseLSTM": DenseLSTM,
    "DenseRNN": DenseRNN
}
