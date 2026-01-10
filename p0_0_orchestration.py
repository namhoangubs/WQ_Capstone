from p0_1_raw_to_rets import p0_1_raw_to_rets
from p1_0_hmm_params import p1_0_hmm_params
from p1_1_best_params import p1_1_best_params
from p1_2_hmm import p1_2_hmm
from itertools import product
from os import listdir, remove
from p2_2_wgan import p2_2_wgan
from p3_0_sims import p3_0_sims
from p3_1_stat_prop import p3_1_stat_prop
from p3_2_stat_prop_mv import p3_2_stat_prop_mv
from p3_3_risk_man import p3_3_risk_man

from pathlib import Path

BASE_DIR = Path(r"C:\Users\namho\Desktop\Previous PC\World Quant\Thesis\Replication")
INPUT_DIR = BASE_DIR / "Inputs"

import os, sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

p0_1_raw_to_rets()
p1_0_hmm_params()
p1_1_best_params()
p1_2_hmm()

for n_states, n_stocks in product([1, 2, 4], [8, 16, 32, 64]):
    if n_states != 1 and n_stocks == 8:
        [remove(str(BASE_DIR / 'models' / file)) for file in listdir(BASE_DIR / 'models')]
    p2_2_wgan(n_states, n_stocks)
    p3_0_sims(n_states, n_stocks)
    p3_1_stat_prop(n_states, n_stocks)
    p3_2_stat_prop_mv(n_states, n_stocks)
    p3_3_risk_man(n_states, n_stocks)