import neural_structured_learning as nsl
import tensorflow_hub as hub
import os
import argparse
import numpy as np
import tensorflow as tf
from get_graph_reg_data import HParams
from base_model import Model

hp = HParams()

def get_GNN_Model():
    base_model = Model()
    gnn_config = nsl.configs.make_graph_reg_config(
    max_neighbors=hp.num_neighbors,
    multiplier=hp.graph_regularization_multiplier,
    distance_type=hp.distance_type,
    sum_over_axis=-1)
    gnn_model = nsl.keras.GraphRegularization(base_model,gnn_config)

    gnn_model.compile(
        optimizer=hp.OPT,
        loss=hp.LOSS,
        metrics=hp.METRICS)

    return gnn_model

