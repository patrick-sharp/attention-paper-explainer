from configuration import DEFAULT_CONFIG
from components import Components
from training import train_transformer

cmp = components.Components(DEFAULT_CONFIG)
train_transformer(cmp)
