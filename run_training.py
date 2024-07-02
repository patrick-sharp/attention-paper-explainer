from configuration import CONFIG
from components import Components
from training import train_transformer

cmp = components.Components(CONFIG)
train_transformer(cmp)
