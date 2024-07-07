from enum import Enum, global_enum


@global_enum
class ComponentType(Enum):
    TRAIN_RAW = 0
    TOKENIZER = 1
    TRAIN_TOKENIZED = 2
    TRAIN_BATCHED = 3
    MODEL_TRAIN_STATE = 4
    TEST_RAW = 5
    TEST_TOKENIZED = 6
    TEST_BATCHED = 7


component_dependencies = {
    TRAIN_RAW: [],
    TOKENIZER: [TRAIN_RAW],
    TRAIN_TOKENIZED: [TRAIN_RAW, TOKENIZER],
    TRAIN_BATCHED: [TRAIN_TOKENIZED],
    MODEL_TRAIN_STATE: [],
    TEST_RAW: [],
    TEST_TOKENIZED: [TEST_RAW, TOKENIZER],
    TEST_BATCHED: [TEST_TOKENIZED],
}
