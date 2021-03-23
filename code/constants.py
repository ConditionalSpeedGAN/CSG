# DATASET OPTIONS
OBS_LEN = 8  # For single agent obs len is 8 and for multi agent obs len is 20
PRED_LEN = 12  # For single agent pred len is 12 and for multi agent pred len is 30
MULTI_TRAIN_DATASET_PATH = ''
MULTI_VAL_DATASET_PATH = ''
MULTI_TEST_DATASET_PATH = ''

CHECKPOINT_NAME = '' #Test

SINGLE_TRAIN_DATASET_PATH = 'single_condition_dataset/eth/train'
SINGLE_VAL_DATASET_PATH = 'single_condition_dataset/eth/val'
SINGLE_TEST_DATASET_PATH = 'single_condition_dataset/eth/test'

# Activate any one of the following flags
SINGLE_CONDITIONAL_MODEL = True  # For single condition
MULTI_CONDITIONAL_MODEL = False  # For multi condition

# for argoverse
AV_MAX_SPEED = 2.46
OTHER_MAX_SPEED = 1.71
AGENT_MAX_SPEED = 3.33

# for eth/ucy
SINGLE_AGENT_MAX_SPEED = 2.0

# PYTORCH DATA LOADER OPTIONSn
NUM_WORKERS = 4
BATCH_MULTI_CONDITION = 32
BATCH_SINGLE_CONDITION = 16
BATCH_NORM = False
ACTIVATION_RELU = 'relu'
ACTIVATION_LEAKYRELU = 'leakyrelu'
ACTIVATION_SIGMOID = 'sigmoid'

# Time between consecutive frames
FRAMES_PER_SECOND_SINGLE_CONDITION = 0.4  # for eth and ucy
FRAMES_PER_SECOND_MULTI_CONDITION = 0.1  # for argoverse
NORMALIZATION_FACTOR = 10

# HIDDEN DIMENSION OPTIONS FOR SINGLE AND MULTI CONDITION
H_DIM_GENERATOR_MULTI_CONDITION = 32
H_DIM_DISCRIMINATOR_MULTI_CONDITION = 64

H_DIM_GENERATOR_SINGLE_CONDITION = 32
H_DIM_DISCRIMINATOR_SINGLE_CONDITION = 64

MLP_INPUT_DIM_MULTI_CONDITION = 6
MLP_INPUT_DIM_SINGLE_CONDITION = 3

G_LEARNING_RATE, D_LEARNING_RATE = 1e-3, 1e-3
NUM_LAYERS = 1
DROPOUT = 0
NUM_EPOCHS_MULTI_CONDITION = 50
NUM_EPOCHS_SINGLE_CONDITION = 50
CHECKPOINT_EVERY = 100
MLP_DIM = 64
EMBEDDING_DIM = 16
BOTTLENECK_DIM = 32
NOISE_DIM = (8, )

L2_LOSS_WEIGHT = 1

NUM_ITERATIONS = 3200
AGGREGATION_TYPE = 'None'  # the aggregations can be pooling or concat or attention or None
USE_GPU = 0  # use 1 for ETH/UCY and 0 for Argoverse

# SPEED CONTROL FLAGS
TEST_METRIC = 2  # To simulate trajectories, change the flag to 2 and for prediction environment, change the flag to 1.
TRAIN_METRIC = 0  # Used for training the model with the ground truth
VERIFY_OUTPUT_SPEED = 1

# Below flag is set to true if multi condition model on argoverse dataset is set to true.
DIFFERENT_SPEED_MULTI_CONDITION = True
AV_SPEED = 0.2
OTHER_SPEED = 0.9
AGENT_SPEED = 0.5

# Change any one of the below flag to True for Single Condition
STOP_PED_SINGLE_CONDITION = False  # Speed 0 will be imposed if the flag is set to True

CONSTANT_SPEED_SINGLE_CONDITION = True
CS_SINGLE_CONDITION = 0.2  # Constant speed single condition

ANIMATED_VISUALIZATION_CHECK = 0
MAX_CONSIDERED_PED = 5

G_STEPS = 1
D_STEPS = 2
SR_STEPS = 1
BEST_K = 20
PRINT_EVERY = 100
NUM_SAMPLES = 20
NOISE = True
NUM_SAMPLE_CHECK = 100

# Flags used during Single-agent Extrapolation
EXTRAPOLATE_MIN = False
EXTRAPOLATE_MID = False
EXTRAPOLATE_MAX = False