import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
import pickle


# Get dataset
history_u_lists, history_ur_lists, history_v_lists, \
    history_vr_lists, train_u, train_v, train_r, test_u, \
    test_v, test_r, social_adj_lists, ratings_list = pickle.load(
        open('toy_dataset.pickle', 'rb'))
NUMBER_OF_USERS = len(history_u_lists)
NUMBER_OF_ITEMS = len(history_v_lists)
NUMBER_OF_RATINGS = len(ratings_list)

# Generate required lists
user_interaction = []
user_item = []
user_rating = []
for user in range(NUMBER_OF_USERS):
    user_interaction.append(list(social_adj_lists[user]))
    user_item.append(history_u_lists[user])
    user_rating.append(history_ur_lists[user])

user_interaction_encoded = np.zeros((NUMBER_OF_USERS, 2000))
user_item_encoded = np.zeros((NUMBER_OF_USERS, 2000))
user_rating_encoded = np.zeros((NUMBER_OF_USERS, 2000))

for user, interactions in enumerate(user_interaction):
    user_interaction_encoded[user, interactions] = 1
for user, items in enumerate(user_item):
    user_item_encoded[user, items] = 1
for user, ratings in enumerate(user_rating):
    user_rating_encoded[user, ratings] = 1

# Inputs
user_interaction_input = tf.keras.layers.Input(
    shape=(None,))
user_item_input = tf.keras.layers.Input(
    shape=(None,))
user_rating_input = tf.keras.layers.Input(
    shape=(None,))

# Embeddings
EMBEDDING_DIMENSION = 64
user_item_embedding = tf.keras.layers.Embedding(
    NUMBER_OF_ITEMS, EMBEDDING_DIMENSION)(user_item_input)
user_opinion_embedding = tf.keras.layers.Embedding(
    NUMBER_OF_RATINGS, EMBEDDING_DIMENSION)(user_rating_input)
user_interaction_embedding = tf.keras.layers.Embedding(
    NUMBER_OF_USERS, EMBEDDING_DIMENSION)(user_interaction_input)

"""
Item aggregation:
1 - Calculate OpinionAwareRepresentation*AttentionWeight
2 - Pass the calculated value to a fully connected layer
"""
# OpinionAwareRepresentation
opinion_aware_representation = tf.keras.layers.Concatenate()(
    [user_item_embedding, user_opinion_embedding])
opinion_aware_representation = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(opinion_aware_representation)
# AttentionWeight
item_aggregation_attention = tf.keras.layers.Concatenate(
)([user_interaction_embedding, opinion_aware_representation])
item_aggregation_attention = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(item_aggregation_attention)
item_aggregation_attention = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(item_aggregation_attention)
item_aggregation_attention = tf.keras.layers.Dense(
    1, activation='softmax')(item_aggregation_attention)
# OpinionAwareRepresentation*AttentionWeight
aggregate_items = tf.keras.layers.Multiply()([
    item_aggregation_attention,
    opinion_aware_representation
])
# Fully connected layer that generates item-space
item_space = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(aggregate_items)
# Constructing the item aggregation model
item_aggregation_model = tf.keras.Model(
    inputs=[
        user_item_input,
        user_rating_input,
        user_interaction_input
    ],
    outputs=item_space
)


# Social aggregation
# Create a custom to get only the neighbor nodes in the future layers

# Social aggregation attention weights
social_aggregation_attention = tf.keras.layers.Concatenate()([
    item_space,
    user_interaction_embedding
])
social_aggregation_attention = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(social_aggregation_attention)
social_aggregation_attention = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(social_aggregation_attention)
social_aggregation_attention = tf.keras.layers.Dense(
    1, activation='softmax')(social_aggregation_attention)

# Social Space
social_space = tf.keras.layers.Multiply()([
    social_aggregation_attention,
    item_space,
])

# # @tf.function
# def get_neighbors(x):
#     neighbors = np.zeros((NUMBER_OF_USERS, EMBEDDING_DIMENSION))
#     for batch in range()

# neighbors = tf.keras.layers.Lambda(
#     get_neighbors)(user_interaction_input)


# # Neighbors' item-space
# item_space_transpose = tf.keras.layers.Permute((3, 1))(item_space)
# neighbor_item_space = tf.keras.layers.Dot(
#     axes=1)([neighbors, item_space_transpose])

# # Social aggregation attention weights
# # Neighbors' user embeddings
# user_interaction_embedding_transpose = tf.keras.layers.Permute(
#     (3, 1))(user_interaction_embedding)
# neighbor_user_interaction_embedding = tf.keras.layers.Dot(
#     axes=1)([neighbors, user_interaction_embedding])

# # Concatenate neighbors' item-space and neighbors' user embedding
# social_aggregation_attention = tf.keras.layers.Concatenate()(
#     [neighbor_user_interaction_embedding, neighbor_item_space])
# social_aggregation_attention = tf.keras.layers.Dense(
#     EMBEDDING_DIMENSION, activation='relu')(social_aggregation_attention)
# social_aggregation_attention = tf.keras.layers.Dense(
#     EMBEDDING_DIMENSION, activation='relu')(social_aggregation_attention)
# social_aggregation_attention = tf.keras.layers.Dense(
#     1, activation='softmax')(social_aggregation_attention)

# # Attention * NeighborItemSpace
# social_aggregate = tf.keras.layers.Multiply()(
#     [social_aggregation_attention, neighbor_item_space])
# # Social space
# social_space = tf.keras.layers.Dense(
#     EMBEDDING_DIMENSION, activation='relu')(social_aggregate)

# # Constructing the social aggregation model
# social_aggregation_model = tf.keras.Model(
#     inputs=[
#         user_item_input,
#         user_rating_input,
#         user_interaction_input
#     ],
#     outputs=social_space
# )

# User Modeling
user_latent_factor = tf.keras.layers.Concatenate()([item_space, social_space])
user_latent_factor = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(user_latent_factor)
user_latent_factor = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(user_latent_factor)
user_latent_factor = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(user_latent_factor)

# Constructing the model of user modeling module
user_modeling_model = tf.keras.Model(
    inputs=[
        user_item_input,
        user_rating_input,
        user_interaction_input
    ],
    outputs=user_latent_factor
)

# Item modeling
item_opinion_aware_interaction = tf.keras.layers.Concatenate()(
    [user_interaction_embedding, user_opinion_embedding])
item_opinion_aware_interaction = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(item_opinion_aware_interaction)

# User aggregation attention
user_aggregation_attention = tf.keras.layers.Concatenate()(
    [item_opinion_aware_interaction, user_item_embedding])
user_aggregation_attention = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(user_aggregation_attention)
user_aggregation_attention = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(user_aggregation_attention)
user_aggregation_attention = tf.keras.layers.Dense(
    1, activation='softmax')(user_aggregation_attention)

# Item latent factor
item_latent_factor = tf.keras.layers.Multiply()([
    user_aggregation_attention,
    item_opinion_aware_interaction,
])
item_latent_factor = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(item_latent_factor)
item_latent_factor = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(item_latent_factor)
item_latent_factor = tf.keras.layers.Dense(
    EMBEDDING_DIMENSION, activation='relu')(item_latent_factor)

# Constructing the model of item modeling module
item_modeling_model = tf.keras.Model(
    inputs=[
        user_item_input,
        user_rating_input,
        user_interaction_input
    ],
    outputs=item_latent_factor
)

# Rating prediction
output = tf.keras.layers.Concatenate()(
    [item_latent_factor, user_latent_factor])
output = tf.keras.layers.Dense(EMBEDDING_DIMENSION, activation='relu')(output)
output = tf.keras.layers.Dense(EMBEDDING_DIMENSION, activation='relu')(output)
output = tf.keras.layers.Dense(
    user_rating_encoded.shape[1], activation='relu')(output)

# GraphRec model
graphrec = tf.keras.Model(
    inputs=[
        user_item_input,
        user_rating_input,
        user_interaction_input
    ],
    outputs=output
)
plot_model(graphrec, to_file='model.png', show_shapes=True)
graphrec.compile(
    optimizer='rmsprop',
    loss='mse',
    metrics=['accuracy']
)


graphrec.fit(
    x=[
        user_item_encoded,
        user_rating_encoded,
        user_interaction_encoded
    ],
    y=user_rating_encoded,
    batch_size=1,
    epochs=100
)
