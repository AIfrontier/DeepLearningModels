import numpy as np
import tensorflow as tf
from PIL import Image

from model import *
import trainer

##############################################################################
# Model

# This model achieves about 98.6% accuracy after 8 epochs. About 98.0% after only 4 epochs.
model = FeedForwardModel([224, 224, 3], 1000, [Conv2DComponent(3, num_kernels=64),
                                               ActivationComponent(tf.nn.relu),
                                               Conv2DComponent(3, num_kernels=64),
                                               ActivationComponent(tf.nn.relu),
                                               MaxPool2DComponent(2),
                                               Conv2DComponent(3, num_kernels=128),
                                               ActivationComponent(tf.nn.relu),
                                               Conv2DComponent(3, num_kernels=128),
                                               ActivationComponent(tf.nn.relu),
                                               MaxPool2DComponent(2),
                                               Conv2DComponent(3, num_kernels=256),
                                               ActivationComponent(tf.nn.relu),
                                               Conv2DComponent(3, num_kernels=256),
                                               ActivationComponent(tf.nn.relu),
                                               MaxPool2DComponent(2),
                                               Conv2DComponent(3, num_kernels=512),
                                               ActivationComponent(tf.nn.relu),
                                               Conv2DComponent(3, num_kernels=512),
                                               ActivationComponent(tf.nn.relu),
                                               MaxPool2DComponent(2),
                                               Conv2DComponent(3, num_kernels=512),
                                               ActivationComponent(tf.nn.relu),
                                               Conv2DComponent(3, num_kernels=512),
                                               ActivationComponent(tf.nn.relu),
                                               MaxPool2DComponent(2),
                                               FullyConnectedComponent(4096),
                                               ActivationComponent(tf.nn.relu),
                                               FullyConnectedComponent(4096),
                                               ActivationComponent(tf.nn.relu),
                                               FullyConnectedComponent(1000),
                                               ActivationComponent(tf.nn.relu),
                                               FullyConnectedComponent()])
x, out = model.build()

print
print(model) # Displays configured model components
print

learning_rate = 0.01

### Other variables used in training/evaluation
loss_fn = lambda out, y: tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y))
optimizer_fn = lambda loss: tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
accuracy_fn = lambda out, y: tf.reduce_mean(tf.cast(tf.nn.in_top_k(out, y, 1), tf.float32))

##############################################################################
# Data loader and setup
print('Loading images.')
imagenetdata = Image('./imagenet')
training_data, training_labels = imagenetdata.load_training()
validation_data, validation_labels = imagenetdata.load_testing()
print('Training images: {}'.format(len(training_data)))
print('Validation images: {}'.format(len(validation_data)))

print('Reshaping images..')
training_data = np.reshape(training_data, [-1, 28, 28])
validation_data = np.reshape(validation_data, [-1, 28, 28])

##############################################################################
# Training

trainer = trainer.SupervisedTrainer(x, out,
                                    loss_display_interval=100,
                                    accuracy_display_interval=100,
                                    show_accuracy=False,
                                    loss_display_starting_iteration=5) # Loss super high at beginning

trainer.train(training_data,
              training_labels,
              validation_data,
              validation_labels,
              loss_fn=loss_fn,#loss function
              optimizer_fn=optimizer_fn,
              accuracy_fn=accuracy_fn,
              batch_size=100, # Trains in batches of 100 training data points
              validation_set_size=10000, # Evaluates on all 10000 validation data points
              validation_interval=200) # Evaluates validation data every 200 batches
