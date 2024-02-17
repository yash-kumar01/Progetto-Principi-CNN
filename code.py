import tensorflow as tf

# Define the Neocognitron-like CNN model
def neocognitron_model(input_shape, num_classes):
    model = tf.keras.Sequential()
    
    # Convolutional Layer 1
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Convolutional Layer 2
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the feature maps
    model.add(tf.keras.layers.Flatten())
    
    # Fully Connected Layers
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    return model

# Define the input shape and number of classes
input_shape = (32, 32, 3)  # Example input shape for RGB images of size 32x32
num_classes = 10  # Example number of classes (e.g., for CIFAR-10 dataset)

# Create an instance of the Neocognitron-like CNN model
model = neocognitron_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model. Summary()
