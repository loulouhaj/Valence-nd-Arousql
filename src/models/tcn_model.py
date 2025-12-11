import tensorflow as tf
from tensorflow.keras import layers, models
from config import config

class TCNBlock(layers.Layer):
    def __init__(self, filters: int, kernel_size: int, dilation_rate: int, dropout_rate: float, name: str = None):
        super(TCNBlock, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
        # Branch 1
        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            name=f"{name}_conv1"
        )
        self.relu1 = layers.Activation('relu', name=f"{name}_relu1")
        self.dropout1 = layers.Dropout(dropout_rate, name=f"{name}_dropout1")
        
        # Branch 2
        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            name=f"{name}_conv2"
        )
        self.relu2 = layers.Activation('relu', name=f"{name}_relu2")
        self.dropout2 = layers.Dropout(dropout_rate, name=f"{name}_dropout2")
        
        # 1x1 convolution for residual connection if dimensions mismatch
        self.downsample = layers.Conv1D(filters=filters, kernel_size=1, padding='same', name=f"{name}_downsample")
        
        self.add = layers.Add(name=f"{name}_add")
        self.final_relu = layers.Activation('relu', name=f"{name}_final_relu")

    def call(self, inputs, training=False):
        residual = inputs
        
        # TODO: Check if this structure matches strict requirements for all use cases
        
        # Branch 1(x)
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)
        
        # Flow: Conv1D -> ReLU -> Dropout
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)
        
        # Residual connection
        if inputs.shape[-1] != self.filters:
            residual = self.downsample(inputs)
            
        x = self.add([x, residual])
        x = self.final_relu(x)
        return x

    def get_config(self):
        config = super(TCNBlock, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout_rate": self.dropout_rate,
        })
        return config

class TCNModel(models.Model):
    def __init__(self, input_shape: tuple):
        super(TCNModel, self).__init__()
        self.tcn_config = config
        
        self.tcn_blocks = []
        num_stacks = config.NB_STACKS
        dilations = config.DILATIONS # Should be (1, 1, 1) per user request
        
        # TODO: Ensure dilation logic is robust if config.DILATIONS changes
        
        current_dilation_index = 0
        for stack in range(num_stacks):
             # TODO: Validate dilation cycling logic
             d = dilations[stack] if stack < len(dilations) else 1
             
             self.tcn_blocks.append(
                TCNBlock(
                    filters=config.NB_FILTERS,
                    kernel_size=config.KERNEL_SIZE,
                    dilation_rate=d,
                    dropout_rate=config.DROPOUT_RATE,
                    name=f"tcn_block_{stack}"
                )
             )
        
        # Final Layer: Conv1D(kernel_size=1, filters=2)
        # Outputs a pair (valence, arousal) for each time step.
        self.final_conv = layers.Conv1D(
            filters=config.OUTPUT_DIM, 
            kernel_size=1, 
            activation='tanh', 
            name="final_output"
        ) 

    def call(self, inputs, training=False):
        x = inputs
        
        for block in self.tcn_blocks:
            x = block(x, training=training)
            
        output = self.final_conv(x)
        
        return output

    def get_config(self):
        config = super(TCNModel, self).get_config()
        # TODO: Add any other necessary config parameters
        return config

def build_tcn_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    model = TCNModel(input_shape)
    outputs = model(inputs)
    return models.Model(inputs=inputs, outputs=outputs, name="TCN_Model")

if __name__ == "__main__":
    # Test model build
    dummy_input_shape = (config.SEQUENCE_LENGTH, 260) 
    model = build_tcn_model(dummy_input_shape)
    model.summary()
