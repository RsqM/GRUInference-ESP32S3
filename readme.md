# Micro-Climate Nowcaster: GRU on Bare-Metal ESP32

Target Hardware: Arduino Nano ESP32 (ESP32-S3 N16R8)
Framework: TensorFlow Keras + TFLite Micro (C++)
Model Type: Multivariate Multi-step Regression (GRU)

## 1. Abstract
This project implements a Gated Recurrent Unit (GRU) neural network on an ESP32-S3 microcontroller to predict local weather trends (Temperature, Humidity, Pressure) 60 minutes into the future. Unlike typical IoT weather stations that merely report current conditions, this system performs on-device inferencing to predict near-future micro-climate events (storms, saturation, heat spikes) without internet connectivity.

The system overcomes significant challenges inherent to deploying Recurrent Neural Networks (RNNs) on microcontrollers, specifically addressing dynamic control flow incompatibility in TFLite Micro, tensor arena memory fragmentation, and operator versioning conflicts.

## 2. System Architecture
### 2.1 Neural Network Topology
We utilize a compact GRU-based architecture optimized for embedded execution.

Input Tensor: float32[1, 30, 3]

Batch Size: 1 (Strictly enforced for static allocation).

Sequence Length: 30 steps (representing 30 minutes of history).

Features: 3 (Temperature, Humidity, Pressure).

Hidden Layer: GRU(16 units)

Configuration: return_sequences=False, stateful=False.

Optimization: Static Unrolling (See Section 3.1).

Dense Layer: Dense(180 units)

Mapping: Flattens the hidden state to 60 steps * 3 features.

Output: A flat vector [180] representing the forecast for the next hour.

### 2.2 Hardware Specifications
MCU: Arduino Nano ESP32 (ESP32-S3)

Core: Xtensa LX7 Dual-Core @ 240MHz.

Memory: 16MB Flash, 8MB PSRAM.

FPU: 32-bit Floating Point Unit (critical for GRU tanh/sigmoid ops).

Sensor: Bosch BME280 (I2C)

Precision: ±1 hPa, ±1.0°C, ±3% RH.

Interface: I2C (Address 0x76/0x77).

## 3. Critical Engineering Challenges & Solutions
Deploying RNNs to bare-metal silicon is non-trivial. We encountered and solved three major technical roadblocks.

### 3.1 The "NaN" Crash Loop (Dynamic Control Flow)
Problem: Initial deployment resulted in the ESP32 crashing (Guru Meditation Error) or outputting NaN/Inf values immediately upon inference.

Root Cause: Standard Keras RNN layers utilize a dynamic While loop operation in the computation graph to handle variable sequence lengths. The TFLite Micro interpreter has limited support for dynamic memory allocation during execution. The interpreter often attempted to read uninitialized memory from the Tensor Arena when processing the recurrent loop, leading to corruption.

Solution: Static Unrolling.
We forced the model to unroll the recurrent loop during the graph freeze process:

```python
tf.keras.layers.GRU(16, unroll=True)
```

Technical Impact: This compiles the dynamic loop into a long, linear chain of 30 sequential operations (MatMul -> Add -> Tanh -> MatMul...). While this increases the model binary size (Flash usage), it eliminates dynamic allocation entirely, ensuring 100% runtime stability on the MCU.

### 3.2 Tensor Arena Sizing & Alignment
Problem: The "Unrolled" model requires significantly more scratch memory for intermediate tensors than a standard RNN, as it must store the state of 30 individual gates simultaneously.

Root Cause: Initial arena size estimates (60KB) caused Invoke() failures due to memory overwrites in the scratch buffer.

Solution:

Sizing: We recalculated the arena requirement to 200KB, leveraging the ESP32-S3's generous PSRAM.

Alignment: The ESP32-S3's FPU requires 16-byte aligned memory for vector instructions. We enforced this in C++:

```cpp
alignas(16) uint8_t tensor_arena[kArenaSize];
```

### 3.3 Operator Versioning Hell
Problem: Models trained in TensorFlow 2.16+ export "Version 12" operators (e.g., FULLY_CONNECTED). The TFLite Micro library on Arduino typically supports up to Version 9.

Solution: We implemented a custom "Safety Conversion Pipeline" using Python's tf.function API. This script bypasses the high-level Keras converter defaults and explicitly forces the TFLite converter to emit only standard Builtin Ops (Version 9) compatible with the embedded runtime.

### 3.4 The Edge Impulse "BYOM" Hack
We utilized Edge Impulse for deployment but hit a wall with the platform's "Free Tier" limitations regarding custom model shapes.

The Limitation: The UI rejects models with [180] outputs if the project is configured for standard Regression (which expects 1 output).

The "Trojan Horse" Workflow:

Ingestion: We bypassed CSV upload jitter by uploading JSON data with explicit interval_ms: 1000, forcing the DSP block to accept our 30-step window.

Model Injection: We used the Edge Impulse CLI to force-upload the .tflite file, bypassing the UI validation wizards:

```bash
edge-impulse-upload-model --category regression weather_model.tflite
```

Classification Trick: In some iterations, we categorized the model as a "Classifier" to trick the platform into accepting the 180-float output as "class probabilities," allowing us to access the raw array in C++.

## 4. Firmware Implementation (Forecaster.ino)
The firmware is a bare-metal implementation that bypasses high-level wrappers to ensure direct tensor access.

### 4.1 Circular Buffer
To perform inference every minute based on the last 30 minutes, we implemented a FIFO circular buffer.

Efficiency: O(1) access without memcpy shifting. We maintain a head pointer and calculate the relative index for the model input: (head + i) % 30.

### 4.2 On-Device Normalization
The model expects Z-score normalized inputs. The microcontroller stores the Mean ($\mu$) and Std Dev ($\sigma$) derived from the training set in PROGMEM to perform this transformation in real-time:

$x$ $=$ $x$ - $\mu$ $/$ $\sigma$
 

### 4.3 Heuristic Decision Engine
Raw regression outputs are parsed into actionable alerts:

Storm Logic: if (Pressure_t+60 - Pressure_t0 < -1.0 hPa)

Precipitation Logic: if (Humidity > 85% AND Pressure_slope < 0)

## 5. Hardware Prototyping (3D Printing)
A physical enclosure was engineered to make the device field-ready.

Design: Custom Stevenson Screen (louvered) enclosure designed in Fusion 360.

Function: Blocks direct sunlight and rain while allowing passive airflow to prevent sensor self-heating.

Fabrication: Printed in PETG for UV resistance and thermal stability. (STL files included in repository).

## 6. Repository Contents
/model_training: Jupyter notebooks + convert_safe.py (Schema downgrade script).

/arduino_firmware: Production C++ firmware (Forecaster.ino).

/stl: 3D print files for the Stevenson Screen.

/docs: Detailed logs of the Edge Impulse API integration.

Reference implementation for deploying Recurrent Neural Networks on memory-constrained devices.
