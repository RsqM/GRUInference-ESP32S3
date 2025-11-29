#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

// 1. Include MicroTFLite Library
#include <MicroTFLite.h>
#include "model_new.h"

// ================= CONFIGURATION =================
// Normalization constants (Must match Python training!)
const float MEANS[] = { 25.8489, 71.6869, 1014.3396 };
const float STDS[]  = { 1.8326, 8.2414, 1.5094 };

#define HISTORY_STEPS 30
#define PREDICT_STEPS 60
#define FEATURES 3 

// TFLite Memory (60KB is safe for N16R8)
const int kArenaSize = 256 * 1024;
uint8_t tensorArena[kArenaSize];

// Hardware Objects
Adafruit_BME280 bme; 

// Circular Buffer for History
float historyBuffer[HISTORY_STEPS][FEATURES];
int bufferHead = 0;
bool bufferFilled = false;

// Function Prototypes
void addReading(float t, float h, float p);
void runForecast();

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("\n=== Weather Forecaster (MicroTFLite) ===");

  // 1. Initialize BME280
  Wire.begin(); // SDA=11, SCL=12 for Nano ESP32
  if (!bme.begin(0x76, &Wire)) { 
    Serial.println("Standard init failed. Trying forced ID...");
  Serial.println("BME280 Initialized.");
  }

  // 2. Initialize MicroTFLite
  // API: ModelInit(model_data, arena_pointer, arena_size)
  if (!ModelInit(weather_model_new, tensorArena, kArenaSize)) {
    Serial.println("Model Initialization Failed!");
    while(1);
  }
  Serial.println("Model Loaded Successfully.");
  
  // Optional: Print model info
  //Serial.print("Input Tensor Size: "); Serial.println(ModelGetInputSize(0));
  //Serial.print("Output Tensor Size: "); Serial.println(ModelGetOutputSize(0));

  // 3. Pre-fill Buffer with 30 mins of history
  Serial.println("Pre-filling buffer...");
  const float prefillData[30][3] = {
    { 25.3950, 75.5596, 1011.2584 }, { 25.1133, 74.9915, 1011.2528 },
    { 24.8733, 75.8122, 1011.2687 }, { 24.7517, 76.1501, 1011.2537 },
    { 24.6300, 76.4880, 1011.2388 }, { 24.5800, 76.9477, 1011.2078 },
    { 24.6000, 77.0991, 1011.2034 }, { 24.6150, 77.3452, 1011.2026 },
    { 24.6425, 77.2053, 1011.2025 }, { 24.6650, 76.7959, 1011.2147 },
    { 24.7233, 76.8454, 1011.1916 }, { 24.8225, 76.5620, 1011.2219 },
    { 24.8200, 76.3950, 1011.2301 }, { 24.8000, 76.1641, 1011.1937 },
    { 24.8850, 75.7163, 1011.2161 }, { 24.9000, 75.7031, 1011.2365 },
    { 24.9500, 75.5540, 1011.2335 }, { 24.9400, 75.3188, 1011.2362 },
    { 24.9750, 75.5537, 1011.2307 }, { 25.0167, 75.6143, 1011.1697 },
    { 24.9850, 75.1997, 1011.1671 }, { 24.9475, 75.3091, 1011.1870 },
    { 24.9333, 75.5120, 1011.1874 }, { 24.9750, 75.3403, 1011.1866 },
    { 24.8933, 75.0391, 1011.1830 }, { 24.8400, 74.8421, 1011.1948 },
    { 24.8750, 75.1914, 1011.1969 }, { 24.8933, 75.3994, 1011.1791 },
    { 24.8933, 75.5557, 1011.1844 }, { 24.9050, 75.9697, 1011.1694 }
  };

  for (int i = 0; i < 30; i++) {
    addReading(prefillData[i][0], prefillData[i][1], prefillData[i][2]);
  }
  
  Serial.println("Buffer Ready. Running Initial Forecast...");
  runForecast();
}

void loop() {
  static unsigned long lastReadingTime = 0;
  
  // Run every 60 seconds
  if (millis() - lastReadingTime > 60000) {
    lastReadingTime = millis();
    
    float t = bme.readTemperature();
    float h = bme.readHumidity();
    float p = bme.readPressure() / 100.0F;

    if (isnan(t)) return;

    Serial.print("\nLive: "); Serial.print(t); Serial.println("C");
    addReading(t, h, p);
    runForecast();
  }
}

void addReading(float t, float h, float p) {
  historyBuffer[bufferHead][0] = t;
  historyBuffer[bufferHead][1] = h;
  historyBuffer[bufferHead][2] = p;
  
  bufferHead++;
  if (bufferHead >= HISTORY_STEPS) {
    bufferHead = 0;
    bufferFilled = true;
  }
}

void runForecast() {
  // 1. Prepare Input Data (Flattened)
  // The library expects us to set inputs one by one or as a block.
  // For efficiency, we'll calculate values and set them.
  
  int readIdx = bufferHead; 
  if (!bufferFilled) readIdx = 0;

  for (int i = 0; i < HISTORY_STEPS; i++) {
    int bufPos = (readIdx + i) % HISTORY_STEPS;
    
    // Normalize
    float nT = (historyBuffer[bufPos][0] - MEANS[0]) / STDS[0];
    float nH = (historyBuffer[bufPos][1] - MEANS[1]) / STDS[1];
    float nP = (historyBuffer[bufPos][2] - MEANS[2]) / STDS[2];
    
    if (i == 0) {
       Serial.print("Debug Input [0]: ");
       Serial.print(nT); Serial.print(", ");
       Serial.print(nH); Serial.print(", ");
       Serial.println(nP);
    }
    
    // Set Input Tensor values (Index = step * features + feature_idx)
    ModelSetInput(nT, i * FEATURES + 0);
    ModelSetInput(nH, i * FEATURES + 1);
    ModelSetInput(nP, i * FEATURES + 2);
  }

  // 2. Run Inference
  if (!ModelRunInference()) {
    Serial.println("Inference Failed!");
    return;
  }

  // 3. Read Output
  Serial.println("--- FORECAST (MicroTFLite) ---");
  Serial.println("Min  Temp   Hum   Pres");
  
  for (int step = 9; step < PREDICT_STEPS; step += 10) {
    int baseIdx = step * FEATURES;
    
    // Get Output Tensor values
    float outT = ModelGetOutput(baseIdx + 0);
    float outH = ModelGetOutput(baseIdx + 1);
    float outP = ModelGetOutput(baseIdx + 2);

    // Denormalize
    float predT = (outT * STDS[0]) + MEANS[0];
    float predH = (outH * STDS[1]) + MEANS[1];
    float predP = (outP * STDS[2]) + MEANS[2];

    Serial.print("+"); Serial.print(step + 1); Serial.print("m ");
    Serial.print(predT, 1); Serial.print("  ");
    Serial.print(predH, 1); Serial.print("  ");
    Serial.println(predP, 1);
  }
  Serial.println("------------------------------");
}
