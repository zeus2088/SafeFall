/* Includes ---------------------------------------------------------------- */
#include <SafeFall2_inferencing.h>
#include <Arduino_LSM9DS1.h> 
#include <ArduinoBLE.h>

/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2    9.80665f
#define MAX_ACCEPTED_RANGE  2.0f 

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static uint32_t run_inference_every_ms = 800;
static rtos::Thread inference_thread(osPriorityLow);
static float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };
static float inference_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

int buzzer = 3;
unsigned long fallDetectedTime = 0;
bool fallDetected = false;

// BLE service and characteristic
BLEService fallDetectionService("180D");
BLECharacteristic fallDetectionCharacteristic("2A37", BLERead | BLENotify, 1);

/* Forward declaration */
void run_inference_background();

/**
* @brief      Arduino setup function
*/
void setup() {
    Serial.begin(115200);

    // Initialize IMU
    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1);
    }
    Serial.println("IMU initialized");

    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 3) {
        Serial.println("ERR: EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME should be equal to 3 (the 3 sensor axes)");
        return;
    }

    pinMode(buzzer, OUTPUT);

    // Initialize BLE
    if (!BLE.begin()) {
        Serial.println("Starting BLE failed!");
        while (1);
    }

    // Set BLE device name and service
    BLE.setLocalName("Nano33BLE-FallDetector");
    BLE.setAdvertisedService(fallDetectionService);

    // Add characteristic to the service
    fallDetectionService.addCharacteristic(fallDetectionCharacteristic);

    // Add service
    BLE.addService(fallDetectionService);

    // Start advertising
    BLE.advertise();

    Serial.println("BLE device active, waiting for connections...");

    // Start the inference thread
    inference_thread.start(mbed::callback(&run_inference_background));
}

/**
 * @brief Return the sign of the number
 * 
 * @param number 
 * @return int 1 if positive (or 0) -1 if negative
 */
float ei_get_sign(float number) {
    return (number >= 0.0) ? 1.0 : -1.0;
}

void buzzerOn() { 
  tone(buzzer, 1000);
}

void buzzerOff() { 
  noTone(buzzer);
}

/**
 * @brief      Run inferencing in the background.
 */
void run_inference_background() {
    delay((EI_CLASSIFIER_INTERVAL_MS * EI_CLASSIFIER_RAW_SAMPLE_COUNT) + 100);

    ei_classifier_smooth_t smooth;
    ei_classifier_smooth_init(&smooth, 10, 7, 0.8, 0.3);

    while (1) {
        memcpy(inference_buffer, buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE * sizeof(float));
        signal_t signal;
        int err = numpy::signal_from_buffer(inference_buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);

        if (err != 0) {
            Serial.println("Failed to create signal from buffer");
            return;
        }

        ei_impulse_result_t result = { 0 };
        err = run_classifier(&signal, &result, debug_nn);

        if (err != EI_IMPULSE_OK) {
            Serial.println("Failed to run classifier");
            return;
        }

        const char* prediction = ei_classifier_smooth_update(&smooth, &result);
        Serial.print("Prediction: ");
        Serial.println(prediction);

        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            Serial.print(result.classification[ix].label);
            Serial.print(": ");
            Serial.println(result.classification[ix].value, 5);

            if ((strcmp(result.classification[ix].label, "Fall") == 0) && (result.classification[ix].value > 0.9)) {
                if (!fallDetected) {
                    tone(buzzer, 1000);
                    fallDetectedTime = millis();
                    fallDetected = true;
                }
                fallDetectionCharacteristic.writeValue((uint8_t)1);
            }

            if ((strcmp(result.classification[ix].label, "Stand") == 0) && (result.classification[ix].value > 0.7)) {
                noTone(buzzer);
                fallDetectionCharacteristic.writeValue((uint8_t)0);
                fallDetected = false;
            }
        }

        if (fallDetected && (millis() - fallDetectedTime >= 5000)) {
            noTone(buzzer);
            fallDetected = false;
        }

        delay(run_inference_every_ms);
    }

    ei_classifier_smooth_free(&smooth);
}

/**
* @brief      Get data and run inferencing
*
* @param[in]  debug  Get debug info if true
*/
void loop() {
    while (1) {
        uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

        // Roll the buffer -3 points so we can overwrite the last one
        numpy::roll(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, -3);

        // Read to the end of the buffer
        IMU.readAcceleration(
            buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 3],
            buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 2],
            buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 1]
        );

        for (int i = 0; i < 3; i++) {
            if (fabs(buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 3 + i]) > MAX_ACCEPTED_RANGE) {
                buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 3 + i] = ei_get_sign(buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 3 + i]) * MAX_ACCEPTED_RANGE;
            }
        }

        buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 3] *= CONVERT_G_TO_MS2;
        buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 2] *= CONVERT_G_TO_MS2;
        buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 1] *= CONVERT_G_TO_MS2;

        uint64_t time_to_wait = next_tick - micros();
        delay((int)floor((float)time_to_wait / 1000.0f));
        delayMicroseconds(time_to_wait % 1000);

        // Poll for BLE events
        BLE.poll();
        delay(10);
    }
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_ACCELEROMETER
#error "Invalid model for current sensor"
#endif
