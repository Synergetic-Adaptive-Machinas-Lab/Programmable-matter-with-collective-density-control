#include "command.h"
#include <Adafruit_PWMServoDriver.h>

static constexpr int UART_TX_PIN = 0;
static constexpr int UART_RX_PIN = 1;

static constexpr uint8_t  BASE_ADDR = 0x40;

static constexpr uint32_t I2C_HZ = 400000;

static constexpr float    PCA_PWM_FREQ_HZ = 1000.0f;

static constexpr uint8_t STATUS_OK = 1;

static uint8_t seq4[UART_SEQ_BYTES];
static uint8_t gamma1[UART_GAMMA_BYTES];
static uint8_t packed256[UART_PAYLOAD_BYTES];
static uint8_t X[X_VALUES];
static uint8_t ack7[ACK_BYTES];

static Adafruit_PWMServoDriver* boards0[32];
static Adafruit_PWMServoDriver* boards1[32];

static void initPcaBus(Adafruit_PWMServoDriver* boards[32], TwoWire& wire, uint8_t base_addr) {
  for (int i = 0; i < 32; ++i) {
    const uint8_t addr = (uint8_t)(base_addr + i);
    boards[i] = new Adafruit_PWMServoDriver(addr, wire);
    boards[i]->begin();
    boards[i]->setPWMFreq(PCA_PWM_FREQ_HZ);
    delay(2);
  }
}

void setup() {

  Serial1.setTX(UART_TX_PIN);
  Serial1.setRX(UART_RX_PIN);

  Serial1.begin(460800);

  Wire.setSDA(4);
  Wire.setSCL(5);
  Wire.begin();

  Wire1.setSDA(6);
  Wire1.setSCL(7);
  Wire1.begin();

  Wire.setClock(I2C_HZ);
  Wire1.setClock(I2C_HZ);

  initPcaBus(boards0, Wire,  BASE_ADDR);
  initPcaBus(boards1, Wire1, BASE_ADDR);
  delay(3000);
  Serial.println("pico1 setup complete");
}

void loop() {

  readExactBytes(Serial1, seq4, UART_SEQ_BYTES);
  readExactBytes(Serial1, gamma1, UART_GAMMA_BYTES);
  readExactBytes(Serial1, packed256, UART_PAYLOAD_BYTES);

  const uint32_t seq   = rd_u32_le(&seq4[0]);
  const int8_t   gamma = (int8_t)gamma1[0];

  buildX(packed256, X);

  actionX(boards0, boards1, X, gamma);

  makeAck(ack7, seq, 1);
  writeExactBytes(Serial1, ack7, ACK_BYTES);
}
