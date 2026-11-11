#include "command.h"
#include <Adafruit_PWMServoDriver.h>
#define LED_PIN 25

static constexpr int UART_TX_PIN = 0;
static constexpr int UART_RX_PIN = 1;

static constexpr uint8_t  BASE_ADDR = 0x40;

static constexpr uint32_t I2C_HZ = 400000;

static constexpr float    PCA_PWM_FREQ_HZ = 1000.0f;

static constexpr uint8_t STATUS_OK            = 1;
static constexpr uint8_t STATUS_ERR_MAGIC     = 0;
static constexpr uint8_t STATUS_ERR_CRC       = 2;
static constexpr uint8_t STATUS_ERR_PICO1_ACK = 3;

static uint8_t hdr[HDR_BYTES];
static uint8_t gamma1[GAMMA_BYTES];
static uint8_t data512[DATA_BYTES];
static uint8_t crc2[CRC_BYTES];

static uint8_t checkBuf[HDR_BYTES + GAMMA_BYTES + DATA_BYTES];

static uint8_t X[X_VALUES];

static uint8_t ack7[ACK_BYTES];
static uint8_t pico1_status = 0;

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

  Serial.begin(921600);

  Serial1.setTX(UART_TX_PIN);
  Serial1.setRX(UART_RX_PIN);
  Serial1.begin(460800);

  pinMode(LED_PIN, OUTPUT);

  const uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 2000) {}

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
  Serial.println("pico2 setup complete");
}

void loop() {

  readExactBytes(Serial, hdr, HDR_BYTES);

  const uint16_t magic = rd_u16_le(&hdr[0]);
  const uint32_t seq   = rd_u32_le(&hdr[2]);

  if (magic != MAGIC) {

    readExactBytes(Serial, gamma1, GAMMA_BYTES);
    readExactBytes(Serial, data512, DATA_BYTES);
    readExactBytes(Serial, crc2, CRC_BYTES);

    makeAck(ack7, seq, STATUS_ERR_MAGIC);
    writeExactBytes(Serial, ack7, ACK_BYTES);
    return;
  }

  readExactBytes(Serial, gamma1, GAMMA_BYTES);
  readExactBytes(Serial, data512, DATA_BYTES);
  readExactBytes(Serial, crc2, CRC_BYTES);

  const int8_t gamma = (int8_t)gamma1[0];

  memcpy(checkBuf, hdr, HDR_BYTES);
  memcpy(checkBuf + HDR_BYTES, gamma1, GAMMA_BYTES);
  memcpy(checkBuf + HDR_BYTES + GAMMA_BYTES, data512, DATA_BYTES);

  const uint16_t crc_recv = rd_u16_le(&crc2[0]);
  const uint16_t crc_calc = crc16_ccitt(checkBuf, (HDR_BYTES + GAMMA_BYTES + DATA_BYTES), 0xFFFF);

  if (crc_recv != crc_calc) {
    makeAck(ack7, seq, STATUS_ERR_CRC);
    writeExactBytes(Serial, ack7, ACK_BYTES);
    return;
  }

  uint8_t uart_pkt[UART_SEQ_BYTES + UART_GAMMA_BYTES + UART_PAYLOAD_BYTES];
  wr_u32_le(&uart_pkt[0], seq);
  uart_pkt[UART_SEQ_BYTES] = gamma1[0];
  memcpy(&uart_pkt[UART_SEQ_BYTES + UART_GAMMA_BYTES], data512, UART_PAYLOAD_BYTES);

  writeExactBytes(Serial1, uart_pkt, (UART_SEQ_BYTES + UART_GAMMA_BYTES + UART_PAYLOAD_BYTES));

  buildX(data512 + DATA_HALF, X);

  actionX(boards0, boards1, X, gamma);

  const uint32_t ACK_TIMEOUT_US = 400000;

  bool ok = readAck(Serial1, seq, &pico1_status, ACK_TIMEOUT_US);
  if (!ok) {
    makeAck(ack7, seq, STATUS_ERR_PICO1_ACK);
    writeExactBytes(Serial, ack7, ACK_BYTES);
    return;
  }

  const uint8_t final_status = (pico1_status == 1) ? STATUS_OK : pico1_status;

  makeAck(ack7, seq, final_status);
  writeExactBytes(Serial, ack7, ACK_BYTES);
}
