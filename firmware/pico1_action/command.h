#pragma once

#include <Arduino.h>
#include <stdint.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

static constexpr uint16_t MAGIC     = 0x55AA;
static constexpr uint16_t ACK_MAGIC = 0x55AA;

static constexpr int HDR_BYTES   = 6;
static constexpr int GAMMA_BYTES = 1;
static constexpr int CRC_BYTES   = 2;
static constexpr int ACK_BYTES   = 7;

static constexpr int DATA_BYTES   = 512;
static constexpr int DATA_HALF    = DATA_BYTES / 2;
static constexpr int X_VALUES     = DATA_HALF * 2;

static constexpr int FRAME_BYTES  = HDR_BYTES + GAMMA_BYTES + DATA_BYTES + CRC_BYTES;

static constexpr int UART_SEQ_BYTES      = 4;
static constexpr int UART_GAMMA_BYTES    = 1;
static constexpr int UART_PAYLOAD_BYTES  = 256;

uint16_t rd_u16_le(const uint8_t* p);
uint32_t rd_u32_le(const uint8_t* p);
void     wr_u16_le(uint8_t* p, uint16_t v);
void     wr_u32_le(uint8_t* p, uint32_t v);

void readExactBytes(Stream& s, uint8_t* dst, int n);
void writeExactBytes(Stream& s, const uint8_t* src, int n);

uint16_t crc16_ccitt(const uint8_t* data, int n, uint16_t init = 0xFFFF);

void buildX(const uint8_t* packed256, uint8_t* X512);

void actionX(Adafruit_PWMServoDriver* boards0[32], Adafruit_PWMServoDriver* boards1[32], const uint8_t* X512, int8_t gamma);

void makeAck(uint8_t* ack7, uint32_t seq, uint8_t status);

bool readAck(Stream& s, uint32_t expected_seq, uint8_t* out_status, uint32_t timeout_us);
